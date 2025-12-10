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
| 2025-12-09 | [Secure and Privacy-Preserving Federated Learning for Next-Generation Underground Mine Safety](http://arxiv.org/abs/2512.08862v1) | Mohamed Elmahallawy, Sanjay Madria et al. | Underground mining operations depend on sensor networks to monitor critical parameters such as temperature, gas concentration, and miner movement, enabling timely hazard detection and safety decisions. However, transmitting raw sensor data to a centralized server for machine learning (ML) model training raises serious privacy and security concerns. Federated Learning (FL) offers a promising alternative by enabling decentralized model training without exposing sensitive local data. Yet, applying FL in underground mining presents unique challenges: (i) Adversaries may eavesdrop on shared model updates to launch model inversion or membership inference attacks, compromising data privacy and operational safety; (ii) Non-IID data distributions across mines and sensor noise can hinder model convergence. To address these issues, we propose FedMining--a privacy-preserving FL framework tailored for underground mining. FedMining introduces two core innovations: (1) a Decentralized Functional Encryption (DFE) scheme that keeps local models encrypted, thwarting unauthorized access and inference attacks; and (2) a balancing aggregation mechanism to mitigate data heterogeneity and enhance convergence. Evaluations on real-world mining datasets demonstrate FedMining's ability to safeguard privacy while maintaining high model accuracy and achieving rapid convergence with reduced communication and computation overhead. These advantages make FedMining both secure and practical for real-time underground safety monitoring. |
| 2025-12-09 | [A Methodology for Quantitative AI Risk Modeling](http://arxiv.org/abs/2512.08844v1) | Malcolm Murray, Steve Barrett et al. | Although general-purpose AI systems offer transformational opportunities in science and industry, they simultaneously raise critical concerns about safety, misuse, and potential loss of control. Despite these risks, methods for assessing and managing them remain underdeveloped. Effective risk management requires systematic modeling to characterize potential harms, as emphasized in frameworks such as the EU General-Purpose AI Code of Practice. This paper advances the risk modeling component of AI risk management by introducing a methodology that integrates scenario building with quantitative risk estimation, drawing on established approaches from other high-risk industries. Our methodology models risks through a six-step process: (1) defining risk scenarios, (2) decomposing them into quantifiable parameters, (3) quantifying baseline risk without AI models, (4) identifying key risk indicators such as benchmarks, (5) mapping these indicators to model parameters to estimate LLM uplift, and (6) aggregating individual parameters into risk estimates that enable concrete claims (e.g., X% probability of >\$Y in annual cyber damages). We examine the choices that underlie our methodology throughout the article, with discussions of strengths, limitations, and implications for future research. Our methodology is designed to be applicable to key systemic AI risks, including cyber offense, biological weapon development, harmful manipulation, and loss-of-control, and is validated through extensive application in LLM-enabled cyber offense. Detailed empirical results and cyber-specific insights are presented in a companion paper. |
| 2025-12-09 | [Multicalibration for LLM-based Code Generation](http://arxiv.org/abs/2512.08810v1) | Viola Campos, Robin Kuschnereit et al. | As AI-based code generation becomes widespread, researchers are investigating the calibration of code LLMs - ensuring their confidence scores faithfully represent the true likelihood of code correctness. To do so, we investigate multicalibration, which can capture additional factors about a coding problem, such as complexity, code length, or programming language used. We study four multicalibration approaches on three function synthesis benchmarks, using latest-generation code LLMs (Qwen3 Coder, GPT-OSS, DeepSeek-R1-Distill). Our results demonstrate that multicalibration can yield distinct improvements over both uncalibrated token likelihoods (+1.03 in skill score) and baseline calibrations (+0.37 in skill score). We study the influence of the aforementioned factors in ablations, and make our dataset (consisting of code generations, likelihoods, and correctness labels) available for future research on code LLM calibration. |
| 2025-12-09 | [Commissioning of an experiment for thermodynamic and spectroscopic studies of hydrogen isotopologues at cryogenic conditions](http://arxiv.org/abs/2512.08788v1) | Joshua Kohpeiß, Dominic Batzler et al. | To study thermodynamic properties and dynamic phase space behavior of hydrogen isotopologues (Q$_2$) at cryogenic temperatures and at high density, the Tritium Absorption InfraRed Spectroscopy 2 (T$_2$ApIR) experiment has been set up and commissioned at Tritium Laboratory Karlsruhe (TLK). In the frame of the experiment, Q$_2$ behavior in different phases, ortho/para states, temperatures (10 K - 300 K) and pressures (up to 2.5 bar a) will be investigated with optical methods, infrared and Raman spectroscopy. The facility consists of a fully tritium compatible cryostat, which includes an optical cell, ortho/para converter and windows for optical and spectroscopic studies. The cryostat can be cooled below the H$_2$ triple point by a two-stage cryocooler and contains openings in the cryogenic shielding for the optical access. The challenge of combining these scientific requirements in a design with high amounts of tritium (14 g), in a limited space, all while maintaining the TLK safety philosophy was solved by the presented design. The experiment is ready to be fully integrated into the TLK closed loop tritium infrastructure. This contribution reports a comprehensive overview of the commissioning phase of the experimental facility and the results of the first commissioning experiments, including cryogenic performance tests, commissioning experiments with non-radioactive gases, and tests of the analytical instruments. |
| 2025-12-09 | [A Practical Guide for Designing, Developing, and Deploying Production-Grade Agentic AI Workflows](http://arxiv.org/abs/2512.08769v1) | Eranga Bandara, Ross Gore et al. | Agentic AI marks a major shift in how autonomous systems reason, plan, and execute multi-step tasks. Unlike traditional single model prompting, agentic workflows integrate multiple specialized agents with different Large Language Models(LLMs), tool-augmented capabilities, orchestration logic, and external system interactions to form dynamic pipelines capable of autonomous decision-making and action. As adoption accelerates across industry and research, organizations face a central challenge: how to design, engineer, and operate production-grade agentic AI workflows that are reliable, observable, maintainable, and aligned with safety and governance requirements. This paper provides a practical, end-to-end guide for designing, developing, and deploying production-quality agentic AI systems. We introduce a structured engineering lifecycle encompassing workflow decomposition, multi-agent design patterns, Model Context Protocol(MCP), and tool integration, deterministic orchestration, Responsible-AI considerations, and environment-aware deployment strategies. We then present nine core best practices for engineering production-grade agentic AI workflows, including tool-first design over MCP, pure-function invocation, single-tool and single-responsibility agents, externalized prompt management, Responsible-AI-aligned model-consortium design, clean separation between workflow logic and MCP servers, containerized deployment for scalable operations, and adherence to the Keep it Simple, Stupid (KISS) principle to maintain simplicity and robustness. To demonstrate these principles in practice, we present a comprehensive case study: a multimodal news-analysis and media-generation workflow. By combining architectural guidance, operational patterns, and practical implementation insights, this paper offers a foundational reference to build robust, extensible, and production-ready agentic AI workflows. |
| 2025-12-09 | [IoT-based Cost-Effective Fruit Quality Monitoring System using Electronic Nose](http://arxiv.org/abs/2512.08753v1) | Anindya Bhattacharjee, Nittya Ananda Biswas et al. | Post-harvest losses due to subjective quality assessment cause significant damage to the economy and food safety, especially in countries like Bangladesh. To mitigate such damages, objective decision-making backed by scientific methods is necessary. An IoT-based, cost-effective quality monitoring system can provide a solution by going beyond subjective quality monitoring and decision-making practices. Here, we propose a low-power, cost-effective fruit quality monitoring system with an array of MQ gas sensors, which can be used as an electronic nose. We track the volatile gas emissions, specifically ethanol, methane, and ammonia, encompassing both ripening and decomposition for a set of bananas. Based on the gas concentration thresholds, we develop a mathematical model to accurately assess fruit quality. We also integrate this information into a dashboard for prompt decision-making and monitoring to make it useful to the farmers. This approach has the potential to reduce economic losses, enhance food safety, and provide scalable solutions for the supply chain. |
| 2025-12-09 | [Towards Foundation Models with Native Multi-Agent Intelligence](http://arxiv.org/abs/2512.08743v1) | Shuyue Hu, Haoyang Yan et al. | Foundation models (FMs) are increasingly assuming the role of the "brain" of AI agents. While recent efforts have begun to equip FMs with native single-agent abilities -- such as GUI interaction or integrated tool use -- we argue that the next frontier is endowing FMs with native multi-agent intelligence. We identify four core capabilities of FMs in multi-agent contexts: understanding, planning, efficient communication, and adaptation. Contrary to assumptions about the spontaneous emergence of such abilities, we provide extensive empirical evidence across 41 large language models showing that strong single-agent performance alone does not automatically yield robust multi-agent intelligence. To address this gap, we outline key research directions -- spanning dataset construction, evaluation, training paradigms, and safety considerations -- for building FMs with native multi-agent intelligence. |
| 2025-12-09 | [The Role of Risk Modeling in Advanced AI Risk Management](http://arxiv.org/abs/2512.08723v1) | Chloé Touzet, Henry Papadatos et al. | Rapidly advancing artificial intelligence (AI) systems introduce novel, uncertain, and potentially catastrophic risks. Managing these risks requires a mature risk-management infrastructure whose cornerstone is rigorous risk modeling. We conceptualize AI risk modeling as the tight integration of (i) scenario building$-$causal mapping from hazards to harms$-$and (ii) risk estimation$-$quantifying the likelihood and severity of each pathway. We review classical techniques such as Fault and Event Tree Analyses, FMEA/FMECA, STPA and Bayesian networks, and show how they can be adapted to advanced AI. A survey of emerging academic and industry efforts reveals fragmentation: capability benchmarks, safety cases, and partial quantitative studies are valuable but insufficient when divorced from comprehensive causal scenarios. Comparing the nuclear, aviation, cybersecurity, financial, and submarine domains, we observe that every sector combines deterministic guarantees for unacceptable events with probabilistic assessments of the broader risk landscape. We argue that advanced-AI governance should adopt a similar dual approach and that verifiable, provably-safe AI architectures are urgently needed to supply deterministic evidence where current models are the result of opaque end-to-end optimization procedures rather than specified by hand. In one potential governance-ready framework, developers conduct iterative risk modeling and regulators compare the results with predefined societal risk tolerance thresholds. The paper provides both a methodological blueprint and opens a discussion on the best way to embed sound risk modeling at the heart of advanced-AI risk management. |
| 2025-12-09 | [Non Normalized Shared-Constraint Dynamic Games for Human-Robot Collaboration with Asymmetric Responsibility](http://arxiv.org/abs/2512.08688v1) | Mark Pustilnik, Francesco Borrelli | This paper proposes a dynamic game formulation for cooperative human-robot navigation in shared workspaces with obstacles, where the human and robot jointly satisfy shared safety constraints while pursuing a common task. A key contribution is the introduction of a non-normalized equilibrium structure for the shared constraints. This structure allows the two agents to contribute different levels of effort towards enforcing safety requirements such as collision avoidance and inter-players spacing. We embed this non-normalized equilibrium into a receding-horizon optimal control scheme. |
| 2025-12-09 | [Chain-of-Image Generation: Toward Monitorable and Controllable Image Generation](http://arxiv.org/abs/2512.08645v1) | Young Kyung Kim, Oded Schlesinger et al. | While state-of-the-art image generation models achieve remarkable visual quality, their internal generative processes remain a "black box." This opacity limits human observation and intervention, and poses a barrier to ensuring model reliability, safety, and control. Furthermore, their non-human-like workflows make them difficult for human observers to interpret. To address this, we introduce the Chain-of-Image Generation (CoIG) framework, which reframes image generation as a sequential, semantic process analogous to how humans create art. Similar to the advantages in monitorability and performance that Chain-of-Thought (CoT) brought to large language models (LLMs), CoIG can produce equivalent benefits in text-to-image generation. CoIG utilizes an LLM to decompose a complex prompt into a sequence of simple, step-by-step instructions. The image generation model then executes this plan by progressively generating and editing the image. Each step focuses on a single semantic entity, enabling direct monitoring. We formally assess this property using two novel metrics: CoIG Readability, which evaluates the clarity of each intermediate step via its corresponding output; and Causal Relevance, which quantifies the impact of each procedural step on the final generated image. We further show that our framework mitigates entity collapse by decomposing the complex generation task into simple subproblems, analogous to the procedural reasoning employed by CoT. Our experimental results indicate that CoIG substantially enhances quantitative monitorability while achieving competitive compositional robustness compared to established baseline models. The framework is model-agnostic and can be integrated with any image generation model. |
| 2025-12-09 | [The SMART+ Framework for AI Systems](http://arxiv.org/abs/2512.08592v1) | Laxmiraju Kandikatla, Branislav Radeljic | Artificial Intelligence (AI) systems are now an integral part of multiple industries. In clinical research, AI supports automated adverse event detection in clinical trials, patient eligibility screening for protocol enrollment, and data quality validation. Beyond healthcare, AI is transforming finance through real-time fraud detection, automated loan risk assessment, and algorithmic decision-making. Similarly, in manufacturing, AI enables predictive maintenance to reduce equipment downtime, enhances quality control through computer-vision inspection, and optimizes production workflows using real-time operational data. While these technologies enhance operational efficiency, they introduce new challenges regarding safety, accountability, and regulatory compliance. To address these concerns, we introduce the SMART+ Framework - a structured model built on the pillars of Safety, Monitoring, Accountability, Reliability, and Transparency, and further enhanced with Privacy & Security, Data Governance, Fairness & Bias, and Guardrails. SMART+ offers a practical, comprehensive approach to evaluating and governing AI systems across industries. This framework aligns with evolving mechanisms and regulatory guidance to integrate operational safeguards, oversight procedures, and strengthened privacy and governance controls. SMART+ demonstrates risk mitigation, trust-building, and compliance readiness. By enabling responsible AI adoption and ensuring auditability, SMART+ provides a robust foundation for effective AI governance in clinical research. |
| 2025-12-09 | [Disrupting Hierarchical Reasoning: Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models](http://arxiv.org/abs/2512.08503v1) | Jiaming Zhang, Che Wang et al. | Multi-modal large reasoning models (MLRMs) pose significant privacy risks by inferring precise geographic locations from personal images through hierarchical chain-of-thought reasoning. Existing privacy protection techniques, primarily designed for perception-based models, prove ineffective against MLRMs' sophisticated multi-step reasoning processes that analyze environmental cues. We introduce \textbf{ReasonBreak}, a novel adversarial framework specifically designed to disrupt hierarchical reasoning in MLRMs through concept-aware perturbations. Our approach is founded on the key insight that effective disruption of geographic reasoning requires perturbations aligned with conceptual hierarchies rather than uniform noise. ReasonBreak strategically targets critical conceptual dependencies within reasoning chains, generating perturbations that invalidate specific inference steps and cascade through subsequent reasoning stages. To facilitate this approach, we contribute \textbf{GeoPrivacy-6K}, a comprehensive dataset comprising 6,341 ultra-high-resolution images ($\geq$2K) with hierarchical concept annotations. Extensive evaluation across seven state-of-the-art MLRMs (including GPT-o3, GPT-5, Gemini 2.5 Pro) demonstrates ReasonBreak's superior effectiveness, achieving a 14.4\% improvement in tract-level protection (33.8\% vs 19.4\%) and nearly doubling block-level protection (33.5\% vs 16.8\%). This work establishes a new paradigm for privacy protection against reasoning-based threats. |
| 2025-12-09 | [Developing Distance-Aware Uncertainty Quantification Methods in Physics-Guided Neural Networks for Reliable Bearing Health Prediction](http://arxiv.org/abs/2512.08499v1) | Waleed Razzaq, Yun-Bo Zhao | Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA dataset and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise. |
| 2025-12-09 | [Prospect Theory in Physical Human-Robot Interaction: A Pilot Study of Probability Perception](http://arxiv.org/abs/2512.08481v1) | Yixiang Lin, Tiancheng Yang et al. | Understanding how humans respond to uncertainty is critical for designing safe and effective physical human-robot interaction (pHRI), as physically working with robots introduces multiple sources of uncertainty, including trust, comfort, and perceived safety. Conventional pHRI control frameworks typically build on optimal control theory, which assumes that human actions minimize a cost function; however, human behavior under uncertainty often departs from such optimal patterns. To address this gap, additional understanding of human behavior under uncertainty is needed. This pilot study implemented a physically coupled target-reaching task in which the robot delivered assistance or disturbances with systematically varied probabilities (10\% to 90\%). Analysis of participants' force inputs and decision-making strategies revealed two distinct behavioral clusters: a "trade-off" group that modulated their physical responses according to disturbance likelihood, and an "always-compensate" group characterized by strong risk aversion irrespective of probability. These findings provide empirical evidence that human decision-making in pHRI is highly individualized and that the perception of probability can differ to its true value. Accordingly, the study highlights the need for more interpretable behavioral models, such as cumulative prospect theory (CPT), to more accurately capture these behaviors and inform the design of future adaptive robot controllers. |
| 2025-12-09 | [Trans-Arctic route feasibility on a pan-Arctic grid under bathymetric and sea-ice constraints](http://arxiv.org/abs/2512.08434v1) | Abdella Mohamed, Xiangyu Hu | Climate driven reductions in Arctic sea ice have renewed interest in trans Arctic shipping, but adoption remains limited by basic questions of route feasibility, safety and excess distance. Existing studies mostly compare idealised great circle shortcuts or use full weather routing systems, leaving a gap for simple basin scale diagnostics on realistic bathymetry and sea ice. We develop an offline graph based framework on a 0.5 degree pan Arctic grid that combines GEBCO 2024 bathymetry with a summer 2018 Arctic sea ice reanalysis from the Copernicus Marine Environment Monitoring Service (CMEMS). An A* pathfinding algorithm is applied to a canonical Europe Asia origin destination pair to quantify route availability and route length inflation relative to a great circle. Enforcing sea only feasibility increases route length by about 10 percent before depth and ice constraints are applied. Depth thresholds representative of under keel clearance (hmin = 20-50 m) remove up to roughly 15 percent of the sea mask but preserve a trans Arctic connection for hmin = 20 m. Summer sea ice exerts a strong seasonal control: continuous ice safe routes emerge only from mid August, with distances inflated by roughly 20-25 percent even in late summer. When depth and ice constraints are imposed jointly, only about 75 percent of sea cells remain safe and no continuous joint safe trans Arctic route exists in the tested season. The framework provides a basin scale screening tool for Arctic shipping and a baseline for forecast driven, multi objective routing studies. |
| 2025-12-09 | [Formation and Investigation of Cooperative Platooning at the Early Stage of Connected and Automated Vehicles Deployment](http://arxiv.org/abs/2512.08298v1) | Zeyu Mu, Sergei S. Avedisov et al. | Cooperative platooning, enabled by cooperative adaptive cruise control (CACC), is a cornerstone technology for connected automated vehicles (CAVs), offering significant improvements in safety, comfort, and traffic efficiency over traditional adaptive cruise control (ACC). This paper addresses a key challenge in the initial deployment phase of CAVs: the limited benefits of cooperative platooning due to the sparse distribution of CAVs on the road. To overcome this limitation, we propose an innovative control framework that enhances cooperative platooning in mixed traffic environments. Two techniques are utilized: (1) a mixed cooperative platooning strategy that integrates CACC with unconnected vehicles (CACCu), and (2) a strategic lane-change decision model designed to facilitate safe and efficient lane changes for platoon formation. Additionally, a surrounding vehicle identification system is embedded in the framework to enable CAVs to effectively identify and select potential platooning leaders. Simulation studies across various CV market penetration rates (MPRs) show that incorporating CACCu systems significantly improves safety, comfort, and traffic efficiency compared to existing systems with only CACC and ACC systems, even at CV penetration as low as 10%. The maximized platoon formation increases by up to 24%, accompanied by an 11% reduction in acceleration and a 7% decrease in fuel consumption. Furthermore, the strategic lane-change model enhances CAV performance, achieving notable improvements between 6% and 60% CV penetration, without adversely affecting overall traffic flow. |
| 2025-12-09 | [Systematization of Knowledge: Security and Safety in the Model Context Protocol Ecosystem](http://arxiv.org/abs/2512.08290v1) | Shiva Gaire, Srijan Gyawali et al. | The Model Context Protocol (MCP) has emerged as the de facto standard for connecting Large Language Models (LLMs) to external data and tools, effectively functioning as the "USB-C for Agentic AI." While this decoupling of context and execution solves critical interoperability challenges, it introduces a profound new threat landscape where the boundary between epistemic errors (hallucinations) and security breaches (unauthorized actions) dissolves. This Systematization of Knowledge (SoK) aims to provide a comprehensive taxonomy of risks in the MCP ecosystem, distinguishing between adversarial security threats (e.g., indirect prompt injection, tool poisoning) and epistemic safety hazards (e.g., alignment failures in distributed tool delegation). We analyze the structural vulnerabilities of MCP primitives, specifically Resources, Prompts, and Tools, and demonstrate how "context" can be weaponized to trigger unauthorized operations in multi-agent environments. Furthermore, we survey state-of-the-art defenses, ranging from cryptographic provenance (ETDI) to runtime intent verification, and conclude with a roadmap for securing the transition from conversational chatbots to autonomous agentic operating systems. |
| 2025-12-09 | [Semantic-Metric Bayesian Risk Fields: Learning Robot Safety from Human Videos with a VLM Prior](http://arxiv.org/abs/2512.08233v1) | Timothy Chen, Marcus Dominguez-Kuhne et al. | Humans interpret safety not as a binary signal but as a continuous, context- and spatially-dependent notion of risk. While risk is subjective, humans form rational mental models that guide action selection in dynamic environments. This work proposes a framework for extracting implicit human risk models by introducing a novel, semantically-conditioned and spatially-varying parametrization of risk, supervised directly from safe human demonstration videos and VLM common sense. Notably, we define risk through a Bayesian formulation. The prior is furnished by a pretrained vision-language model. In order to encourage the risk estimate to be more human aligned, a likelihood function modulates the prior to produce a relative metric of risk. Specifically, the likelihood is a learned ViT that maps pretrained features, to pixel-aligned risk values. Our pipeline ingests RGB images and a query object string, producing pixel-dense risk images. These images that can then be used as value-predictors in robot planning tasks or be projected into 3D for use in conventional trajectory optimization to produce human-like motion. This learned mapping enables generalization to novel objects and contexts, and has the potential to scale to much larger training datasets. In particular, the Bayesian framework that is introduced enables fast adaptation of our model to additional observations or common sense rules. We demonstrate that our proposed framework produces contextual risk that aligns with human preferences. Additionally, we illustrate several downstream applications of the model; as a value learner for visuomotor planners or in conjunction with a classical trajectory optimization algorithm. Our results suggest that our framework is a significant step toward enabling autonomous systems to internalize human-like risk. Code and results can be found at https://riskbayesian.github.io/bayesian_risk/. |
| 2025-12-09 | [Primal-dual policy learning for mean-field stochastic LQR problem](http://arxiv.org/abs/2512.08205v1) | Xiushan Jiang, Dong Wang et al. | Integrating data-driven techniques with mechanism-driven insights has recently gained popularity as a powerful learning approach to solving traditional LQR problems for designing intelligent controllers in complex dynamic systems. However, the theoretical understanding of various reinforcement learning algorithms needs further exploration to enhance their efficiency and safety. In this article, by means of primal-dual optimization tools, we study the partially model-free design of the mean-field stochastic LQR (MF-SLQR) controller using a policy learning approach. Firstly, by designing appropriate optimizing variables, the considered MF-SLQR problem is transformed into a new static nonconvex constrained optimization problem with equivalence preserved in certain senses. After that, the equivalent formulation of the duality results is constructed via finding the solution of the generalized Lyapunov equation. Then, the strong duality is analyzed, based on which we establish a primal-dual algorithm by Karush-Kuhn-Tucker conditions. More importantly, a partially model-free implementation is also presented, which has a direct connection with the classical policy iteration algorithm. Finally, we use a high-dimensional example to validate our methods. |
| 2025-12-09 | [Evaluating Vulnerabilities of Connected Vehicles Under Cyber Attacks by Attack-Defense Tree](http://arxiv.org/abs/2512.08204v1) | Muhammad Baqer Mollah, Honggang Wang et al. | Connected vehicles represent a key enabler of intelligent transportation systems, where vehicles are equipped with advanced communication, sensing, and computing technologies to interact not only with one another but also with surrounding infrastructures and the environment. Through continuous data exchange, such vehicles are capable of enhancing road safety, improving traffic efficiency, and ensuring more reliable mobility services. Further, when these capabilities are integrated with advanced automation technologies, the concept essentially evolves into connected and autonomous vehicles (CAVs). While connected vehicles primarily focus on seamless information sharing, autonomous vehicles are mainly dependent on advanced perception, decision-making, and control mechanisms to operate with minimal or without human intervention. However, as a result of connectivity, an adversary with malicious intentions might be able to compromise successfully by breaching the system components of CAVs. In this paper, we present an attack-tree based methodology for evaluating cyber security vulnerabilities in CAVs. In particular, we utilize the attack-defense tree formulation to systematically assess attack-leaf vulnerabilities, and before analyzing the vulnerability indices, we also define a measure of vulnerabilities, which is based on existing cyber security threats and corresponding defensive countermeasures. |

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



