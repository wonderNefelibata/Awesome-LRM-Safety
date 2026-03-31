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
| 2026-03-30 | [$\mathcal{L}_1$-Certified Distributionally Robust Planning for Safety-Constrained Adaptive Control](http://arxiv.org/abs/2603.28758v1) | Astghik Hakobyan, Amaras Nazarians et al. | Safe operation of autonomous systems requires robustness to both model uncertainty and uncertainty in the environment. We propose a hierarchical framework for stochastic nonlinear systems that integrates distributionally robust model predictive control (DR-MPC) with $\mathcal{L}_1$-adaptive control. The key idea is to use the $\mathcal{L}_1$ adaptive controller's online distributional certificates that bound the Wasserstein distance between nominal and true state distributions, thereby certifying the ambiguity sets used for planning without requiring distribution samples. Environment uncertainty is captured via data-driven ambiguity sets constructed from finite samples. These are incorporated into a DR-MPC planner enforcing distributionally robust chance constraints over a receding horizon. Using Wasserstein duality, the resulting problem admits tractable reformulations and a sample-based implementation. We show theoretically and via numerical experimentation that our framework ensures certifiable safety in the presence of simultaneous system and environment uncertainties. |
| 2026-03-30 | [C2RustXW: Program-Structure-Aware C-to-Rust Translation via Program Analysis and LLM](http://arxiv.org/abs/2603.28686v1) | Yanyan Yan, Yang Feng et al. | The growing adoption of Rust for its memory safety and performance has increased the demand for effective migration of legacy C codebases. However, existing rule-based translators (e.g., \ctorust) often generate verbose, non-idiomatic code that preserves unsafe C semantics, limiting readability, maintainability, and practical adoption. Moreover, manual post-processing of such outputs is labor-intensive and rarely yields high-quality Rust code, posing a significant barrier to large-scale migration. To address these limitations, we present \tool, a program-structure-aware C-to-Rust translation approach that integrates program analysis with Large Language Models (LLMs). \tool extracts the multi-level program structure, including global symbols, function dependencies, and control- and data-flow information, and encodes these as structured textual representations injected into LLM prompts to guide translation and repair. Based on this design, \tool performs dependency-aware translation and adopts a multi-stage repair pipeline that combines rule-based and structure-guided LLM-based techniques to ensure syntactic correctness. For semantic correctness, \tool further integrates execution-based validation with structure-guided reasoning to localize and repair behavioral inconsistencies. Experimental results show that \tool achieves 100\% syntactic correctness on CodeNet and 97.78\% on GitHub, while significantly reducing code size (up to 43.70\%) and unsafe usage (to 5.75\%). At the project level, \tool achieves perfect syntactic correctness and an average semantic correctness of 78.87\%, demonstrating its effectiveness for practical and scalable C-to-Rust migration. |
| 2026-03-30 | [Information-Theoretic Limits of Safety Verification for Self-Improving Systems](http://arxiv.org/abs/2603.28650v1) | Arsenios Scrivens | Can a safety gate permit unbounded beneficial self-modification while maintaining bounded cumulative risk? We formalize this question through dual conditions -- requiring sum delta_n < infinity (bounded risk) and sum TPR_n = infinity (unbounded utility) -- and establish a theory of their (in)compatibility.   Classification impossibility (Theorem 1): For power-law risk schedules delta_n = O(n^{-p}) with p > 1, any classifier-based gate under overlapping safe/unsafe distributions satisfies TPR_n <= C_alpha * delta_n^beta via Holder's inequality, forcing sum TPR_n < infinity. This impossibility is exponent-optimal (Theorem 3). A second independent proof via the NP counting method (Theorem 4) yields a 13% tighter bound without Holder's inequality.   Universal finite-horizon ceiling (Theorem 5): For any summable risk schedule, the exact maximum achievable classifier utility is U*(N, B) = N * TPR_NP(B/N), growing as exp(O(sqrt(log N))) -- subpolynomial. At N = 10^6 with budget B = 1.0, a classifier extracts at most U* ~ 87 versus a verifier's ~500,000.   Verification escape (Theorem 2): A Lipschitz ball verifier achieves delta = 0 with TPR > 0, escaping the impossibility. Formal Lipschitz bounds for pre-LayerNorm transformers under LoRA enable LLM-scale verification. The separation is strict. We validate on GPT-2 (d_LoRA = 147,456): conditional delta = 0 with TPR = 0.352. Comprehensive empirical validation is in the companion paper [D2]. |
| 2026-03-30 | [Toxicity Monitoring Rule for a Two-Cohort Phase II Clinical Trial with Bivariate Beta Prior](http://arxiv.org/abs/2603.28615v1) | Yu Wang, Aniko Szabo | Toxicity monitoring is essential in Phase II clinical trials to ensure participant safety. While monitoring rules are well-established for single-arm trials, two-cohort trials present unique challenges because toxicities are expected to be similar between cohorts but may still differ. Current approaches either monitor the two cohorts independently, which ignores their similarity, or pool them together as a single arm, which neglects heterogeneity between cohorts. We propose a Bayesian method based on a bivariate beta prior that provides a compromise between these two approaches. The marginal posterior distribution is derived as a mixture of beta distributions, enabling exact calculations of the proposed method's operating characteristics. Examples demonstrate that joint monitoring offers a balanced approach between the independent and pooled methods.   Keywords: Toxicity; Two-cohort; Phase II clinical trial; Monitoring rules; Bivariate Beta; Exact Operating characteristics |
| 2026-03-30 | [Detection of Adversarial Attacks in Robotic Perception](http://arxiv.org/abs/2603.28594v1) | Ziad Sharawy, Mohammad Nakshbandiand et al. | Deep Neural Networks (DNNs) achieve strong performance in semantic segmentation for robotic perception but remain vulnerable to adversarial attacks, threatening safety-critical applications. While robustness has been studied for image classification, semantic segmentation in robotic contexts requires specialized architectures and detection strategies. |
| 2026-03-30 | [Fine-Tuning Large Language Models for Cooperative Tactical Deconfliction of Small Unmanned Aerial Systems](http://arxiv.org/abs/2603.28561v1) | Iman Sharifi, Alex Zongo et al. | The growing deployment of small Unmanned Aerial Systems (sUASs) in low-altitude airspaces has increased the need for reliable tactical deconfliction under safety-critical constraints. Tactical deconfliction involves short-horizon decision-making in dense, partially observable, and heterogeneous multi-agent environments, where both cooperative separation assurance and operational efficiency must be maintained. While Large Language Models (LLMs) exhibit strong reasoning capabilities, their direct application to air traffic control remains limited by insufficient domain grounding and unpredictable output inconsistency. This paper investigates LLMs as decision-makers in cooperative multi-agent tactical deconfliction using fine-tuning strategies that align model outputs to human operator heuristics. We propose a simulation-to-language data generation pipeline based on the BlueSky air traffic simulator that produces rule-consistent deconfliction datasets reflecting established safety practices. A pretrained Qwen-Math-7B model is fine-tuned using two parameter-efficient strategies: supervised fine-tuning with Low-Rank Adaptation (LoRA) and preference-based fine-tuning combining LoRA with Group-Relative Policy Optimization (GRPO). Experimental results on validation datasets and closed-loop simulations demonstrate that supervised LoRA fine-tuning substantially improves decision accuracy, consistency, and separation performance compared to the pretrained LLM, with significant reductions in near mid-air collisions. GRPO provides additional coordination benefits but exhibits reduced robustness when interacting with heterogeneous agent policies. |
| 2026-03-30 | [The FreeGSNKE Pulse Design Tool (FPDT): a computational framework for evolutive plasma scenario and control design](http://arxiv.org/abs/2603.28513v1) | K. Pentland, N. C. Amorisco et al. | We present the FreeGSNKE Pulse Design Tool (FPDT), an open-source, Python-based computational framework that enables in silico testing and predictive design of tokamak plasma scenarios and control strategies. The FPDT couples the FreeGSNKE evolutive equilibrium solver with a virtual Plasma Control System (PCS) containing modular and customisable controllers. Given a set of user-defined waveforms and control parameters, the virtual PCS uses feedback and feedforward control to modulate plasma current, position, and shape, while adhering to machine safety limits on poloidal field coil currents and voltages. The resulting framework allows simulation of the controlled dynamic evolution of plasma equilibria, along with the currents in both active poloidal field coils and passive conducting structures, under the assumption of axisymmetry. The FPDT can be used to develop plasma scenarios, test control schemes, calibrate control parameters, and perform uncertainty quantification studies, thereby reducing iterative and expensive experimental testing on a physical tokamak. The FPDT is machine-agnostic and can be customised to implement different control algorithms tailored to the specific tokamak of interest. Here, we outline the overall framework and validate its performance on plasma discharges on the MAST Upgrade tokamak in the `flat-top' phase. We demonstrate excellent quantitative agreement between the FPDT simulations, the desired control waveforms, and the experimental shot data. With this extension to the FreeGSNKE open-source suite of codes we aim to encourage more reproducible and collaborative research in plasma modelling and control. |
| 2026-03-30 | [With a Little Help From My Friends: Collective Manipulation in Risk-Controlling Recommender Systems](http://arxiv.org/abs/2603.28476v1) | Giovanni De Toni, Cristian Consonni et al. | Recommendation systems have become central gatekeepers of online information, shaping user behaviour across a wide range of activities. In response, users increasingly organize and coordinate to steer algorithmic outcomes toward diverse goals, such as promoting relevant content or limiting harmful material, relying on platform affordances -- such as likes, reviews, or ratings. While these mechanisms can serve beneficial purposes, they can also be leveraged for adversarial manipulation, particularly in systems where such feedback directly informs safety guarantees. In this paper, we study this vulnerability in recently proposed risk-controlling recommender systems, which use binary user feedback (e.g., "Not Interested") to provably limit exposure to unwanted content via conformal risk control. We empirically demonstrate that their reliance on aggregate feedback signals makes them inherently susceptible to coordinated adversarial user behaviour. Using data from a large-scale online video-sharing platform, we show that a small coordinated group (comprising only 1% of the user population) can induce up to a 20% degradation in nDCG for non-adversarial users by exploiting the affordances provided by risk-controlling recommender systems. We evaluate simple, realistic attack strategies that require little to no knowledge of the underlying recommendation algorithm and find that, while coordinated users can significantly harm overall recommendation quality, they cannot selectively suppress specific content groups through reporting alone. Finally, we propose a mitigation strategy that shifts guarantees from the group level to the user level, showing empirically how it can reduce the impact of adversarial coordinated behaviour while ensuring personalized safety for individuals. |
| 2026-03-30 | [A Predictive Control Strategy to Offset-Point Tracking for Agricultural Mobile Robots](http://arxiv.org/abs/2603.28439v1) | Stephane Ngnepiepaye Wembe, Vincent Rousseau et al. | Robots are increasingly being deployed in agriculture to support sustainable practices and improve productivity. They offer strong potential to enable precise, efficient, and environmentally friendly operations. However, most existing path-following controllers focus solely on the robot's center of motion and neglect the spatial footprint and dynamics of attached implements. In practice, implements such as mechanical weeders or spring-tine cultivators are often large, rigidly mounted, and directly interacting with crops and soil; ignoring their position can degrade tracking performance and increase the risk of crop damage. To address this limitation, we propose a closed-form predictive control strategy extending the approach introduced in [1]. The method is developed specifically for Ackermann-type agricultural vehicles and explicitly models the implement as a rigid offset point, while accounting for lateral slip and lever-arm effects. The approach is benchmarked against state-of-the-art baseline controllers, including a reactive geometric method, a reactive backstepping method, and a model-based predictive scheme. Real-world agricultural experiments with two different implements show that the proposed method reduces the median tracking error by 24% to 56%, and decreases peak errors during curvature transitions by up to 70%. These improvements translate into enhanced operational safety, particularly in scenarios where the implement operates in close proximity to crop rows. |
| 2026-03-30 | [Structural-Ambiguity-Aware Translation from Natural Language to Signal Temporal Logic](http://arxiv.org/abs/2603.28426v1) | Kosei Fushimi, Kazunobu Serizawa et al. | Signal Temporal Logic (STL) is widely used to specify timed and safety-critical tasks for cyber-physical systems, but writing STL formulas directly is difficult for non-expert users. Natural language (NL) provides a convenient interface, yet its inherent structural ambiguity makes one-to-one translation into STL unreliable. In this paper, we propose an \textit{ambiguity-preserving} method for translating NL task descriptions into STL candidate formulas. The key idea is to retain multiple plausible syntactic analyses instead of forcing a single interpretation at the parsing stage. To this end, we develop a three-stage pipeline based on Combinatory Categorial Grammar (CCG): ambiguity-preserving $n$-best parsing, STL-oriented template-based semantic composition, and canonicalization with score aggregation. The proposed method outputs a deduplicated set of STL candidates with plausibility scores, thereby explicitly representing multiple possible formal interpretations of an ambiguous instruction. In contrast to existing one-best NL-to-logic translation methods, the proposed approach is designed to preserve attachment and scope ambiguity. Case studies on representative task descriptions demonstrate that the method generates multiple STL candidates for genuinely ambiguous inputs while collapsing unambiguous or canonically equivalent derivations to a single STL formula. |
| 2026-03-30 | [Unified Restoration-Perception Learning: Maritime Infrared-Visible Image Fusion and Segmentation](http://arxiv.org/abs/2603.28414v1) | Weichao Cai, Weiliang Huang et al. | Marine scene understanding and segmentation plays a vital role in maritime monitoring and navigation safety. However, prevalent factors like fog and strong reflections in maritime environments cause severe image degradation, significantly compromising the stability of semantic perception. Existing restoration and enhancement methods typically target specific degradations or focus solely on visual quality, lacking end-to-end collaborative mechanisms that simultaneously improve structural recovery and semantic effectiveness. Moreover, publicly available infrared-visible datasets are predominantly collected from urban scenes, failing to capture the authentic characteristics of coupled degradations in marine environments. To address these challenges, the Infrared-Visible Maritime Ship Dataset (IVMSD) is proposed to cover various maritime scenarios under diverse weather and illumination conditions. Building upon this dataset, a Multi-task Complementary Learning Framework (MCLF) is proposed to collaboratively perform image restoration, multimodal fusion, and semantic segmentation within a unified architecture. The framework includes a Frequency-Spatial Enhancement Complementary (FSEC) module for degradation suppression and structural enhancement, a Semantic-Visual Consistency Attention (SVCA) module for semantic-consistent guidance, and a cross-modality guided attention mechanism for selective fusion. Experimental results on IVMSD demonstrate that the proposed method achieves state-of-the-art segmentation performance, significantly enhancing robustness and perceptual quality under complex maritime conditions. |
| 2026-03-30 | [Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes](http://arxiv.org/abs/2603.28319v1) | Luke Palmer, Petar Palasek et al. | Accurately modelling human attention is essential for numerous computer vision applications, particularly in the domain of automotive safety. Existing methods typically collapse gaze into saliency maps or scanpaths, treating gaze dynamics only implicitly. We instead formulate gaze modelling as an autoregressive dynamical system and explicitly unroll raw gaze trajectories over time, conditioned on both gaze history and the evolving environment. Driving scenes are represented as gaze-centric graphs processed by the Affinity Relation Transformer (ART), a heterogeneous graph transformer that models interactions between driver gaze, traffic objects, and road structure. We further introduce the Object Density Network (ODN) to predict next-step gaze distributions, capturing the stochastic and object-centric nature of attentional shifts in complex environments. We also release Focus100, a new dataset of raw gaze data from 30 participants viewing egocentric driving footage. Trained directly on raw gaze, without fixation filtering, our unified approach produces more natural gaze trajectories, scanpath dynamics, and saliency maps than existing attention models, offering valuable insights for the temporal modelling of human attention in dynamic environments. |
| 2026-03-30 | [DiffAttn: Diffusion-Based Drivers' Visual Attention Prediction with LLM-Enhanced Semantic Reasoning](http://arxiv.org/abs/2603.28251v1) | Weimin Liu, Qingkun Li et al. | Drivers' visual attention provides critical cues for anticipating latent hazards and directly shapes decision-making and control maneuvers, where its absence can compromise traffic safety. To emulate drivers' perception patterns and advance visual attention prediction for intelligent vehicles, we propose DiffAttn, a diffusion-based framework that formulates this task as a conditional diffusion-denoising process, enabling more accurate modeling of drivers' attention. To capture both local and global scene features, we adopt Swin Transformer as encoder and design a decoder that combines a Feature Fusion Pyramid for cross-layer interaction with dense, multi-scale conditional diffusion to jointly enhance denoising learning and model fine-grained local and global scene contexts. Additionally, a large language model (LLM) layer is incorporated to enhance top-down semantic reasoning and improve sensitivity to safety-critical cues. Extensive experiments on four public datasets demonstrate that DiffAttn achieves state-of-the-art (SoTA) performance, surpassing most video-based, top-down-feature-driven, and LLM-enhanced baselines. Our framework further supports interpretable driver-centric scene understanding and has the potential to improve in-cabin human-machine interaction, risk perception, and drivers' state measurement in intelligent vehicles. |
| 2026-03-30 | [Effects of gravity on lean hydrogen/air flame instability: From linear scaling law to nonlinear morphology evolution](http://arxiv.org/abs/2603.28249v1) | Qizhe Wen, Yan Wang et al. | The instability characteristics of lean hydrogen/air flames have attracted considerable research attention, yet the effect of gravity remains insufficiently understood. In this study, time-resolved two-dimensional simulations with detailed chemistry and transport are conducted to investigate the influence of gravity-induced Rayleigh-Taylor (RT) instability on the linear growth rate of disturbances and nonlinear morphology evolution of cellular flame fronts at different length scales. In the linear regime, a parametric study is performed across various equivalence ratios, initial temperatures and pressures; in each case, the dispersion relation is calculated for various gravity levels. The influence of gravity is most pronounced under ultra-lean, low-temperature, and high-pressure conditions, and a universal scaling law between gravity sensitivity and the Froude number is established. In the nonlinear regime, gravity has opposite effects on the large-scale and small-scale structures of lean hydrogen flames. On the one hand, gravity inhibits the splitting of small-scale cellular structures through a baroclinic torque mechanism; on the other hand, it promotes the development of large-scale finger-like structures, thereby increasing the total surface area and the global consumption speed of the flame. The effects of gravity on the probability distributions of cell size, displacement speed, Karlovitz number, and local curvature are also analyzed. The results and findings of the present study should advance the fundamental understanding of hydrogen flame dynamics under varying gravity conditions and provide insight for relevant applications, including fire safety and space propulsion. |
| 2026-03-30 | [Detecting the Unexpected: AI-Driven Anomaly Detection in Smart Bridge Monitoring](http://arxiv.org/abs/2603.28225v1) | Rahul Jaiswal, Joakim Hellum et al. | Bridges are critical components of national infrastructure and smart cities. Therefore, smart bridge monitoring is essential for ensuring public safety and preventing catastrophic failures or accidents. Traditional bridge monitoring methods rely heavily on human visual inspections, which are time-consuming and prone to subjectivity and error. This paper proposes an artificial intelligence (AI)-driven anomaly detection approach for smart bridge monitoring. Specifically, a simple machine learning (ML) model is developed using real-time sensor data collected by the iBridge sensor devices installed on a bridge in Norway. The proposed model is evaluated against different ML models. Experimental results demonstrate that the density-based spatial clustering of applications with noise (DBSCAN)-based model outperforms in accurately detecting the anomalous events (bridge accident). These findings indicate that the proposed model is well-suited for smart bridge monitoring and can enhance public safety by enabling the timely detection of unforeseen incidents. |
| 2026-03-30 | [Event-Based Method for High-Speed 3D Deformation Measurement under Extreme Illumination Conditions](http://arxiv.org/abs/2603.28159v1) | Banglei Guan, Yifei Bian et al. | Background: Large engineering structures, such as space launch towers and suspension bridges, are subjected to extreme forces that cause high-speed 3D deformation and compromise safety. These structures typically operate under extreme illumination conditions. Traditional cameras often struggle to handle strong light intensity, leading to overexposure due to their limited dynamic range.   Objective: Event cameras have emerged as a compelling alternative to traditional cameras in high dynamic range and low-latency applications. This paper presents an integrated method, from calibration to measurement, using a multi-event camera array for high-speed 3D deformation monitoring of structures in extreme illumination conditions.   Methods: Firstly, the proposed method combines the characteristics of the asynchronous event stream and temporal correlation analysis to extract the corresponding marker center point. Subsequently, the method achieves rapid calibration by solving the Kruppa equations in conjunction with a parameter optimization framework. Finally, by employing a unified coordinate transformation and linear intersection, the method enables the measurement of 3D deformation of the target structure.   Results: Experiments confirmed that the relative measurement error is below 0.08%. Field experiments under extreme illumination conditions, including self-calibration of a multi-event camera array and 3D deformation measurement, verified the performance of the proposed method.   Conclusions: This paper addressed the critical limitation of traditional cameras in measuring high-speed 3D deformations under extreme illumination conditions. The experimental results demonstrate that, compared to other methods, the proposed method can accurately measure 3D deformations of structures under harsh lighting conditions, and the relative error of the measured deformation is less than 0.1%. |
| 2026-03-30 | [A Position Statement on Endovascular Models and Effectiveness Metrics for Mechanical Thrombectomy Navigation, on behalf of the Stakeholder Taskforce for AI-assisted Robotic Thrombectomy (START)](http://arxiv.org/abs/2603.28129v1) | Harry Robertshaw, Anna Barnes et al. | While we are making progress in overcoming infectious diseases and cancer; one of the major medical challenges of the mid-21st century will be the rising prevalence of stroke. Large vessels occlusions are especially debilitating, yet effective treatment (needed within hours to achieve best outcomes) remains limited due to geography. One solution for improving timely access to mechanical thrombectomy in geographically diverse populations is the deployment of robotic surgical systems. Artificial intelligence (AI) assistance may enable the upskilling of operators in this emerging therapeutic delivery approach. Our aim was to establish consensus frameworks for developing and validating AI-assisted robots for thrombectomy. Objectives included standardizing effectiveness metrics and defining reference testbeds across in silico, in vitro, ex vivo, and in vivo environments. To achieve this, we convened experts in neurointervention, robotics, data science, health economics, policy, statistics, and patient advocacy. Consensus was built through an incubator day, a Delphi process, and a final Position Statement. We identified that the four essential testbed environments each had distinct validation roles. Realism requirements vary: simpler testbeds should include realistic vessel anatomy compatible with guidewire and catheter use, while standard testbeds should incorporate deformable vessels. More advanced testbeds should include blood flow, pulsatility, and disease features. There are two macro-classes of effectiveness metrics: one for in silico, in vitro, and ex vivo stages focusing on technical navigation, and another for in vivo stages, focused on clinical outcomes. Patient safety is central to this technology's development. One requisite patient safety task needed now is to correlate in vitro measurements to in vivo complications. |
| 2026-03-30 | [From Vessel Trajectories to Safety-Critical Encounter Scenarios: A Generative AI Framework for Autonomous Ship Digital Testing](http://arxiv.org/abs/2603.28067v1) | Sijin Sun, Liangbin Zhao et al. | Digital testing has emerged as a key paradigm for the development and verification of autonomous maritime navigation systems, yet the availability of realistic and diverse safety-critical encounter scenarios remains limited. Existing approaches either rely on handcrafted templates, which lack realism, or extract cases directly from historical data, which cannot systematically expand rare high-risk situations.   This paper proposes a data-driven framework that converts large-scale Automatic Identification System (AIS) trajectories into structured safety-critical encounter scenarios. The framework combines generative trajectory modeling with automated encounter pairing and temporal parameterization to enable scalable scenario construction while preserving real traffic characteristics. To enhance trajectory realism and robustness under noisy AIS observations, a multi-scale temporal variational autoencoder is introduced to capture vessel motion dynamics across different temporal resolutions.   Experiments on real-world maritime traffic flows demonstrate that the proposed method improves trajectory fidelity and smoothness, maintains statistical consistency with observed data, and enables the generation of diverse safety-critical encounter scenarios beyond those directly recorded. The resulting framework provides a practical pathway for building scenario libraries to support digital testing, benchmarking, and safety assessment of autonomous navigation and intelligent maritime traffic management systems. Code is available at https://anonymous.4open.science/r/traj-gen-anonymous-review. |
| 2026-03-30 | [Beyond the Answer: Decoding the Behavior of LLMs as Scientific Reasoners](http://arxiv.org/abs/2603.28038v1) | Rohan Pandey, Eric Ye et al. | As Large Language Models (LLMs) achieve increasingly sophisticated performance on complex reasoning tasks, current architectures serve as critical proxies for the internal heuristics of frontier models. Characterizing emergent reasoning is vital for long-term interpretability and safety. Furthermore, understanding how prompting modulates these processes is essential, as natural language will likely be the primary interface for interacting with AGI systems. In this work, we use a custom variant of Genetic Pareto (GEPA) to systematically optimize prompts for scientific reasoning tasks, and analyze how prompting can affect reasoning behavior. We investigate the structural patterns and logical heuristics inherent in GEPA-optimized prompts, and evaluate their transferability and brittleness. Our findings reveal that gains in scientific reasoning often correspond to model-specific heuristics that fail to generalize across systems, which we call "local" logic. By framing prompt optimization as a tool for model interpretability, we argue that mapping these preferred reasoning structures for LLMs is an important prerequisite for effectively collaborating with superhuman intelligence. |
| 2026-03-30 | [Effort-Based Criticality Metrics for Evaluating 3D Perception Errors in Autonomous Driving](http://arxiv.org/abs/2603.28029v1) | Sharang Kaul, Simon Bultmann et al. | Criticality metrics such as time-to-collision (TTC) quantify collision urgency but conflate the consequences of false-positive (FP) and false-negative (FN) perception errors. We propose two novel effort-based metrics: False Speed Reduction (FSR), the cumulative velocity loss from persistent phantom detections, and Maximum Deceleration Rate (MDR), the peak braking demand from missed objects under a constant-acceleration model. These longitudinal metrics are complemented by Lateral Evasion Acceleration (LEA), adapted from prior lateral evasion kinematics and coupled with reachability-based collision timing to quantify the minimum steering effort to avoid a predicted collision. A reachability-based ellipsoidal collision filter ensures only dynamically plausible threats are scored, with frame-level matching and track-level aggregation. Evaluation of different perception pipelines on nuScenes and Argoverse~2 shows that 65-93% of errors are non-critical, and Spearman correlation analysis confirms that all three metrics capture safety-relevant information inaccessible to established time-based, deceleration-based, or normalized criticality measures, enabling targeted mining of the most critical perception failures. |

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



