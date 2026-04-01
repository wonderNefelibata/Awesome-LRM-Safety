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
| 2026-03-31 | [EnsembleSHAP: Faithful and Certifiably Robust Attribution for Random Subspace Method](http://arxiv.org/abs/2603.30034v1) | Yanting Wang, Jinyuan Jia | Random subspace method has wide security applications such as providing certified defenses against adversarial and backdoor attacks, and building robustly aligned LLM against jailbreaking attacks. However, the explanation of random subspace method lacks sufficient exploration. Existing state-of-the-art feature attribution methods, such as Shapley value and LIME, are computationally impractical and lacks security guarantee when applied to random subspace method. In this work, we propose EnsembleSHAP, an intrinsically faithful and secure feature attribution for random subspace method that reuses its computational byproducts. Specifically, our feature attribution method is 1) computationally efficient, 2) maintains essential properties of effective feature attribution (such as local accuracy), and 3) offers guaranteed protection against privacy-preserving attacks on feature attribution methods. To the best of our knowledge, this is the first work to establish provable robustness against explanation-preserving attacks. We also perform comprehensive evaluations for our explanation's effectiveness when faced with different empirical attacks, including backdoor attacks, adversarial attacks, and jailbreak attacks. The code is at https://github.com/Wang-Yanting/EnsembleSHAP. WARNING: This document may include content that could be considered harmful. |
| 2026-03-31 | [Extending MONA in Camera Dropbox: Reproduction, Learned Approval, and Design Implications for Reward-Hacking Mitigation](http://arxiv.org/abs/2603.29993v1) | Nathan Heath | Myopic Optimization with Non-myopic Approval (MONA) mitigates multi-step reward hacking by restricting the agent's planning horizon while supplying far-sighted approval as a training signal~\cite{farquhar2025mona}. The original paper identifies a critical open question: how the method of constructing approval -- particularly the degree to which approval depends on achieved outcomes -- affects whether MONA's safety guarantees hold. We present a reproduction-first extension of the public MONA Camera Dropbox environment that (i)~repackages the released codebase as a standard Python project with scripted PPO training, (ii)~confirms the published contrast between ordinary RL (91.5\% reward-hacking rate) and oracle MONA (0.0\% hacking rate) using the released reference arrays, and (iii)~introduces a modular learned-approval suite spanning oracle, noisy, misspecified, learned, and calibrated approval mechanisms. In reduced-budget pilot sweeps across approval methods, horizons, dataset sizes, and calibration strategies, the best calibrated learned-overseer run achieves zero observed reward hacking but substantially lower intended-behavior rates than oracle MONA (11.9\% vs.\ 99.9\%), consistent with under-optimization rather than re-emergent hacking. These results operationalize the MONA paper's approval-spectrum conjecture as a runnable experimental object and suggest that the central engineering challenge shifts from proving MONA's concept to building learned approval models that preserve sufficient foresight without reopening reward-hacking channels. Code, configurations, and reproduction commands are publicly available. https://github.com/codernate92/mona-camera-dropbox-repro |
| 2026-03-31 | [Performative Scenario Optimization](http://arxiv.org/abs/2603.29982v1) | Quanyan Zhu, Zhengye Han | This paper introduces a performative scenario optimization framework for decision-dependent chance-constrained problems. Unlike classical stochastic optimization, we account for the feedback loop where decisions actively shape the underlying data-generating process. We define performative solutions as self-consistent equilibria and establish their existence using Kakutani's fixed-point theorem. To ensure computational tractability without requiring an explicit model of the environment, we propose a model-free, scenario-based approximation that alternates between data generation and optimization. Under mild regularity conditions, we prove that a stochastic fixed-point iteration, equipped with a logarithmic sample size schedule, converges almost surely to the unique performative solution. The effectiveness of the proposed framework is demonstrated through an emerging AI safety application: deploying performative guardrails against Large Language Model (LLM) jailbreaks. Numerical results confirm the co-evolution and convergence of the guardrail classifier and the induced adversarial prompt distribution to a stable equilibrium. |
| 2026-03-31 | [SurgTEMP: Temporal-Aware Surgical Video Question Answering with Text-guided Visual Memory for Laparoscopic Cholecystectomy](http://arxiv.org/abs/2603.29962v1) | Shi Li, Vinkle Srivastav et al. | Surgical procedures are inherently complex and risky, requiring extensive expertise and constant focus to well navigate evolving intraoperative scenes. Computer-assisted systems such as surgical visual question answering (VQA) offer promises for education and intraoperative support. Current surgical VQA research largely focuses on static frame analysis, overlooking rich temporal semantics. Surgical video question answering is further challenged by low visual contrast, its highly knowledge-driven nature, diverse analytical needs spanning scattered temporal windows, and the hierarchy from basic perception to high-level intraoperative assessment. To address these challenges, we propose SurgTEMP, a multimodal LLM framework featuring (i) a query-guided token selection module that builds hierarchical visual memory (spatial and temporal memory banks) and (ii) a Surgical Competency Progression (SCP) training scheme. Together, these components enable effective modeling of variable-length surgical videos while preserving procedure-relevant cues and temporal coherence, and better support diverse downstream assessment tasks. To support model development, we introduce CholeVidQA-32K, a surgical video question answering dataset comprising 32K open-ended QA pairs and 3,855 video segments (approximately 128 h total) from laparoscopic cholecystectomy. The dataset is organized into a three-level hierarchy -- Perception, Assessment, and Reasoning -- spanning 11 tasks from instrument/action/anatomy perception to Critical View of Safety (CVS), intraoperative difficulty, skill proficiency, and adverse event assessment. In comprehensive evaluations against state-of-the-art open-source multimodal and video LLMs (fine-tuned and zero-shot), SurgTEMP achieves substantial performance improvements, advancing the state of video-based surgical VQA. |
| 2026-03-31 | [Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance](http://arxiv.org/abs/2603.29941v1) | Vanessa Emanuela Guarino, Claudia Winklmayr et al. | Uncertainty Quantification (UQ) is crucial for ensuring the reliability of automated image segmentations in safety-critical domains like biomedical image analysis or autonomous driving. In segmentation, UQ generates pixel-wise uncertainty scores that must be aggregated into image-level scores for downstream tasks like Out-of-Distribution (OoD) or failure detection. Despite routine use of aggregation strategies, their properties and impact on downstream task performance have not yet been comprehensively studied. Global Average is the default choice, yet it does not account for spatial and structural features of segmentation uncertainty. Alternatives like patch-, class- and threshold-based strategies exist, but lack systematic comparison, leading to inconsistent reporting and unclear best practices. We address this gap by (1) formally analyzing properties, limitations, and pitfalls of common strategies; (2) proposing novel strategies that incorporate spatial uncertainty structure and (3) benchmarking their performance on OoD and failure detection across ten datasets that vary in image geometry and structure. We find that aggregators leveraging spatial structure yield stronger performance in both downstream tasks studied. However, the performance of individual aggregators depends heavily on dataset characteristics, so we (4) propose a meta-aggregator that integrates multiple aggregators and performs robustly across datasets. |
| 2026-03-31 | [C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving](http://arxiv.org/abs/2603.29908v1) | Zhihong Cui, Haoran Tang et al. | Trajectory planning for autonomous driving increasingly leverages large language models (LLMs) for commonsense reasoning, yet LLM outputs are inherently unreliable, posing risks in safety-critical applications. We propose C-TRAIL, a framework built on a Commonsense World that couples LLM-derived commonsense with a trust mechanism to guide trajectory planning. C-TRAIL operates through a closed-loop Recall, Plan, and Update cycle: the Recall module queries an LLM for semantic relations and quantifies their reliability via a dual-trust mechanism; the Plan module injects trust-weighted commonsense into Monte Carlo Tree Search (MCTS) through a Dirichlet trust policy; and the Update module adaptively refines trust scores and policy parameters from environmental feedback. Experiments on four simulated scenarios in Highway-env and two real-world levelXData datasets (highD, rounD) show that C-TRAIL consistently outperforms state-of-the-art baselines, reducing ADE by 40.2%, FDE by 51.7%, and improving SR by 16.9 percentage points on average. The source code is available at https://github.com/ZhihongCui/CTRAIL. |
| 2026-03-31 | [Security and Privacy in Virtual and Robotic Assistive Systems: A Comparative Framework](http://arxiv.org/abs/2603.29907v1) | Nelly Elsayed | Assistive technologies increasingly support independence, accessibility, and safety for older adults, people with disabilities, and individuals requiring continuous care. Two major categories are virtual assistive systems and robotic assistive systems operating in physical environments. Although both offer significant benefits, they introduce important security and privacy risks due to their reliance on artificial intelligence, network connectivity, and sensor-based perception. Virtual systems are primarily exposed to threats involving data privacy, unauthorized access, and adversarial voice manipulation. In contrast, robotic systems introduce additional cyber-physical risks such as sensor spoofing, perception manipulation, command injection, and physical safety hazards. In this paper, we present a comparative analysis of security and privacy challenges across these systems. We develop a unified comparative threat-modeling framework that enables structured analysis of attack surfaces, risk profiles, and safety implications across both systems. Moreover, we provide design recommendations for developing secure, privacy-preserving, and trustworthy assistive technologies. |
| 2026-03-31 | [Where to Put Safety? Control Barrier Function Placement in Networked Control Systems](http://arxiv.org/abs/2603.29792v1) | Severin Beger, Yuling Chen et al. | Ensuring safe behavior is critical for modern autonomous cyber-physical systems. Control barrier functions (CBFs) are widely used to enforce safety in autonomous systems, yet their placement within networked control architectures remains largely unexplored. In this work, we investigate where to enforce safety in a networked control system in which a remote model predictive controller (MPC) communicates with the plant over a delayed network. We compare two safety strategies: i) a local myopic CBF filter applied at the plant and ii) predictive CBF constraints embedded in the remote MPC. For both architectures, we derive state-dependent disturbance tolerance bounds and show that safety placement induces a fundamental trade-off: local CBFs provide higher disturbance tolerance due to access to fresh state measurements, whereas MPC-CBF enables improved performance through anticipatory behavior, but yields stricter admissible disturbance levels. Motivated by this insight, we propose a combined architecture that integrates predictive and local safety mechanisms. The theoretical findings are illustrated in simulations on a planar three-degree-of-freedom robot performing a collision-avoidance task. |
| 2026-03-31 | [From Skeletons to Semantics: Design and Deployment of a Hybrid Edge-Based Action Detection System for Public Safety](http://arxiv.org/abs/2603.29777v1) | Ganen Sethupathy, Lalit Dumka et al. | Public spaces such as transport hubs, city centres, and event venues require timely and reliable detection of potentially violent behaviour to support public safety. While automated video analysis has made significant progress, practical deployment remains constrained by latency, privacy, and resource limitations, particularly under edge-computing conditions. This paper presents the design and demonstrator-based deployment of a hybrid edge-based action detection system that combines skeleton-based motion analysis with vision-language models for semantic scene interpretation. Skeleton-based processing enables continuous, privacy-aware monitoring with low computational overhead, while vision-language models provide contextual understanding and zero-shot reasoning capabilities for complex and previously unseen situations. Rather than proposing new recognition models, the contribution focuses on a system-level comparison of both paradigms under realistic edge constraints. The system is implemented on a GPU-enabled edge device and evaluated with respect to latency, resource usage, and operational trade-offs using a demonstrator-based setup. The results highlight the complementary strengths and limitations of motioncentric and semantic approaches and motivate a hybrid architecture that selectively augments fast skeletonbased detection with higher-level semantic reasoning. The presented system provides a practical foundation for privacy-aware, real-time video analysis in public safety applications. |
| 2026-03-31 | [TSHA: A Benchmark for Visual Language Models in Trustworthy Safety Hazard Assessment Scenarios](http://arxiv.org/abs/2603.29759v1) | Qiucheng Yu, Ruijie Xu et al. | Recent advances in vision-language models (VLMs) have accelerated their application to indoor safety hazards assessment. However, existing benchmarks suffer from three fundamental limitations: (1) heavy reliance on synthetic datasets constructed via simulation software, creating a significant domain gap with real-world environments; (2) oversimplified safety tasks with artificial constraints on hazard and scene types, thereby limiting model generalization; and (3) absence of rigorous evaluation protocols to thoroughly assess model capabilities in complex home safety scenarios. To address these challenges, we introduce TSHA (\textbf{T}rustworthy \textbf{S}afety \textbf{H}azards \textbf{A}ssessment), a comprehensive benchmark comprising 81,809 carefully curated training samples drawn from four complementary sources: existing indoor datasets, internet images, AIGC images, and newly captured images. This benchmark set also includes a highly challenging test set with 1707 samples, comprising not only a carefully selected subset from the training distribution but also newly added videos and panoramic images containing multiple safety hazards, used to evaluate the model's robustness in complex safety scenarios. Extensive experiments on 23 popular VLMs demonstrate that current VLMs lack robust capabilities for safety hazard assessment. Importantly, models trained on the TSHA training set not only achieve a significant performance improvement of up to +18.3 points on the TSHA test set but also exhibit enhanced generalizability across other benchmarks, underscoring the substantial contribution and importance of the TSHA benchmark. |
| 2026-03-31 | [Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding](http://arxiv.org/abs/2603.29709v1) | Joakim Edin, Andreas Motzfeldt et al. | Medical coding translates free-text clinical documentation into standardized codes drawn from classification systems that contain tens of thousands of entries and are updated annually. It is central to billing, clinical research, and quality reporting, yet remains largely manual, slow, and error-prone. Existing automated approaches learn to predict a fixed set of codes from labeled data, thereby preventing adaptation to new codes or different coding systems without retraining on different data. They also provide no explanation for their predictions, limiting trust in safety-critical settings. We introduce Symphony for Medical Coding, a system that approaches the task the way expert human coders do: by reasoning over the clinical narrative with direct access to the coding guidelines. This design allows Symphony to operate across any coding system and to provide span-level evidence linking each predicted code to the text that supports it. We evaluate on two public benchmarks and three real-world datasets spanning inpatient, outpatient, emergency, and subspecialty settings across the United States and the United Kingdom. Symphony achieves state-of-the-art results across all settings, establishing itself as a flexible, deployment-ready foundation for automated clinical coding. |
| 2026-03-31 | [SafeDMPs: Integrating Formal Safety with DMPs for Adaptive HRI](http://arxiv.org/abs/2603.29708v1) | Soumyodipta Nath, Pranav Tiwari et al. | Robots operating in human-centric environments must be both robust to disturbances and provably safe from collisions. Achieving these properties simultaneously and efficiently remains a central challenge. While Dynamic Movement Primitives (DMPs) offer inherent stability and generalization from single demonstrations, they lack formal safety guarantees. Conversely, formal methods like Control Barrier Functions (CBFs) provide provable safety but often rely on computationally expensive, real-time optimization, hindering their use in high-frequency control. This paper introduces SafeDMPs, a novel framework that resolves this trade-off. We integrate the closed-form efficiency and dynamic robustness of DMPs with a provably safe, non-optimization-based control law derived from Spatio-Temporal Tubes (STTs). This synergy allows us to generate motions that are not only robust to perturbations and adaptable to new goals, but also guaranteed to avoid static and dynamic obstacles. Our approach achieves a closed-form solution for a problem that traditionally requires online optimization. Experimental results on a 7-DOF robot manipulator demonstrate that SafeDMPs is orders of magnitude faster and more accurate than optimization-based baselines, making it an ideal solution for real-time, safe, and collaborative robotics. |
| 2026-03-31 | [Machine Learning in the Wild: Early Evidence of Non-Compliant ML-Automation in Open-Source Software](http://arxiv.org/abs/2603.29698v1) | Zohaib Arshid, Daniele Bifolco et al. | The increasing availability of Machine Learning (ML) models, particularly foundation models, enables their use across a range of downstream applications, from scenarios with missing data to safety-critical contexts. This, in principle, may contravene not only the models' terms of use, but also governmental principles and regulations. This paper presents a preliminary investigation into the use of ML models by 173 open-source projects on GitHub, spanning 16 application domains. We evaluate whether models are used to make decisions, the scope of these decisions, and whether any post-processing measures are taken to reduce the risks inherent in fully autonomous systems. Lastly, we investigate the models' compliance with established terms of use. This study lays the groundwork for defining guidelines for developers and creating analysis tools that automatically identify potential regulatory violations in the use of ML models in software systems. |
| 2026-03-31 | [SCORE: Statistical Certification of Regions of Attraction via Extreme Value Theory](http://arxiv.org/abs/2603.29658v1) | Pietro Zanotta, Panos Stinis et al. | Certifying the Region of Attraction (ROA) for high-dimensional nonlinear dynamical systems remains a severe computational bottleneck. Traditional deterministic verification methods, such as Sum-of-Squares (SOS) programming and Satisfiability Modulo Theories (SMT), provide hard guarantees but suffer from the curse of dimensionality, typically failing to scale beyond 20 dimensions. To overcome these limitations, we propose SCORE, a statistical certification framework that shifts from seeking deterministic guarantees to bounding the worst-case safety violation with high statistical confidence. By integrating Projected Stochastic Gradient Langevin Dynamics (PSGLD) with Extreme Value Theory (EVT), we frame ROA certification as a constrained extreme-value estimation problem on the sublevel set boundary. We theoretically demonstrate that modeling the optimization process as a stochastic diffusion on a compact manifold places the local maxima of the Lyapunov derivative into the Weibull maximum domain of attraction. Since the Weibull domain features a finite right endpoint, we can compute a rigorous statistical upper bound on the global maximum of the Lyapunov derivative. Numerical experiments validate that our EVT-based approach achieves certification tightness competitive to exact SOS programming on a 2D Van der Pol benchmark. Furthermore, we demonstrate unprecedented scalability by successfully certifying a dense, unstructured 500-dimensional ODE system up to a confidence level of 99.99\%, effectively bypassing the severe combinatorial constraints that limit existing formal verification pipelines. |
| 2026-03-31 | [Disentangled Graph Prompting for Out-Of-Distribution Detection](http://arxiv.org/abs/2603.29644v1) | Cheng Yang, Yu Hao et al. | When testing data and training data come from different distributions, deep neural networks (DNNs) will face significant safety risks in practical applications. Therefore, out-of-distribution (OOD) detection techniques, which can identify OOD samples at test time and alert the system, are urgently needed. Existing graph OOD detection methods usually characterize fine-grained in-distribution (ID) patterns from multiple perspectives, and train end-to-end graph neural networks (GNNs) for prediction. However, due to the unavailability of OOD data during training, the absence of explicit supervision signals could lead to sub-optimal performance of end-to-end encoders. To address this issue, we follow the pre-training+prompting paradigm to utilize pre-trained GNN encoders, and propose Disentangled Graph Prompting (DGP), to capture fine-grained ID patterns with the help of ID graph labels. Specifically, we design two prompt generators that respectively generate class-specific and class-agnostic prompt graphs by modifying the edge weights of an input graph. We also design several effective losses to train the prompt generators and prevent trivial solutions. We conduct extensive experiments on ten datasets to demonstrate the superiority of our proposed DGP, which achieves a relative AUC improvement of 3.63% over the best graph OOD detection baseline. Ablation studies and hyper-parameter experiments further show the effectiveness of DGP. Code is available at https://github.com/BUPT-GAMMA/DGP. |
| 2026-03-31 | [Optimizing Donor Outreach for Blood Collection Sessions: A Scalable Decision Support Framework](http://arxiv.org/abs/2603.29643v1) | André Carneiro, Pedro T. Monteiro et al. | Blood donation centers face challenges in matching supply with demand while managing donor availability. Although targeted outreach is important, it can cause donor fatigue via over-solicitation. Effective recruitment requires targeting the right donors at the right time, balancing constraints with donor convenience and eligibility. Despite extensive work on blood supply chain optimization and growing interest in algorithmic donor recruitment, the operational problem of assigning donors to sessions across a multi-site network, taking into account eligibility, capacity, blood-type demand targets, geographic convenience, and donor safety, remains unaddressed.   We address this gap with an optimization framework for donor invitation scheduling incorporating donor eligibility, travel convenience, blood-type demand targets, and penalties. We evaluate two strategies: (i) a binary integer linear programming (BILP) formulation and (ii) an efficient greedy heuristic. Evaluation uses the registry from Instituto Português do Sangue e da Transplantação (IPST) for invite planning in the Lisbon operational region using 4-month windows. A prospective pipeline integrates organic attendance forecasting, quantile-based demand targets, and residual capacity estimation for forward-looking invitation plans. Results reveal its key role in closing the supply-demand gap in the Lisbon operational region. A controlled comparison shows that the greedy heuristic achieves results comparable to the BILP, with 188x less peak memory and 115x faster runtime; trade-offs include 3.9 pp lower demand fulfillment (86.1% vs. 90.0%), larger donor-session distance, higher adverse-reaction donor exposure, and greater invitation burden per non-high-frequency donor, reflecting local versus global optimization. Experiments assess how constraint-aware scheduling can close gaps by mobilizing eligible inactive/lapsing donors. |
| 2026-03-31 | [Adaptive Mitigation of Insider Threats via Off-Policy Learning](http://arxiv.org/abs/2603.29594v1) | Gehui Xu, Kaiwen Chen et al. | An insider is a team member who covertly deviates from the team's optimal collaborative strategy to pursue a private objective while still appearing cooperative. Such an insider may initially behave cooperatively but later switch to selfish or malicious actions, thereby degrading collective performance, threatening mission success, and compromising operational safety. In this paper, we study such insider threats within an insider-aware, game-theoretic formulation, where the insider interacts with a decision maker (DM) under a continuous-time switched system, with each time interval characterized by a distinct insider behavioral pattern or threat level. We develop a periodic off-policy mitigation scheme that enables the DM to learn optimal mitigation policies from online data when encountering different insider threats, without requiring a priori knowledge of insider intentions. By designing appropriate conditions on the inter-learning interval time, we establish convergence guarantees for both the learning process and the closed-loop system, and characterize the corresponding mitigation performance achieved by the DM. |
| 2026-03-31 | [Be Water: An Evolutionary Proof for Trend-Following](http://arxiv.org/abs/2603.29593v1) | Yijia Chen | The proliferation of diverse, high-leverage trading instruments in modern financial markets presents a complex, "noisy" environment, leading to a critical question: which trading strategies are evolutionarily viable? To investigate this, we construct a large-scale agent-based model, "MAS-Utopia," comprising 10,000 agents with five distinct archetypes. This society is immersed in five years of high-frequency data under a counterfactual baseline: zero transaction friction and a robust Unconditional Basic Income (UBI) safety net. The simulation reveals a powerful evolutionary convergence. Strategies that attempt to fight the market's current - namely Mean-Reversion ("buy-the-dip") - prove structurally fragile. In contrast, the Trend-Following archetype, which adapts to the market's flow, emerges as the dominant phenotype. Translating this finding, we architect an LLM-driven system that emulates this successful logic. Our findings offer profound implications, echoing the ancient wisdom of "Be Water": for investors, it demonstrates that survival is achieved not by rigid opposition, but by disciplined alignment with the prevailing current; for markets, it critiques tools that encourage contrarian gambling; for society, it underscores the stabilizing power of economic safety nets. |
| 2026-03-31 | [Distributed Predictive Control Barrier Functions: Towards Scalable Safety Certification in Modular Multi-Agent Systems](http://arxiv.org/abs/2603.29560v1) | Jonas Ohnemus, Alexandre Didier et al. | We consider safety-critical multi-agent systems with distributed control architectures and potentially varying network topologies. While learning-based distributed control enables scalability and high performance, a lack of formal safety guarantees in the face of unforeseen disturbances and unsafe network topology changes may lead to system failure. To address this challenge, we introduce structured control barrier functions (s-CBFs) as a multi-agent safety framework. The s-CBFs are augmented to a distributed predictive control barrier function (D-PCBF), a predictive, optimization-based safety layer that uses model predictions to guarantee recoverable safety at all times. The proposed approach enables a permissive yet formal plug-and-play protocol, allowing agents to join or leave the network while ensuring safety recovery if a change in network topology requires temporarily unsafe behavior. We validate the formulation through simulations and real-time experiments of a miniature race-car platoon. |
| 2026-03-31 | [Security in LLM-as-a-Judge: A Comprehensive SoK](http://arxiv.org/abs/2603.29403v1) | Aiman Almasoud, Antony Anju et al. | LLM-as-a-Judge (LaaJ) is a novel paradigm in which powerful language models are used to assess the quality, safety, or correctness of generated outputs. While this paradigm has significantly improved the scalability and efficiency of evaluation processes, it also introduces novel security risks and reliability concerns that remain largely unexplored. In particular, LLM-based judges can become both targets of adversarial manipulation and instruments through which attacks are conducted, potentially compromising the trustworthiness of evaluation pipelines. In this paper, we present the first Systematization of Knowledge (SoK) focusing on the security aspects of LLM-as-a-Judge systems. We perform a comprehensive literature review across major academic databases, analyzing 863 works and selecting 45 relevant studies published between 2020 and 2026. Based on this study, we propose a taxonomy that organizes recent research according to the role played by LLM-as-a-Judge in the security landscape, distinguishing between attacks targeting LaaJ systems, attacks performed through LaaJ, defenses leveraging LaaJ for security purposes, and applications where LaaJ is used as an evaluation strategy in security-related domains. We further provide a comparative analysis of existing approaches, highlighting current limitations, emerging threats, and open research challenges. Our findings reveal significant vulnerabilities in LLM-based evaluation frameworks, as well as promising directions for improving their robustness and reliability. Finally, we outline key research opportunities that can guide the development of more secure and trustworthy LLM-as-a-Judge systems. |

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



