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
| 2025-06-06 | [PyGemini: Unified Software Development towards Maritime Autonomy Systems](http://arxiv.org/abs/2506.06262v1) | Kjetil Vasstein, Christian Le et al. | Ensuring the safety and certifiability of autonomous surface vessels (ASVs) requires robust decision-making systems, supported by extensive simulation, testing, and validation across a broad range of scenarios. However, the current landscape of maritime autonomy development is fragmented -- relying on disparate tools for communication, simulation, monitoring, and system integration -- which hampers interdisciplinary collaboration and inhibits the creation of compelling assurance cases, demanded by insurers and regulatory bodies. Furthermore, these disjointed tools often suffer from performance bottlenecks, vendor lock-in, and limited support for continuous integration workflows. To address these challenges, we introduce PyGemini, a permissively licensed, Python-native framework that builds on the legacy of Autoferry Gemini to unify maritime autonomy development. PyGemini introduces a novel Configuration-Driven Development (CDD) process that fuses Behavior-Driven Development (BDD), data-oriented design, and containerization to support modular, maintainable, and scalable software architectures. The framework functions as a stand-alone application, cloud-based service, or embedded library -- ensuring flexibility across research and operational contexts. We demonstrate its versatility through a suite of maritime tools -- including 3D content generation for simulation and monitoring, scenario generation for autonomy validation and training, and generative artificial intelligence pipelines for augmenting imagery -- thereby offering a scalable, maintainable, and performance-oriented foundation for future maritime robotics and autonomy research. |
| 2025-06-06 | [Statistical Guarantees in Data-Driven Nonlinear Control: Conformal Robustness for Stability and Safety](http://arxiv.org/abs/2506.06228v1) | Ting-Wei Hsu, Hiroyasu Tsukamoto | We present a true-dynamics-agnostic, statistically rigorous framework for establishing exponential stability and safety guarantees of closed-loop, data-driven nonlinear control. Central to our approach is the novel concept of conformal robustness, which robustifies the Lyapunov and zeroing barrier certificates of data-driven dynamical systems against model prediction uncertainties using conformal prediction. It quantifies these uncertainties by leveraging rank statistics of prediction scores over system trajectories, without assuming any specific underlying structure of the prediction model or distribution of the uncertainties. With the quantified uncertainty information, we further construct the conformally robust control Lyapunov function (CR-CLF) and control barrier function (CR-CBF), data-driven counterparts of the CLF and CBF, for fully data-driven control with statistical guarantees of finite-horizon exponential stability and safety. The performance of the proposed concept is validated in numerical simulations with four benchmark nonlinear control problems. |
| 2025-06-06 | ["We need to avail ourselves of GenAI to enhance knowledge distribution": Empowering Older Adults through GenAI Literacy](http://arxiv.org/abs/2506.06225v1) | Eunhye Grace Ko, Shaini Nanayakkara et al. | As generative AI (GenAI) becomes increasingly widespread, it is crucial to equip users, particularly vulnerable populations such as older adults (65 and older), with the knowledge to understand its benefits and potential risks. Older adults often exhibit greater reservations about adopting emerging technologies and require tailored literacy support. Using a mixed methods approach, this study examines strategies for delivering GenAI literacy to older adults through a chatbot named Litti, evaluating its impact on their AI literacy (knowledge, safety, and ethical use). The quantitative data indicated a trend toward improved AI literacy, though the results were not statistically significant. However, qualitative interviews revealed diverse levels of familiarity with generative AI and a strong desire to learn more. Findings also show that while Litti provided a positive learning experience, it did not significantly enhance participants' trust or sense of safety regarding GenAI. This exploratory case study highlights the challenges and opportunities in designing AI literacy education for the rapidly growing older adult population. |
| 2025-06-06 | [Experimental Study On Flashing-Induced Instabilities In An Open Natural Circulation System](http://arxiv.org/abs/2506.06213v1) | Yuliang Fang, Xiaxin Cao et al. | The natural circulation system (NCS) uses gravity pressure drop caused by density differences in the loop to generate the driving force without any external mechanical devices, which has been widely applied to the design of the nuclear reactor system and the passive safety system due to its simple structure, high intrinsic safety, and strong heat discharge capacity. However, the low-pressure condition can lead to a two-phase flow and make the flow characteristics in the NCS more complex. Flashing-induced instability occurring in the open NCS will cause the system structural vibration as well as mechanical damage and bring safety problems. The study on flashing-flow behaviors in an open NPS has been conducted experimentally in this paper. High-speed camera, thermal needle probe and wire-mesh sensor were adopted to record the flow pattern and measure the void fraction in the polycarbonate visualization riser section. In the start-up process, with the inlet temperature in the riser section increasing, the open NCS has experienced single-phase stable flow, intermittent oscillation between single-phase and two-phase, high subcooling two-phase stable flow, flashing-induced instabilities flow, and low subcooling two-phase stable flow. The flow pattern evolution of flow flashing goes through bubble flow, cap-slug flow, churn flow and wispy annular flow, in which the length of churn can account for more than 40% length of the two-phase regime. The flash number Nflash is used to divide the region of flashing-induced instabilities. It is found that the open NCS is in a stable two-phase flow when the flash number at the outlet of the riser section N_{flash,out} = 4\sim 5. |
| 2025-06-06 | [Let's CONFER: A Dataset for Evaluating Natural Language Inference Models on CONditional InFERence and Presupposition](http://arxiv.org/abs/2506.06133v1) | Tara Azin, Daniel Dumitrescu et al. | Natural Language Inference (NLI) is the task of determining whether a sentence pair represents entailment, contradiction, or a neutral relationship. While NLI models perform well on many inference tasks, their ability to handle fine-grained pragmatic inferences, particularly presupposition in conditionals, remains underexplored. In this study, we introduce CONFER, a novel dataset designed to evaluate how NLI models process inference in conditional sentences. We assess the performance of four NLI models, including two pre-trained models, to examine their generalization to conditional reasoning. Additionally, we evaluate Large Language Models (LLMs), including GPT-4o, LLaMA, Gemma, and DeepSeek-R1, in zero-shot and few-shot prompting settings to analyze their ability to infer presuppositions with and without prior context. Our findings indicate that NLI models struggle with presuppositional reasoning in conditionals, and fine-tuning on existing NLI datasets does not necessarily improve their performance. |
| 2025-06-06 | [Self driving algorithm for an active four wheel drive racecar](http://arxiv.org/abs/2506.06077v1) | Gergely Bari, Laszlo Palkovics | Controlling autonomous vehicles at their handling limits is a significant challenge, particularly for electric vehicles with active four wheel drive (A4WD) systems offering independent wheel torque control. While traditional Vehicle Dynamics Control (VDC) methods use complex physics-based models, this study explores Deep Reinforcement Learning (DRL) to develop a unified, high-performance controller. We employ the Proximal Policy Optimization (PPO) algorithm to train an agent for optimal lap times in a simulated racecar (TORCS) at the tire grip limit. Critically, the agent learns an end-to-end policy that directly maps vehicle states, like velocities, accelerations, and yaw rate, to a steering angle command and independent torque commands for each of the four wheels. This formulation bypasses conventional pedal inputs and explicit torque vectoring algorithms, allowing the agent to implicitly learn the A4WD control logic needed for maximizing performance and stability. Simulation results demonstrate the RL agent learns sophisticated strategies, dynamically optimizing wheel torque distribution corner-by-corner to enhance handling and mitigate the vehicle's inherent understeer. The learned behaviors mimic and, in aspects of grip utilization, potentially surpass traditional physics-based A4WD controllers while achieving competitive lap times. This research underscores DRL's potential to create adaptive control systems for complex vehicle dynamics, suggesting RL is a potent alternative for advancing autonomous driving in demanding, grip-limited scenarios for racing and road safety. |
| 2025-06-06 | [Trajectory Optimization for UAV-Based Medical Delivery with Temporal Logic Constraints and Convex Feasible Set Collision Avoidance](http://arxiv.org/abs/2506.06038v1) | Kaiyuan Chen, Yuhan Suo et al. | This paper addresses the problem of trajectory optimization for unmanned aerial vehicles (UAVs) performing time-sensitive medical deliveries in urban environments. Specifically, we consider a single UAV with 3 degree-of-freedom dynamics tasked with delivering blood packages to multiple hospitals, each with a predefined time window and priority. Mission objectives are encoded using Signal Temporal Logic (STL), enabling the formal specification of spatial-temporal constraints. To ensure safety, city buildings are modeled as 3D convex obstacles, and obstacle avoidance is handled through a Convex Feasible Set (CFS) method. The entire planning problem-combining UAV dynamics, STL satisfaction, and collision avoidance-is formulated as a convex optimization problem that ensures tractability and can be solved efficiently using standard convex programming techniques. Simulation results demonstrate that the proposed method generates dynamically feasible, collision-free trajectories that satisfy temporal mission goals, providing a scalable and reliable approach for autonomous UAV-based medical logistics. |
| 2025-06-06 | [On Inverse Problems, Parameter Estimation, and Domain Generalization](http://arxiv.org/abs/2506.06024v1) | Deborah Pereg | Signal restoration and inverse problems are key elements in most real-world data science applications. In the past decades, with the emergence of machine learning methods, inversion of measurements has become a popular step in almost all physical applications, which is normally executed prior to downstream tasks that often involve parameter estimation. In this work, we analyze the general problem of parameter estimation in an inverse problem setting. First, we address the domain-shift problem by re-formulating it in direct relation with the discrete parameter estimation analysis. We analyze a significant vulnerability in current attempts to enforce domain generalization, which we dubbed the Double Meaning Theorem. Our theoretical findings are experimentally illustrated for domain shift examples in image deblurring and speckle suppression in medical imaging. We then proceed to a theoretical analysis of parameter estimation given observed measurements before and after data processing involving an inversion of the observations. We compare this setting for invertible and non-invertible (degradation) processes. We distinguish between continuous and discrete parameter estimation, corresponding with regression and classification problems, respectively. Our theoretical findings align with the well-known information-theoretic data processing inequality, and to a certain degree question the common misconception that data-processing for inversion, based on modern generative models that may often produce outstanding perceptual quality, will necessarily improve the following parameter estimation objective. It is our hope that this paper will provide practitioners with deeper insights that may be leveraged in the future for the development of more efficient and informed strategic system planning, critical in safety-sensitive applications. |
| 2025-06-06 | [Unlocking Recursive Thinking of LLMs: Alignment via Refinement](http://arxiv.org/abs/2506.06009v1) | Haoke Zhang, Xiaobo Liang et al. | The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (https://github.com/Banner-Z/AvR.git). |
| 2025-06-06 | [CrimeMind: Simulating Urban Crime with Multi-Modal LLM Agents](http://arxiv.org/abs/2506.05981v1) | Qingbin Zeng, Ruotong Zhao et al. | Modeling urban crime is an important yet challenging task that requires understanding the subtle visual, social, and cultural cues embedded in urban environments. Previous work has predominantly focused on rule-based agent-based modeling (ABM) and deep learning methods. ABMs offer interpretability of internal mechanisms but exhibit limited predictive accuracy.In contrast, deep learning methods are often effective in prediction but are less interpretable and require extensive training data. Moreover, both lines of work lack the cognitive flexibility to adapt to changing environments. Leveraging the capabilities of large language models (LLMs), we propose CrimeMind, a novel LLM-driven ABM framework for simulating urban crime within a multi-modal urban context.A key innovation of our design is the integration of the Routine Activity Theory (RAT) into the agentic workflow of CrimeMind, enabling it to process rich multi-modal urban features and reason about criminal behavior.However, RAT requires LLM agents to infer subtle cues in evaluating environmental safety as part of assessing guardianship, which can be challenging for LLMs. To address this, we collect a small-scale human-annotated dataset and align CrimeMind's perception with human judgment via a training-free textual gradient method.Experiments across four major U.S. cities demonstrate that CrimeMind outperforms both traditional ABMs and deep learning baselines in crime hotspot prediction and spatial distribution accuracy, achieving up to a 24% improvement over the strongest baseline.Furthermore, we conduct counterfactual simulations of external incidents and policy interventions and it successfully captures the expected changes in crime patterns, demonstrating its ability to reflect counterfactual scenarios.Overall, CrimeMind enables fine-grained modeling of individual behaviors and facilitates evaluation of real-world interventions. |
| 2025-06-06 | [SurGSplat: Progressive Geometry-Constrained Gaussian Splatting for Surgical Scene Reconstruction](http://arxiv.org/abs/2506.05935v1) | Yuchao Zheng, Jianing Zhang et al. | Intraoperative navigation relies heavily on precise 3D reconstruction to ensure accuracy and safety during surgical procedures. However, endoscopic scenarios present unique challenges, including sparse features and inconsistent lighting, which render many existing Structure-from-Motion (SfM)-based methods inadequate and prone to reconstruction failure. To mitigate these constraints, we propose SurGSplat, a novel paradigm designed to progressively refine 3D Gaussian Splatting (3DGS) through the integration of geometric constraints. By enabling the detailed reconstruction of vascular structures and other critical features, SurGSplat provides surgeons with enhanced visual clarity, facilitating precise intraoperative decision-making. Experimental evaluations demonstrate that SurGSplat achieves superior performance in both novel view synthesis (NVS) and pose estimation accuracy, establishing it as a high-fidelity and efficient solution for surgical scene reconstruction. More information and results can be found on the page https://surgsplat.github.io/. |
| 2025-06-06 | [Small Models, Big Support: A Local LLM Framework for Teacher-Centric Content Creation and Assessment using RAG and CAG](http://arxiv.org/abs/2506.05925v1) | Zarreen Reza, Alexander Mazur et al. | While Large Language Models (LLMs) are increasingly utilized as student-facing educational aids, their potential to directly support educators, particularly through locally deployable and customizable open-source solutions, remains significantly underexplored. Many existing educational solutions rely on cloud-based infrastructure or proprietary tools, which are costly and may raise privacy concerns. Regulated industries with limited budgets require affordable, self-hosted solutions. We introduce an end-to-end, open-source framework leveraging small (3B-7B parameters), locally deployed LLMs for customized teaching material generation and assessment. Our system uniquely incorporates an interactive loop crucial for effective small-model refinement, and an auxiliary LLM verifier to mitigate jailbreaking risks, enhancing output reliability and safety. Utilizing Retrieval and Context Augmented Generation (RAG/CAG), it produces factually accurate, customized pedagogically-styled content. Deployed on-premises for data privacy and validated through an evaluation pipeline and a college physics pilot, our findings show that carefully engineered small LLM systems can offer robust, affordable, practical, and safe educator support, achieving utility comparable to larger models for targeted tasks. |
| 2025-06-06 | [Rethinking Semi-supervised Segmentation Beyond Accuracy: Reliability and Robustness](http://arxiv.org/abs/2506.05917v1) | Steven Landgraf, Markus Hillemann et al. | Semantic segmentation is critical for scene understanding but demands costly pixel-wise annotations, attracting increasing attention to semi-supervised approaches to leverage abundant unlabeled data. While semi-supervised segmentation is often promoted as a path toward scalable, real-world deployment, it is astonishing that current evaluation protocols exclusively focus on segmentation accuracy, entirely overlooking reliability and robustness. These qualities, which ensure consistent performance under diverse conditions (robustness) and well-calibrated model confidences as well as meaningful uncertainties (reliability), are essential for safety-critical applications like autonomous driving, where models must handle unpredictable environments and avoid sudden failures at all costs. To address this gap, we introduce the Reliable Segmentation Score (RSS), a novel metric that combines predictive accuracy, calibration, and uncertainty quality measures via a harmonic mean. RSS penalizes deficiencies in any of its components, providing an easy and intuitive way of holistically judging segmentation models. Comprehensive evaluations of UniMatchV2 against its predecessor and a supervised baseline show that semi-supervised methods often trade reliability for accuracy. While out-of-domain evaluations demonstrate UniMatchV2's robustness, they further expose persistent reliability shortcomings. We advocate for a shift in evaluation protocols toward more holistic metrics like RSS to better align semi-supervised learning research with real-world deployment needs. |
| 2025-06-06 | [Bayesian Persuasion as a Bargaining Game](http://arxiv.org/abs/2506.05876v1) | Yue Lin, Shuhui Zhu et al. | Bayesian persuasion, an extension of cheap-talk communication, involves an informed sender committing to a signaling scheme to influence a receiver's actions. Compared to cheap talk, this sender's commitment enables the receiver to verify the incentive compatibility of signals beforehand, facilitating cooperation. While effective in one-shot scenarios, Bayesian persuasion faces computational complexity (NP-hardness) when extended to long-term interactions, where the receiver may adopt dynamic strategies conditional on past outcomes and future expectations. To address this complexity, we introduce the bargaining perspective, which allows: (1) a unified framework and well-structured solution concept for long-term persuasion, with desirable properties such as fairness and Pareto efficiency; (2) a clear distinction between two previously conflated advantages: the sender's informational advantage and first-proposer advantage. With only modest modifications to the standard setting, this perspective makes explicit the common knowledge of the game structure and grants the receiver comparable commitment capabilities, thereby reinterpreting classic one-sided persuasion as a balanced information bargaining framework. The framework is validated through a two-stage validation-and-inference paradigm: We first demonstrate that GPT-o3 and DeepSeek-R1, out of publicly available LLMs, reliably handle standard tasks; We then apply them to persuasion scenarios to test that the outcomes align with what our information-bargaining framework suggests. All code, results, and terminal logs are publicly available at github.com/YueLin301/InformationBargaining. |
| 2025-06-06 | [Towards Next-Generation Intelligent Maintenance: Collaborative Fusion of Large and Small Models](http://arxiv.org/abs/2506.05854v1) | Xiaoyi Yuan, Qiming Huang et al. | With the rapid advancement of intelligent technologies, collaborative frameworks integrating large and small models have emerged as a promising approach for enhancing industrial maintenance. However, several challenges persist, including limited domain adaptability, insufficient real-time performance and reliability, high integration complexity, and difficulties in knowledge representation and fusion. To address these issues, an intelligent maintenance framework for industrial scenarios is proposed. This framework adopts a five-layer architecture and integrates the precise computational capabilities of domain-specific small models with the cognitive reasoning, knowledge integration, and interactive functionalities of large language models. The objective is to achieve more accurate, intelligent, and efficient maintenance in industrial applications. Two realistic implementations, involving the maintenance of telecommunication equipment rooms and the intelligent servicing of energy storage power stations, demonstrate that the framework significantly enhances maintenance efficiency. |
| 2025-06-06 | [Properties of UTxO Ledgers and Programs Implemented on Them](http://arxiv.org/abs/2506.05832v1) | Polina Vinogradova, Alexey Sorokin | Trace-based properties are the gold standard for program behaviour analysis. One of the domains of application of this type of analysis is cryptocurrency ledgers, both for the purpose of analyzing the behaviour of the ledger itself, and any user-defined programs called by it, known as smart contracts. The (extended) UTxO ledger model is a kind of ledger model where all smart contract code is stateless, and additional work must be done to model stateful programs. We formalize the application of trace-based analysis to UTxO ledgers and contracts, expressing it in the languages of topology, as well as graph and category theory. To describe valid traces of UTxO ledger executions, and their relation to the behaviour of stateful programs implemented on the ledger, we define a category of simple graphs, infinite paths in which form an ultra-metric space. Maps in this category are arbitrary partial sieve-define homomorphisms of simple graphs. Programs implemented on the ledger correspond to non-expanding maps out of the graph of valid UTxO execution traces. We reason about safety properties in this framework, and prove properties of valid UTxO ledger traces. |
| 2025-06-06 | [FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](http://arxiv.org/abs/2506.05828v1) | Zichen Tang, Haihong E et al. | We introduce FinanceReasoning, a novel benchmark designed to evaluate the reasoning capabilities of large reasoning models (LRMs) in financial numerical reasoning problems. Compared to existing benchmarks, our work provides three key advancements. (1) Credibility: We update 15.6% of the questions from four public datasets, annotating 908 new questions with detailed Python solutions and rigorously refining evaluation standards. This enables an accurate assessment of the reasoning improvements of LRMs. (2) Comprehensiveness: FinanceReasoning covers 67.8% of financial concepts and formulas, significantly surpassing existing datasets. Additionally, we construct 3,133 Python-formatted functions, which enhances LRMs' financial reasoning capabilities through refined knowledge (e.g., 83.2% $\rightarrow$ 91.6% for GPT-4o). (3) Challenge: Models are required to apply multiple financial formulas for precise numerical reasoning on 238 Hard problems. The best-performing model (i.e., OpenAI o1 with PoT) achieves 89.1% accuracy, yet LRMs still face challenges in numerical precision. We demonstrate that combining Reasoner and Programmer models can effectively enhance LRMs' performance (e.g., 83.2% $\rightarrow$ 87.8% for DeepSeek-R1). Our work paves the way for future research on evaluating and improving LRMs in domain-specific complex reasoning tasks. |
| 2025-06-06 | [Towards Mixed-Criticality Software Architectures for Centralized HPC Platforms in Software-Defined Vehicles: A Systematic Literature Review](http://arxiv.org/abs/2506.05822v1) | Lucas Mauser, Eva Zimmermann et al. | Centralized electrical/electronic architectures and High-Performance Computers (HPCs) are redefining automotive software development, challenging traditional microcontroller-based approaches. Ensuring real-time, safety, and scalability in software-defined vehicles necessitates reevaluating how mixed-criticality software is integrated into centralized architectures. While existing research on automotive SoftWare Architectures (SWAs) is relevant to the industry, it often lacks validation through systematic, empirical methods. To address this gap, we conduct a systematic literature review focusing on automotive mixed-criticality SWAs. Our goal is to provide practitioner-oriented guidelines that assist automotive software architects and developers design centralized, mixed-criticality SWAs based on a rigorous and transparent methodology. First, we set up a systematic review protocol grounded in established guidelines. Second, we apply this protocol to identify relevant studies. Third, we extract key functional domains, constraints, and enabling technologies that drive changes in automotive SWAs, thereby assessing the protocol's effectiveness. Additionally, we extract techniques, architectural patterns, and design practices for integrating mixed-criticality requirements into HPC-based SWAs, further demonstrating the protocol's applicability. Based on these insights, we propose an exemplary SWA for a microprocessor-based system-on-chip. In conclusion, this study provides a structured approach to explore and realize mixed-criticality software integration for next-generation automotive SWAs, offering valuable insights for industry and research applications. |
| 2025-06-06 | [Mechanisms of Afterglow and Thermally Stimulated Luminescence in UV-irradiated InP/ZnS Quantum Dots](http://arxiv.org/abs/2506.05792v1) | S. S. Savchenko, A. O. Shilov et al. | Indium phosphide-based quantum dots (QDs) are a potential material for designing optoelectronic devices, owing their adjustable spectral parameters over the entire visible range, as well as their high biocompatibility and environmental safety. Concurrently, they exhibit structural defects, the rectification of which is crucial for enhancing their optical properties. The present work explores, for the first time, the low-temperature afterglow (AG) and spectrally resolved thermally stimulated luminescence (TSL) of UV-irradiated colloidal core/shell InP/ZnS QDs in the range of 7-340 K. It is shown that, when localized during irradiation and released after additional stimulation, charge carriers recombine involving defect centers based on indium and phosphorus dangling bonds. The mechanisms of the observed luminescent phenomena can be caused by both thermal activation and tunneling processes. By means of the initial rise method, the formalism of general-order kinetics, and the analytical description using the Lambert W function, we have analyzed the kinetic features of possible thermally stimulated mechanisms. We have also estimated the energy characteristics of appropriate trapping centers. A low rate of charge carriers recapture is revealed for InP/ZnS QDs. Active traps in nanocrystals of different sizes are characterized by close values of activation energy in the 26-31 meV range. The current paper discloses new horizons for exploiting TSL approaches to study the properties of local defective states in the energy structure of colloidal QDs, which can contribute to the development of targeted synthesis of nanocrystals with tunable temperature sensitivity for optoelectronic and sensor applications. |
| 2025-06-06 | [Robust sensor fusion against on-vehicle sensor staleness](http://arxiv.org/abs/2506.05780v1) | Meng Fan, Yifan Zuo et al. | Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions. |

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



