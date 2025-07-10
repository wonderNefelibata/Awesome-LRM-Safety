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
| 2025-07-09 | [When Context Is Not Enough: Modeling Unexplained Variability in Car-Following Behavior](http://arxiv.org/abs/2507.07012v1) | Chengyuan Zhang, Zhengbing He et al. | Modeling car-following behavior is fundamental to microscopic traffic simulation, yet traditional deterministic models often fail to capture the full extent of variability and unpredictability in human driving. While many modern approaches incorporate context-aware inputs (e.g., spacing, speed, relative speed), they frequently overlook structured stochasticity that arises from latent driver intentions, perception errors, and memory effects -- factors that are not directly observable from context alone. To fill the gap, this study introduces an interpretable stochastic modeling framework that captures not only context-dependent dynamics but also residual variability beyond what context can explain. Leveraging deep neural networks integrated with nonstationary Gaussian processes (GPs), our model employs a scenario-adaptive Gibbs kernel to learn dynamic temporal correlations in acceleration decisions, where the strength and duration of correlations between acceleration decisions evolve with the driving context. This formulation enables a principled, data-driven quantification of uncertainty in acceleration, speed, and spacing, grounded in both observable context and latent behavioral variability. Comprehensive experiments on the naturalistic vehicle trajectory dataset collected from the German highway, i.e., the HighD dataset, demonstrate that the proposed stochastic simulation method within this framework surpasses conventional methods in both predictive performance and interpretable uncertainty quantification. The integration of interpretability and accuracy makes this framework a promising tool for traffic analysis and safety-critical applications. |
| 2025-07-09 | [Dataset and Benchmark for Enhancing Critical Retained Foreign Object Detection](http://arxiv.org/abs/2507.06937v1) | Yuli Wang, Victoria R. Shi et al. | Critical retained foreign objects (RFOs), including surgical instruments like sponges and needles, pose serious patient safety risks and carry significant financial and legal implications for healthcare institutions. Detecting critical RFOs using artificial intelligence remains challenging due to their rarity and the limited availability of chest X-ray datasets that specifically feature critical RFOs cases. Existing datasets only contain non-critical RFOs, like necklace or zipper, further limiting their utility for developing clinically impactful detection algorithms. To address these limitations, we introduce "Hopkins RFOs Bench", the first and largest dataset of its kind, containing 144 chest X-ray images of critical RFO cases collected over 18 years from the Johns Hopkins Health System. Using this dataset, we benchmark several state-of-the-art object detection models, highlighting the need for enhanced detection methodologies for critical RFO cases. Recognizing data scarcity challenges, we further explore image synthetic methods to bridge this gap. We evaluate two advanced synthetic image methods, DeepDRR-RFO, a physics-based method, and RoentGen-RFO, a diffusion-based method, for creating realistic radiographs featuring critical RFOs. Our comprehensive analysis identifies the strengths and limitations of each synthetic method, providing insights into effectively utilizing synthetic data to enhance model training. The Hopkins RFOs Bench and our findings significantly advance the development of reliable, generalizable AI-driven solutions for detecting critical RFOs in clinical chest X-rays. |
| 2025-07-09 | [Robust and Safe Traffic Sign Recognition using N-version with Weighted Voting](http://arxiv.org/abs/2507.06907v1) | Linyun Gao, Qiang Wen et al. | Autonomous driving is rapidly advancing as a key application of machine learning, yet ensuring the safety of these systems remains a critical challenge. Traffic sign recognition, an essential component of autonomous vehicles, is particularly vulnerable to adversarial attacks that can compromise driving safety. In this paper, we propose an N-version machine learning (NVML) framework that integrates a safety-aware weighted soft voting mechanism. Our approach utilizes Failure Mode and Effects Analysis (FMEA) to assess potential safety risks and assign dynamic, safety-aware weights to the ensemble outputs. We evaluate the robustness of three-version NVML systems employing various voting mechanisms against adversarial samples generated using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Experimental results demonstrate that our NVML approach significantly enhances the robustness and safety of traffic sign recognition systems under adversarial conditions. |
| 2025-07-09 | [Developing and Maintaining an Open-Source Repository of AI Evaluations: Challenges and Insights](http://arxiv.org/abs/2507.06893v1) | Alexandra Abbas, Celia Waggoner et al. | AI evaluations have become critical tools for assessing large language model capabilities and safety. This paper presents practical insights from eight months of maintaining $inspect\_evals$, an open-source repository of 70+ community-contributed AI evaluations. We identify key challenges in implementing and maintaining AI evaluations and develop solutions including: (1) a structured cohort management framework for scaling community contributions, (2) statistical methodologies for optimal resampling and cross-model comparison with uncertainty quantification, and (3) systematic quality control processes for reproducibility. Our analysis reveals that AI evaluation requires specialized infrastructure, statistical rigor, and community coordination beyond traditional software development practices. |
| 2025-07-09 | [Stream Function-Based Navigation for Complex Quadcopter Obstacle Avoidance](http://arxiv.org/abs/2507.06787v1) | Sean Smith, Emmanuel Witrant et al. | This article presents a novel stream function-based navigational control system for obstacle avoidance, where obstacles are represented as two-dimensional (2D) rigid surfaces in inviscid, incompressible flows. The approach leverages the vortex panel method (VPM) and incorporates safety margins to control the stream function and flow properties around virtual surfaces, enabling navigation in complex, partially observed environments using real-time sensing. To address the limitations of the VPM in managing relative distance and avoiding rapidly accelerating obstacles at close proximity, the system integrates a model predictive controller (MPC) based on higher-order control barrier functions (HOCBF). This integration incorporates VPM trajectory generation, state estimation, and constraint handling into a receding-horizon optimization problem. The 2D rigid surfaces are enclosed using minimum bounding ellipses (MBEs), while an adaptive Kalman filter (AKF) captures and predicts obstacle dynamics, propagating these estimates into the MPC-HOCBF for rapid avoidance maneuvers. Evaluation is conducted using a PX4-powered Clover drone Gazebo simulator and real-time experiments involving a COEX Clover quadcopter equipped with a 360 degree LiDAR sensor. |
| 2025-07-09 | [Integrating Perceptions: A Human-Centered Physical Safety Model for Human-Robot Interaction](http://arxiv.org/abs/2507.06700v1) | Pranav Pandey, Ramviyas Parasuraman et al. | Ensuring safety in human-robot interaction (HRI) is essential to foster user trust and enable the broader adoption of robotic systems. Traditional safety models primarily rely on sensor-based measures, such as relative distance and velocity, to assess physical safety. However, these models often fail to capture subjective safety perceptions, which are shaped by individual traits and contextual factors. In this paper, we introduce and analyze a parameterized general safety model that bridges the gap between physical and perceived safety by incorporating a personalization parameter, $\rho$, into the safety measurement framework to account for individual differences in safety perception. Through a series of hypothesis-driven human-subject studies in a simulated rescue scenario, we investigate how emotional state, trust, and robot behavior influence perceived safety. Our results show that $\rho$ effectively captures meaningful individual differences, driven by affective responses, trust in task consistency, and clustering into distinct user types. Specifically, our findings confirm that predictable and consistent robot behavior as well as the elicitation of positive emotional states, significantly enhance perceived safety. Moreover, responses cluster into a small number of user types, supporting adaptive personalization based on shared safety models. Notably, participant role significantly shapes safety perception, and repeated exposure reduces perceived safety for participants in the casualty role, emphasizing the impact of physical interaction and experiential change. These findings highlight the importance of adaptive, human-centered safety models that integrate both psychological and behavioral dimensions, offering a pathway toward more trustworthy and effective HRI in safety-critical domains. |
| 2025-07-09 | [Google Search Advertising after Dobbs v. Jackson](http://arxiv.org/abs/2507.06640v1) | Yelena Mejova, Ronald E. Robertson et al. | Search engines have become the gateway to information, products, and services, including those concerning healthcare. Access to reproductive health has been especially complicated in the wake of the 2022 Dobbs v. Jackson decision by the Supreme Court of the United States, splintering abortion regulations among the states. In this study, we performed an audit of the advertisements shown to Google Search users seeking information about abortion across the United States during the year following the Dobbs decision. We found that Crisis Pregnancy Centers (CPCs) -- organizations that target women with unexpected or "crisis" pregnancies, but do not provide abortions -- accounted for 47% of advertisements, whereas abortion clinics -- for 30%. Advertisements from CPCs were often returned for queries concerning information and safety. The type of advertisements returned, however, varied widely within each state, with Arizona returning the most advertisements from abortion clinics and other pro-choice organizations, and Minnesota the least. The proportion of pro-choice vs. anti-choice advertisements returned also varied over time, but estimates from Staggered Augmented Synthetic Control Methods did not indicate that changes in advertisement results were attributable to changes in state abortion laws. Our findings raise questions about the access to accurate medical information across the U.S. and point to a need for further examination of search engine advertisement policies and geographical bias. |
| 2025-07-09 | [Stacked Intelligent Metasurfaces-Aided eVTOL Delay Sensitive Communications](http://arxiv.org/abs/2507.06632v1) | Liyuan Chen, Kai Xiong et al. | With rapid urbanization and increasing population density, urban traffic congestion has become a critical issue, and traditional ground transportation methods are no longer sufficient to address it effectively. To tackle this challenge, the concept of Advanced Air Mobility (AAM) has emerged, aiming to utilize low-altitude airspace to establish a three-dimensional transportation system. Among various components of the AAM system, electric vertical take-off and landing (eVTOL) aircraft plays a pivotal role due to their flexibility and efficiency. However, the immaturity of Ultra Reliable Low Latency Communication (URLLC) technologies poses significant challenges to safety-critical AAM operations. Specifically, existing Stacked Intelligent Metasurfaces (SIM)-based eVTOL systems lack rigorous mathematical frameworks to quantify probabilistic delay bounds under dynamic air traffic patterns, a prerequisite for collision avoidance and airspace management. To bridge this gap, we employ network calculus tools to derive the probabilistic upper bound on communication delay in the AAM system for the first time. Furthermore, we formulate a complex non-convex optimization problem that jointly minimizes the probabilistic delay bound and the propagation delay. To solve this problem efficiently, we propose a solution based on the Block Coordinate Descent (BCD) algorithm and Semidefinite Relaxation (SDR) method. In addition, we conduct a comprehensive analysis of how various factors impact regret and transmission rate, and explore the influence of varying load intensity and total delay on the probabilistic delay bound. |
| 2025-07-09 | [Q-STAC: Q-Guided Stein Variational Model Predictive Actor-Critic](http://arxiv.org/abs/2507.06625v1) | Shizhe Cai, Jayadeep Jacob et al. | Deep reinforcement learning has shown remarkable success in continuous control tasks, yet often requires extensive training data, struggles with complex, long-horizon planning, and fails to maintain safety constraints during operation. Meanwhile, Model Predictive Control (MPC) offers explainability and constraint satisfaction, but typically yields only locally optimal solutions and demands careful cost function design. This paper introduces the Q-guided STein variational model predictive Actor-Critic (Q-STAC), a novel framework that bridges these approaches by integrating Bayesian MPC with actor-critic reinforcement learning through constrained Stein Variational Gradient Descent (SVGD). Our method optimizes control sequences directly using learned Q-values as objectives, eliminating the need for explicit cost function design while leveraging known system dynamics to enhance sample efficiency and ensure control signals remain within safe boundaries. Extensive experiments on 2D navigation and robotic manipulation tasks demonstrate that Q-STAC achieves superior sample efficiency, robustness, and optimality compared to state-of-the-art algorithms, while maintaining the high expressiveness of policy distributions. Experiment videos are available on our website: https://sites.google.com/view/q-stac |
| 2025-07-09 | [Finding Compiler Bugs through Cross-Language Code Generator and Differential Testing](http://arxiv.org/abs/2507.06584v1) | Qiong Feng, Xiaotian Ma et al. | Compilers play a central role in translating high-level code into executable programs, making their correctness essential for ensuring code safety and reliability. While extensive research has focused on verifying the correctness of compilers for single-language compilation, the correctness of cross-language compilation - which involves the interaction between two languages and their respective compilers - remains largely unexplored. To fill this research gap, we propose CrossLangFuzzer, a novel framework that introduces a universal intermediate representation (IR) for JVM-based languages and automatically generates cross-language test programs with diverse type parameters and complex inheritance structures. After generating the initial IR, CrossLangFuzzer applies three mutation techniques - LangShuffler, FunctionRemoval, and TypeChanger - to enhance program diversity. By evaluating both the original and mutated programs across multiple compiler versions, CrossLangFuzzer successfully uncovered 10 confirmed bugs in the Kotlin compiler, 4 confirmed bugs in the Groovy compiler, 7 confirmed bugs in the Scala 3 compiler, 2 confirmed bugs in the Scala 2 compiler, and 1 confirmed bug in the Java compiler. Among all mutators, TypeChanger is the most effective, detecting 11 of the 24 compiler bugs. Furthermore, we analyze the symptoms and root causes of cross-compilation bugs, examining the respective responsibilities of language compilers when incorrect behavior occurs during cross-language compilation. To the best of our knowledge, this is the firstwork specifically focused on identifying and diagnosing compiler bugs in cross-language compilation scenarios. Our research helps to understand these challenges and contributes to improving compiler correctness in multi-language environments. |
| 2025-07-09 | [AI Space Cortex: An Experimental System for Future Era Space Exploration](http://arxiv.org/abs/2507.06574v1) | Thomas Touma, Ersin Daş et al. | Our Robust, Explainable Autonomy for Scientific Icy Moon Operations (REASIMO) effort contributes to NASA's Concepts for Ocean worlds Life Detection Technology (COLDTech) program, which explores science platform technologies for ocean worlds such as Europa and Enceladus. Ocean world missions pose significant operational challenges. These include long communication lags, limited power, and lifetime limitations caused by radiation damage and hostile conditions. Given these operational limitations, onboard autonomy will be vital for future Ocean world missions. Besides the management of nominal lander operations, onboard autonomy must react appropriately in the event of anomalies. Traditional spacecraft rely on a transition into 'safe-mode' in which non-essential components and subsystems are powered off to preserve safety and maintain communication with Earth. For a severely time-limited Ocean world mission, resolutions to these anomalies that can be executed without Earth-in-the-loop communication and associated delays are paramount for completion of the mission objectives and science goals. To address these challenges, the REASIMO effort aims to demonstrate a robust level of AI-assisted autonomy for such missions, including the ability to detect and recover from anomalies, and to perform missions based on pre-trained behaviors rather than hard-coded, predetermined logic like all prior space missions. We developed an AI-assisted, personality-driven, intelligent framework for control of an Ocean world mission by combining a mix of advanced technologies. To demonstrate the capabilities of the framework, we perform tests of autonomous sampling operations on a lander-manipulator testbed at the NASA Jet Propulsion Laboratory, approximating possible surface conditions such a mission might encounter. |
| 2025-07-09 | [From large-eddy simulations to deep learning: A U-net model for fast urban canopy flow predictions](http://arxiv.org/abs/2507.06533v1) | Themistoklis Vargiemezis, Catherine Gorlé | Accurate prediction of wind flow fields in urban canopies is crucial for ensuring pedestrian comfort, safety, and sustainable urban design. Traditional methods using wind tunnels and Computational Fluid Dynamics, such as Large-Eddy Simulations (LES), are limited by high costs, computational demands, and time requirements. This study presents a deep neural network (DNN) approach for fast and accurate predictions of urban wind flow fields, reducing computation time from an order of 10 hours on 32 CPUs for one LES evaluation to an order of 1 second on a single GPU using the DNN model. We employ a U-Net architecture trained on LES data including 252 synthetic urban configurations at seven wind directions ($0^{o}$ to $90^{o}$ in $15^{o}$ increments). The model predicts two key quantities of interest: mean velocity magnitude and streamwise turbulence intensity, at multiple heights within the urban canopy. The U-net uses 2D building representations augmented with signed distance functions and their gradients as inputs, forming a $256\times256\times9$ tensor. In addition, a Spatial Attention Module is used for feature transfer through skip connections. The loss function combines the root-mean-square error of predictions, their gradient magnitudes, and L2 regularization. Model evaluation on 50 test cases demonstrates high accuracy with an overall mean relative error of 9.3% for velocity magnitude and 5.2% for turbulence intensity. This research shows the potential of deep learning approaches to provide fast, accurate urban wind assessments essential for creating comfortable and safe urban environments. Code is available at https://github.com/tvarg/Urban-FlowUnet.git |
| 2025-07-09 | [What Demands Attention in Urban Street Scenes? From Scene Understanding towards Road Safety: A Survey of Vision-driven Datasets and Studies](http://arxiv.org/abs/2507.06513v1) | Yaoqi Huang, Julie Stephany Berrio et al. | Advances in vision-based sensors and computer vision algorithms have significantly improved the analysis and understanding of traffic scenarios. To facilitate the use of these improvements for road safety, this survey systematically categorizes the critical elements that demand attention in traffic scenarios and comprehensively analyzes available vision-driven tasks and datasets. Compared to existing surveys that focus on isolated domains, our taxonomy categorizes attention-worthy traffic entities into two main groups that are anomalies and normal but critical entities, integrating ten categories and twenty subclasses. It establishes connections between inherently related fields and provides a unified analytical framework. Our survey highlights the analysis of 35 vision-driven tasks and comprehensive examinations and visualizations of 73 available datasets based on the proposed taxonomy. The cross-domain investigation covers the pros and cons of each benchmark with the aim of providing information on standards unification and resource optimization. Our article concludes with a systematic discussion of the existing weaknesses, underlining the potential effects and promising solutions from various perspectives. The integrated taxonomy, comprehensive analysis, and recapitulatory tables serve as valuable contributions to this rapidly evolving field by providing researchers with a holistic overview, guiding strategic resource selection, and highlighting critical research gaps. |
| 2025-07-09 | [Towards LLM-based Root Cause Analysis of Hardware Design Failures](http://arxiv.org/abs/2507.06512v1) | Siyu Qiu, Muzhi Wang et al. | With advances in large language models (LLMs), new opportunities have emerged to develop tools that support the digital hardware design process. In this work, we explore how LLMs can assist with explaining the root cause of design issues and bugs that are revealed during synthesis and simulation, a necessary milestone on the pathway towards widespread use of LLMs in the hardware design process and for hardware security analysis. We find promising results: for our corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model reached a correct determination 100% of the time under pass@5 scoring, with other state of the art models and configurations usually achieving more than 80% performance and more than 90% when assisted with retrieval-augmented generation. |
| 2025-07-09 | [On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](http://arxiv.org/abs/2507.06489v1) | Stephen Obadinma, Xiaodan Zhu | Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses. |
| 2025-07-09 | [Foundation Model Self-Play: Open-Ended Strategy Innovation via Foundation Models](http://arxiv.org/abs/2507.06466v1) | Aaron Dharna, Cong Lu et al. | Multi-agent interactions have long fueled innovation, from natural predator-prey dynamics to the space race. Self-play (SP) algorithms try to harness these dynamics by pitting agents against ever-improving opponents, thereby creating an implicit curriculum toward learning high-quality solutions. However, SP often fails to produce diverse solutions and can get stuck in locally optimal behaviors. We introduce Foundation-Model Self-Play (FMSP), a new direction that leverages the code-generation capabilities and vast knowledge of foundation models (FMs) to overcome these challenges by leaping across local optima in policy space. We propose a family of approaches: (1) \textbf{Vanilla Foundation-Model Self-Play (vFMSP)} continually refines agent policies via competitive self-play; (2) \textbf{Novelty-Search Self-Play (NSSP)} builds a diverse population of strategies, ignoring performance; and (3) the most promising variant, \textbf{Quality-Diveristy Self-Play (QDSP)}, creates a diverse set of high-quality policies by combining the diversity of NSSP and refinement of vFMSP. We evaluate FMSPs in Car Tag, a continuous-control pursuer-evader setting, and in Gandalf, a simple AI safety simulation in which an attacker tries to jailbreak an LLM's defenses. In Car Tag, FMSPs explore a wide variety of reinforcement learning, tree search, and heuristic-based methods, to name just a few. In terms of discovered policy quality, \ouralgo and vFMSP surpass strong human-designed strategies. In Gandalf, FMSPs can successfully automatically red-team an LLM, breaking through and jailbreaking six different, progressively stronger levels of defense. Furthermore, FMSPs can automatically proceed to patch the discovered vulnerabilities. Overall, FMSPs represent a promising new research frontier of improving self-play with foundation models, opening fresh paths toward more creative and open-ended strategy discovery |
| 2025-07-09 | [Evaluating Efficiency and Novelty of LLM-Generated Code for Graph Analysis](http://arxiv.org/abs/2507.06463v1) | Atieh Barati Nia, Mohammad Dindoost et al. | Large Language Models (LLMs) are increasingly used to automate software development, yet most prior evaluations focus on functional correctness or high-level languages such as Python. We present the first systematic study of LLMs' ability to generate efficient C implementations of graph-analysis routines--code that must satisfy the stringent runtime and memory constraints. Eight state-of-the-art models (OpenAI ChatGPT o3 and o4-mini-high, Anthropic Claude 4 Sonnet and Sonnet Extended, Google Gemini 2.5 Flash and Pro, xAI Grok 3-Think, and DeepSeek DeepThink R1) are benchmarked by two distinct approaches. The first approach checks the ability of LLMs in generating an algorithm outperforming other present algorithms in the benchmark. The second approach evaluates the ability of LLMs to generate graph algorithms for integration into the benchmark. Results show that Claude Sonnet 4 Extended achieves the best result in the case of ready-to-use code generation and efficiency, outperforming human-written baselines in triangle counting. The study confirms that contemporary LLMs excel at optimizing and integrating established algorithms but not inventing novel techniques. We provide prompts, the first approach's generated code, and measurement scripts to foster reproducible research. |
| 2025-07-08 | [VisioPath: Vision-Language Enhanced Model Predictive Control for Safe Autonomous Navigation in Mixed Traffic](http://arxiv.org/abs/2507.06441v1) | Shanting Wang, Panagiotis Typaldos et al. | In this paper, we introduce VisioPath, a novel framework combining vision-language models (VLMs) with model predictive control (MPC) to enable safe autonomous driving in dynamic traffic environments. The proposed approach leverages a bird's-eye view video processing pipeline and zero-shot VLM capabilities to obtain structured information about surrounding vehicles, including their positions, dimensions, and velocities. Using this rich perception output, we construct elliptical collision-avoidance potential fields around other traffic participants, which are seamlessly integrated into a finite-horizon optimal control problem for trajectory planning. The resulting trajectory optimization is solved via differential dynamic programming with an adaptive regularization scheme and is embedded in an event-triggered MPC loop. To ensure collision-free motion, a safety verification layer is incorporated in the framework that provides an assessment of potential unsafe trajectories. Extensive simulations in Simulation of Urban Mobility (SUMO) demonstrate that VisioPath outperforms conventional MPC baselines across multiple metrics. By combining modern AI-driven perception with the rigorous foundation of optimal control, VisioPath represents a significant step forward in safe trajectory planning for complex traffic systems. |
| 2025-07-08 | [HEMA: A Hands-on Exploration Platform for MEMS Sensor Attacks](http://arxiv.org/abs/2507.06439v1) | Bhagawat Baanav Yedla Ravi, Md Rafiul Kabir et al. | Automotive safety and security are paramount in the rapidly advancing landscape of vehicular technology. Building safe and secure vehicles demands a profound understanding of automotive systems, particularly in safety and security. Traditional learning approaches, such as reading materials or observing demonstrations, often fail to provide the practical, hands-on experience essential for developing this expertise. For novice users, gaining access to automotive-grade systems and mastering their associated hardware and software can be challenging and overwhelming. In this paper, we present a novel, affordable, and flexible exploration platform, \hema, that enables users to gain practical, hands-on insights into the security compromises of micro-electromechanical systems (MEMS) sensors, a critical component in modern ADAS systems. Furthermore, we discuss the unique challenges and design considerations involved in creating such a platform, emphasizing its role in enhancing the understanding of automotive safety and security. This framework serves as an invaluable resource for educators, researchers, and practitioners striving to build expertise in the field. |
| 2025-07-08 | [Deprecating Benchmarks: Criteria and Framework](http://arxiv.org/abs/2507.06434v1) | Ayrton San Joaquin, Rokas Gipiškis et al. | As frontier artificial intelligence (AI) models rapidly advance, benchmarks are integral to comparing different models and measuring their progress in different task-specific domains. However, there is a lack of guidance on when and how benchmarks should be deprecated once they cease to effectively perform their purpose. This risks benchmark scores over-valuing model capabilities, or worse, obscuring capabilities and safety-washing. Based on a review of benchmarking practices, we propose criteria to decide when to fully or partially deprecate benchmarks, and a framework for deprecating benchmarks. Our work aims to advance the state of benchmarking towards rigorous and quality evaluations, especially for frontier models, and our recommendations are aimed to benefit benchmark developers, benchmark users, AI governance actors (across governments, academia, and industry panels), and policy makers. |

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



