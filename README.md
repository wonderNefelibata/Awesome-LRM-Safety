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
| 2025-05-05 | [Stabilizing dark matter with quantum scale symmetry](http://arxiv.org/abs/2505.02803v1) | Abhishek Chikkaballi, Kamila Kowalska et al. | In the context of gauge-Yukawa theories with trans-Planckian asymptotic safety, quantum scale symmetry can prevent the appearance in the Lagrangian of couplings that would otherwise be allowed by the gauge symmetry. Such couplings correspond to irrelevant Gaussian fixed points of the renormalization group flow. Their absence in the theory implies that different sectors of the gauge-Yukawa theory are secluded from one another, in similar fashion to the effects of a global or a discrete symmetry. As an example, we impose the trans-Planckian scale symmetry on a model of Grand Unification based on the gauge group SU(6), showing that it leads to the emergence of several fermionic WIMP dark matter candidates whose coupling strengths are entirely predicted by the UV completion. |
| 2025-05-05 | [A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law](http://arxiv.org/abs/2505.02665v1) | Qianjun Pan, Wenkai Ji et al. | This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems. |
| 2025-05-05 | [Faithful and secure distributed quantum sensing under general-coherent attacks](http://arxiv.org/abs/2505.02620v1) | G. Bizzarri, M. Barbieri et al. | Quantum metrology and cryptography can be combined in a distributed and/or remote sensing setting, where distant end-users with limited quantum capabilities can employ quantum states, transmitted by a quantum-powerful provider via a quantum network, to perform quantum-enhanced parameter estimation in a private fashion. Previous works on the subject have been limited by restricted assumptions on the capabilities of a potential eavesdropper and the use of abort-based protocols that prevent a simple practical realization. Here we introduce, theoretically analyze, and experimentally demonstrate single- and two-way protocols for distributed sensing combining several unique and desirable features: (i) a safety-threshold mechanism that allows the protocol to proceed in low-noise cases and quantifying the potential tampering with respect to the ideal estimation procedure, effectively paving the way for wide-spread practical realizations; (ii) equivalence of entanglement-based and mutually-unbiased-bases-based formulations; (iii) robustness against collective attacks via a LOCC-de-Finetti theorem, for the first time to our knowledge. Finally, we demonstrate our protocols in a photonic-based implementation, observing that the possibility of guaranteeing a safety threshold may come at a significant price in terms of the estimation bias, potentially overestimating the effect of tampering in practical settings. |
| 2025-05-05 | [LiDAR-Inertial SLAM-Based Navigation and Safety-Oriented AI-Driven Control System for Skid-Steer Robots](http://arxiv.org/abs/2505.02598v1) | Mehdi Heydari Shahna, Eemil Haaparanta et al. | Integrating artificial intelligence (AI) and stochastic technologies into the mobile robot navigation and control (MRNC) framework while adhering to rigorous safety standards presents significant challenges. To address these challenges, this paper proposes a comprehensively integrated MRNC framework for skid-steer wheeled mobile robots (SSWMRs), in which all components are actively engaged in real-time execution. The framework comprises: 1) a LiDAR-inertial simultaneous localization and mapping (SLAM) algorithm for estimating the current pose of the robot within the built map; 2) an effective path-following control system for generating desired linear and angular velocity commands based on the current pose and the desired pose; 3) inverse kinematics for transferring linear and angular velocity commands into left and right side velocity commands; and 4) a robust AI-driven (RAID) control system incorporating a radial basis function network (RBFN) with a new adaptive algorithm to enforce in-wheel actuation systems to track each side motion commands. To further meet safety requirements, the proposed RAID control within the MRNC framework of the SSWMR constrains AI-generated tracking performance within predefined overshoot and steady-state error limits, while ensuring robustness and system stability by compensating for modeling errors, unknown RBF weights, and external forces. Experimental results verify the proposed MRNC framework performance for a 4,836 kg SSWMR operating on soft terrain. |
| 2025-05-05 | [Point Cloud Recombination: Systematic Real Data Augmentation Using Robotic Targets for LiDAR Perception Validation](http://arxiv.org/abs/2505.02476v1) | Hubert Padusinski, Christian Steinhauser et al. | The validation of LiDAR-based perception of intelligent mobile systems operating in open-world applications remains a challenge due to the variability of real environmental conditions. Virtual simulations allow the generation of arbitrary scenes under controlled conditions but lack physical sensor characteristics, such as intensity responses or material-dependent effects. In contrast, real-world data offers true sensor realism but provides less control over influencing factors, hindering sufficient validation. Existing approaches address this problem with augmentation of real-world point cloud data by transferring objects between scenes. However, these methods do not consider validation and remain limited in controllability because they rely on empirical data. We solve these limitations by proposing Point Cloud Recombination, which systematically augments captured point cloud scenes by integrating point clouds acquired from physical target objects measured in controlled laboratory environments. Thus enabling the creation of vast amounts and varieties of repeatable, physically accurate test scenes with respect to phenomena-aware occlusions with registered 3D meshes. Using the Ouster OS1-128 Rev7 sensor, we demonstrate the augmentation of real-world urban and rural scenes with humanoid targets featuring varied clothing and poses, for repeatable positioning. We show that the recombined scenes closely match real sensor outputs, enabling targeted testing, scalable failure analysis, and improved system safety. By providing controlled yet sensor-realistic data, our method enables trustworthy conclusions about the limitations of specific sensors in compound with their algorithms, e.g., object detection. |
| 2025-05-05 | [A Real-Time Control Barrier Function-Based Safety Filter for Motion Planning with Arbitrary Road Boundary Constraints](http://arxiv.org/abs/2505.02395v1) | Jianye Xu, Chang Che et al. | We present a real-time safety filter for motion planning, such as learning-based methods, using Control Barrier Functions (CBFs), which provides formal guarantees for collision avoidance with road boundaries. A key feature of our approach is its ability to directly incorporate road geometries of arbitrary shape without resorting to conservative overapproximations. We formulate the safety filter as a constrained optimization problem in the form of a Quadratic Program (QP). It achieves safety by making minimal, necessary adjustments to the control actions issued by the nominal motion planner. We validate our safety filter through extensive numerical experiments across a variety of traffic scenarios featuring complex roads. The results confirm its reliable safety and high computational efficiency (execution frequency up to 40 Hz). Code & Video Demo: github.com/bassamlab/SigmaRL |
| 2025-05-05 | [Quantitative Analysis of Performance Drop in DeepSeek Model Quantization](http://arxiv.org/abs/2505.02390v1) | Enbo Zhao, Yi Shen et al. | Recently, there is a high demand for deploying DeepSeek-R1 and V3 locally, possibly because the official service often suffers from being busy and some organizations have data privacy concerns. While single-machine deployment offers infrastructure simplicity, the models' 671B FP8 parameter configuration exceeds the practical memory limits of a standard 8-GPU machine. Quantization is a widely used technique that helps reduce model memory consumption. However, it is unclear what the performance of DeepSeek-R1 and V3 will be after being quantized. This technical report presents the first quantitative evaluation of multi-bitwidth quantization across the complete DeepSeek model spectrum. Key findings reveal that 4-bit quantization maintains little performance degradation versus FP8 while enabling single-machine deployment on standard NVIDIA GPU devices. We further propose DQ3_K_M, a dynamic 3-bit quantization method that significantly outperforms traditional Q3_K_M variant on various benchmarks, which is also comparable with 4-bit quantization (Q4_K_M) approach in most tasks. Moreover, DQ3_K_M supports single-machine deployment configurations for both NVIDIA H100/A100 and Huawei 910B. Our implementation of DQ3\_K\_M is released at https://github.com/UnicomAI/DeepSeek-Eval, containing optimized 3-bit quantized variants of both DeepSeek-R1 and DeepSeek-V3. |
| 2025-05-05 | [RouthSearch: Inferring PID Parameter Specification for Flight Control Program by Coordinate Search](http://arxiv.org/abs/2505.02357v1) | Siao Wang, Zhen Dong et al. | Flight control programs use PID control modules with user-configurable Proportional (P), Integral (I), and Derivative (D) parameters to manage UAV flying behaviors. Users can adjust these PID parameters during flight. However, flight control programs lack sufficient safety checks on user-provided PID parameters, leading to a severe UAV vulnerability - the input validation bug. This occurs when a user misconfigures PID parameters, causing dangerous states like deviation from the expected path, loss of control, or crash.   Prior works use random testing like fuzzing, but these are not effective in the three-dimensional search space of PID parameters. The expensive dynamic execution of UAV tests further hinders random testing performance.   We address PID parameter misconfiguration by combining the Routh-Hurwitz stability criterion with coordinate search, introducing RouthSearch. Instead of ad-hoc identification, RouthSearch principledly determines valid ranges for three-dimensional PID parameters. We first leverage the Routh-Hurwitz Criterion to identify a theoretical PID parameter boundary, then refine it using efficient coordinate search. The determined valid range can filter misconfigured PID parameters from users during flight and help discover logical bugs in flight control programs.   We evaluated RouthSearch across eight flight modes in PX4 and Ardupilot. Results show RouthSearch determines valid ranges with 92.0% accuracy compared to ground truth. RouthSearch discovers 3,853 PID misconfigurations within 48 hours, while the STOA work PGFuzz discovers only 449 sets, significantly outperforming prior works by 8.58 times. Our method also helped detect three bugs in ArduPilot and PX4. |
| 2025-05-05 | [HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking](http://arxiv.org/abs/2505.02322v1) | Runquan Gui, Zhihai Wang et al. | Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview. |
| 2025-05-05 | [What Is AI Safety? What Do We Want It to Be?](http://arxiv.org/abs/2505.02313v1) | Jacqueline Harding, Cameron Domenico Kirk-Giannini | The field of AI safety seeks to prevent or reduce the harms caused by AI systems. A simple and appealing account of what is distinctive of AI safety as a field holds that this feature is constitutive: a research project falls within the purview of AI safety just in case it aims to prevent or reduce the harms caused by AI systems. Call this appealingly simple account The Safety Conception of AI safety. Despite its simplicity and appeal, we argue that The Safety Conception is in tension with at least two trends in the ways AI safety researchers and organizations think and talk about AI safety: first, a tendency to characterize the goal of AI safety research in terms of catastrophic risks from future systems; second, the increasingly popular idea that AI safety can be thought of as a branch of safety engineering. Adopting the methodology of conceptual engineering, we argue that these trends are unfortunate: when we consider what concept of AI safety it would be best to have, there are compelling reasons to think that The Safety Conception is the answer. Descriptively, The Safety Conception allows us to see how work on topics that have historically been treated as central to the field of AI safety is continuous with work on topics that have historically been treated as more marginal, like bias, misinformation, and privacy. Normatively, taking The Safety Conception seriously means approaching all efforts to prevent or mitigate harms from AI systems based on their merits rather than drawing arbitrary distinctions between them. |
| 2025-05-05 | [SafeMate: A Model Context Protocol-Based Multimodal Agent for Emergency Preparedness](http://arxiv.org/abs/2505.02306v1) | Junfeng Jiao, Jihyung Park et al. | Despite the abundance of public safety documents and emergency protocols, most individuals remain ill-equipped to interpret and act on such information during crises. Traditional emergency decision support systems (EDSS) are designed for professionals and rely heavily on static documents like PDFs or SOPs, which are difficult for non-experts to navigate under stress. This gap between institutional knowledge and public accessibility poses a critical barrier to effective emergency preparedness and response.   We introduce SafeMate, a retrieval-augmented AI assistant that delivers accurate, context-aware guidance to general users in both preparedness and active emergency scenarios. Built on the Model Context Protocol (MCP), SafeMate dynamically routes user queries to tools for document retrieval, checklist generation, and structured summarization. It uses FAISS with cosine similarity to identify relevant content from trusted sources. |
| 2025-05-05 | [Half-Ice, Half-Fire Driven Ultranarrow Phase Crossover in 1D Decorated q-State Potts Ferrimagnets: An AI-Co-Led Discovery](http://arxiv.org/abs/2505.02303v1) | Weiguo Yin | OpenAI's reasoning model o3-mini-high was used to carry out an exact analytic study of one-dimensional ferrimagnetic decorated $q$-state Potts models. We demonstrate that the finite-temperature ultranarrow phase crossover (UNPC), driven by a hidden "half-ice, half-fire" state in the $q=2$ Potts (Ising) model, persists for $q>2$. We identify novel features for $q>2$, including the dome structure in the field-temperature phase diagram and for large $q$ a secondary high-temperature UNPC to the fully disordered paramagnetic state. The ice-fire mechanism of spin flipping can be applied to higher-dimensional Potts models. These results establish a versatile framework for engineering controlled fast state-flipping switches in low-dimensional systems. Our nine-level AI-contribution rating assigns AI the meritorious status of AI-co-led discovery in this work. |
| 2025-05-05 | [Adaptive Scoring and Thresholding with Human Feedback for Robust Out-of-Distribution Detection](http://arxiv.org/abs/2505.02299v1) | Daisuke Yamada, Harit Vishwakarma et al. | Machine Learning (ML) models are trained on in-distribution (ID) data but often encounter out-of-distribution (OOD) inputs during deployment -- posing serious risks in safety-critical domains. Recent works have focused on designing scoring functions to quantify OOD uncertainty, with score thresholds typically set based solely on ID data to achieve a target true positive rate (TPR), since OOD data is limited before deployment. However, these TPR-based thresholds leave false positive rates (FPR) uncontrolled, often resulting in high FPRs where OOD points are misclassified as ID. Moreover, fixed scoring functions and thresholds lack the adaptivity needed to handle newly observed, evolving OOD inputs, leading to sub-optimal performance. To address these challenges, we propose a human-in-the-loop framework that \emph{safely updates both scoring functions and thresholds on the fly} based on real-world OOD inputs. Our method maximizes TPR while strictly controlling FPR at all times, even as the system adapts over time. We provide theoretical guarantees for FPR control under stationary conditions and present extensive empirical evaluations on OpenOOD benchmarks to demonstrate that our approach outperforms existing methods by achieving higher TPRs while maintaining FPR control. |
| 2025-05-04 | [RNBF: Real-Time RGB-D Based Neural Barrier Functions for Safe Robotic Navigation](http://arxiv.org/abs/2505.02294v1) | Satyajeet Das, Yifan Xue et al. | Autonomous safe navigation in unstructured and novel environments poses significant challenges, especially when environment information can only be provided through low-cost vision sensors. Although safe reactive approaches have been proposed to ensure robot safety in complex environments, many base their theory off the assumption that the robot has prior knowledge on obstacle locations and geometries. In this paper, we present a real-time, vision-based framework that constructs continuous, first-order differentiable Signed Distance Fields (SDFs) of unknown environments without any pre-training. Our proposed method ensures full compatibility with established SDF-based reactive controllers. To achieve robust performance under practical sensing conditions, our approach explicitly accounts for noise in affordable RGB-D cameras, refining the neural SDF representation online for smoother geometry and stable gradient estimates. We validate the proposed method in simulation and real-world experiments using a Fetch robot. |
| 2025-05-04 | [Resolving Conflicting Constraints in Multi-Agent Reinforcement Learning with Layered Safety](http://arxiv.org/abs/2505.02293v1) | Jason J. Choi, Jasmine Jerry Aloor et al. | Preventing collisions in multi-robot navigation is crucial for deployment. This requirement hinders the use of learning-based approaches, such as multi-agent reinforcement learning (MARL), on their own due to their lack of safety guarantees. Traditional control methods, such as reachability and control barrier functions, can provide rigorous safety guarantees when interactions are limited only to a small number of robots. However, conflicts between the constraints faced by different agents pose a challenge to safe multi-agent coordination.   To overcome this challenge, we propose a method that integrates multiple layers of safety by combining MARL with safety filters. First, MARL is used to learn strategies that minimize multiple agent interactions, where multiple indicates more than two. Particularly, we focus on interactions likely to result in conflicting constraints within the engagement distance. Next, for agents that enter the engagement distance, we prioritize pairs requiring the most urgent corrective actions. Finally, a dedicated safety filter provides tactical corrective actions to resolve these conflicts. Crucially, the design decisions for all layers of this framework are grounded in reachability analysis and a control barrier-value function-based filtering mechanism.   We validate our Layered Safe MARL framework in 1) hardware experiments using Crazyflie drones and 2) high-density advanced aerial mobility (AAM) operation scenarios, where agents navigate to designated waypoints while avoiding collisions. The results show that our method significantly reduces conflict while maintaining safety without sacrificing much efficiency (i.e., shorter travel time and distance) compared to baselines that do not incorporate layered safety. The project website is available at \href{https://dinamo-mit.github.io/Layered-Safe-MARL/}{[this https URL]} |
| 2025-05-04 | [On the Need for a Statistical Foundation in Scenario-Based Testing of Autonomous Vehicles](http://arxiv.org/abs/2505.02274v1) | Xingyu Zhao, Robab Aghazadeh-Chakherlou et al. | Scenario-based testing has emerged as a common method for autonomous vehicles (AVs) safety, offering a more efficient alternative to mile-based testing by focusing on high-risk scenarios. However, fundamental questions persist regarding its stopping rules, residual risk estimation, debug effectiveness, and the impact of simulation fidelity on safety claims. This paper argues that a rigorous statistical foundation is essential to address these challenges and enable rigorous safety assurance. By drawing parallels between AV testing and traditional software testing methodologies, we identify shared research gaps and reusable solutions. We propose proof-of-concept models to quantify the probability of failure per scenario (pfs) and evaluate testing effectiveness under varying conditions. Our analysis reveals that neither scenario-based nor mile-based testing universally outperforms the other. Furthermore, we introduce Risk Estimation Fidelity (REF), a novel metric to certify the alignment of synthetic and real-world testing outcomes, ensuring simulation-based safety claims are statistically defensible. |
| 2025-05-04 | [Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents](http://arxiv.org/abs/2505.02077v1) | Christian Schroeder de Witt | Decentralized AI agents will soon interact across internet platforms, creating security challenges beyond traditional cybersecurity and AI safety frameworks. Free-form protocols are essential for AI's task generalization but enable new threats like secret collusion and coordinated swarm attacks. Network effects can rapidly spread privacy breaches, disinformation, jailbreaks, and data poisoning, while multi-agent dispersion and stealth optimization help adversaries evade oversightcreating novel persistent threats at a systemic level. Despite their critical importance, these security challenges remain understudied, with research fragmented across disparate fields including AI security, multi-agent learning, complex systems, cybersecurity, game theory, distributed systems, and technical AI governance. We introduce \textbf{multi-agent security}, a new field dedicated to securing networks of decentralized AI agents against threats that emerge or amplify through their interactionswhether direct or indirect via shared environmentswith each other, humans, and institutions, and characterize fundamental security-performance trade-offs. Our preliminary work (1) taxonomizes the threat landscape arising from interacting AI agents, (2) surveys security-performance tradeoffs in decentralized AI systems, and (3) proposes a unified research agenda addressing open challenges in designing secure agent systems and interaction environments. By identifying these gaps, we aim to guide research in this critical area to unlock the socioeconomic potential of large-scale agent deployment on the internet, foster public trust, and mitigate national security risks in critical infrastructure and defense contexts. |
| 2025-05-04 | [Enhancing Safety Standards in Automated Systems Using Dynamic Bayesian Networks](http://arxiv.org/abs/2505.02050v1) | Kranthi Kumar Talluri, Anders L. Madsen et al. | Cut-in maneuvers in high-speed traffic pose critical challenges that can lead to abrupt braking and collisions, necessitating safe and efficient lane change strategies. We propose a Dynamic Bayesian Network (DBN) framework to integrate lateral evidence with safety assessment models, thereby predicting lane changes and ensuring safe cut-in maneuvers effectively. Our proposed framework comprises three key probabilistic hypotheses (lateral evidence, lateral safety, and longitudinal safety) that facilitate the decision-making process through dynamic data processing and assessments of vehicle positions, lateral velocities, relative distance, and Time-to-Collision (TTC) computations. The DBN model's performance compared with other conventional approaches demonstrates superior performance in crash reduction, especially in critical high-speed scenarios, while maintaining a competitive performance in low-speed scenarios. This paves the way for robust, scalable, and efficient safety validation in automated driving systems. |
| 2025-05-04 | [R-Bench: Graduate-level Multi-disciplinary Benchmarks for LLM & MLLM Complex Reasoning Evaluation](http://arxiv.org/abs/2505.02018v1) | Meng-Hao Guo, Jiajun Xu et al. | Reasoning stands as a cornerstone of intelligence, enabling the synthesis of existing knowledge to solve complex problems. Despite remarkable progress, existing reasoning benchmarks often fail to rigorously evaluate the nuanced reasoning capabilities required for complex, real-world problemsolving, particularly in multi-disciplinary and multimodal contexts. In this paper, we introduce a graduate-level, multi-disciplinary, EnglishChinese benchmark, dubbed as Reasoning Bench (R-Bench), for assessing the reasoning capability of both language and multimodal models. RBench spans 1,094 questions across 108 subjects for language model evaluation and 665 questions across 83 subjects for multimodal model testing in both English and Chinese. These questions are meticulously curated to ensure rigorous difficulty calibration, subject balance, and crosslinguistic alignment, enabling the assessment to be an Olympiad-level multi-disciplinary benchmark. We evaluate widely used models, including OpenAI o1, GPT-4o, DeepSeek-R1, etc. Experimental results indicate that advanced models perform poorly on complex reasoning, especially multimodal reasoning. Even the top-performing model OpenAI o1 achieves only 53.2% accuracy on our multimodal evaluation. Data and code are made publicly available at here. |
| 2025-05-04 | [Visual Dominance and Emerging Multimodal Approaches in Distracted Driving Detection: A Review of Machine Learning Techniques](http://arxiv.org/abs/2505.01973v1) | Anthony Dontoh, Stephanie Ivey et al. | Distracted driving continues to be a significant cause of road traffic injuries and fatalities worldwide, even with advancements in driver monitoring technologies. Recent developments in machine learning (ML) and deep learning (DL) have primarily focused on visual data to detect distraction, often neglecting the complex, multimodal nature of driver behavior. This systematic review assesses 74 peer-reviewed studies from 2019 to 2024 that utilize ML/DL techniques for distracted driving detection across visual, sensor-based, multimodal, and emerging modalities. The review highlights a significant prevalence of visual-only models, particularly convolutional neural networks (CNNs) and temporal architectures, which achieve high accuracy but show limited generalizability in real-world scenarios. Sensor-based and physiological models provide complementary strengths by capturing internal states and vehicle dynamics, while emerging techniques, such as auditory sensing and radio frequency (RF) methods, offer privacy-aware alternatives. Multimodal architecture consistently surpasses unimodal baselines, demonstrating enhanced robustness, context awareness, and scalability by integrating diverse data streams. These findings emphasize the need to move beyond visual-only approaches and adopt multimodal systems that combine visual, physiological, and vehicular cues while keeping in checking the need to balance computational requirements. Future research should focus on developing lightweight, deployable multimodal frameworks, incorporating personalized baselines, and establishing cross-modality benchmarks to ensure real-world reliability in advanced driver assistance systems (ADAS) and road safety interventions. |

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



