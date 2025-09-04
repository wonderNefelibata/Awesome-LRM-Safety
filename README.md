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
| 2025-09-03 | [SafeProtein: Red-Teaming Framework and Benchmark for Protein Foundation Models](http://arxiv.org/abs/2509.03487v1) | Jigang Fan, Zhenghong Zhou et al. | Proteins play crucial roles in almost all biological processes. The advancement of deep learning has greatly accelerated the development of protein foundation models, leading to significant successes in protein understanding and design. However, the lack of systematic red-teaming for these models has raised serious concerns about their potential misuse, such as generating proteins with biological safety risks. This paper introduces SafeProtein, the first red-teaming framework designed for protein foundation models to the best of our knowledge. SafeProtein combines multimodal prompt engineering and heuristic beam search to systematically design red-teaming methods and conduct tests on protein foundation models. We also curated SafeProtein-Bench, which includes a manually constructed red-teaming benchmark dataset and a comprehensive evaluation protocol. SafeProtein achieved continuous jailbreaks on state-of-the-art protein foundation models (up to 70% attack success rate for ESM3), revealing potential biological safety risks in current protein foundation models and providing insights for the development of robust security protection technologies for frontier models. The codes will be made publicly available at https://github.com/jigang-fan/SafeProtein. |
| 2025-09-03 | [Image-Guided Surgery: Technology, Quality, Innovation, and Opportunities for Medical Physics](http://arxiv.org/abs/2509.03420v1) | Jeffrey H. Siewerdsen | The science and clinical practice of medical physics has been integral to the advancement of radiology and radiation therapy for over a century. In parallel, advances in surgery - including intraoperative imaging, registration, and other technologies within the expertise of medical physicists - have advanced primarily in connection to other disciplines, such as biomedical engineering and computer science, and via somewhat distinct translational paths. This review article briefly traces the parallel and convergent evolution of such scientific, engineering, and clinical domains with an eye to a potentially broader, more impactful role of medical physics in research and clinical practice of surgery. A review of image-guided surgery technologies is offered, including intraoperative imaging, tracking / navigation, image registration, visualization, and surgical robotics across a spectrum of surgical applications. Trends and drivers for research and innovation are traced, including federal funding and academic-industry partnership, and some of the major challenges to achieving major clinical impact are described. Opportunities for medical physicists to expand expertise and contribute to the advancement of surgery in the decade ahead are outlined, including research and innovation, data science approaches, improving efficiency through operations research and optimization, improving patient safety, and bringing rigorous quality assurance to technologies and processes in the circle of care for surgery. Challenges abound but appear tractable, including domain knowledge, professional qualifications, and the need for investment and clinical partnership. |
| 2025-09-03 | [Hierarchical Low-Altitude Wireless Network Empowered Air Traffic Management](http://arxiv.org/abs/2509.03386v1) | Ziye Jia, Jia He et al. | As the increasing development of low-altitude aircrafts, the rational design of low-altitude networks directly impacts the aerial safety and resource utilization. To address the challenges of environmental complexity and aircraft diversity in the traffic management, we propose a hierarchical low-altitude wireless network (HLWN) framework. Empowered by the threedimensional spatial discretization and integrated wireless monitoring mechanisms in HLWN, we design low-altitude air corridors to guarantee safe operation and optimization. Besides, we develop the multi-dimensional flight risk assessment through conflict detection and probabilistic collision analysis, facilitating dynamic collision avoidance for heterogeneous aircrafts. Finally, the open issues and future directions are investigated to provide insights into HLAN development. |
| 2025-09-03 | [ANNIE: Be Careful of Your Robots](http://arxiv.org/abs/2509.03383v1) | Yiyang Huang, Zixuan Wang et al. | The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at https://github.com/RLCLab/Annie. |
| 2025-09-03 | [AI Safety Assurance in Electric Vehicles: A Case Study on AI-Driven SOC Estimation](http://arxiv.org/abs/2509.03270v1) | Martin Skoglund, Fredrik Warg et al. | Integrating Artificial Intelligence (AI) technology in electric vehicles (EV) introduces unique challenges for safety assurance, particularly within the framework of ISO 26262, which governs functional safety in the automotive domain. Traditional assessment methodologies are not geared toward evaluating AI-based functions and require evolving standards and practices. This paper explores how an independent assessment of an AI component in an EV can be achieved when combining ISO 26262 with the recently released ISO/PAS 8800, whose scope is AI safety for road vehicles. The AI-driven State of Charge (SOC) battery estimation exemplifies the process. Key features relevant to the independent assessment of this extended evaluation approach are identified. As part of the evaluation, robustness testing of the AI component is conducted using fault injection experiments, wherein perturbed sensor inputs are systematically introduced to assess the component's resilience to input variance. |
| 2025-09-03 | [Parallel-Constraint Model Predictive Control: Exploiting Parallel Computation for Improving Safety](http://arxiv.org/abs/2509.03261v1) | Elias Fontanari, Gianni Lunardi et al. | Ensuring constraint satisfaction is a key requirement for safety-critical systems, which include most robotic platforms. For example, constraints can be used for modeling joint position/velocity/torque limits and collision avoidance. Constrained systems are often controlled using Model Predictive Control, because of its ability to naturally handle constraints, relying on numerical optimization. However, ensuring constraint satisfaction is challenging for nonlinear systems/constraints. A well-known tool to make controllers safe is the so-called control-invariant set (a.k.a. safe set). In our previous work, we have shown that safety can be improved by letting the safe-set constraint recede along the MPC horizon. In this paper, we push that idea further by exploiting parallel computation to improve safety. We solve several MPC problems at the same time, where each problem instantiates the safe-set constraint at a different time step along the horizon. Finally, the controller can select the best solution according to some user-defined criteria. We validated this idea through extensive simulations with a 3-joint robotic arm, showing that significant improvements can be achieved in terms of safety and performance, even using as little as 4 computational cores. |
| 2025-09-03 | [Rashomon in the Streets: Explanation Ambiguity in Scene Understanding](http://arxiv.org/abs/2509.03169v1) | Helge Spieker, Jørn Eirik Betten et al. | Explainable AI (XAI) is essential for validating and trusting models in safety-critical applications like autonomous driving. However, the reliability of XAI is challenged by the Rashomon effect, where multiple, equally accurate models can offer divergent explanations for the same prediction. This paper provides the first empirical quantification of this effect for the task of action prediction in real-world driving scenes. Using Qualitative Explainable Graphs (QXGs) as a symbolic scene representation, we train Rashomon sets of two distinct model classes: interpretable, pair-based gradient boosting models and complex, graph-based Graph Neural Networks (GNNs). Using feature attribution methods, we measure the agreement of explanations both within and between these classes. Our results reveal significant explanation disagreement. Our findings suggest that explanation ambiguity is an inherent property of the problem, not just a modeling artifact. |
| 2025-09-03 | [A Neural Network Approach to Multi-radionuclide TDCR Beta Spectroscopy](http://arxiv.org/abs/2509.03137v1) | Li Yi, Qian Yang | Liquid scintillation triple-to-doubly coincident ratio (TDCR) spectroscopy is widely adopted as a standard method for radionuclide quantification because of its inherent advantages such as high precision, self-calibrating capability, and independence from radioactive reference sources. However, multiradionuclide analysis via TDCR faces the challenges of limited automation and reliance on mixture-specific standards, which may not be easily available. Here, we present an Artificial Intelligence (AI) framework that combines numerical spectral simulation and deep learning for standard-free automated analysis. $\beta$ spectra for model training were generated using Geant4 simulations coupled with statistically modeled detector response sampling. A tailored neural network architecture, trained on this dataset covering various nuclei mix ratio and quenching scenarios, enables autonomous resolution of individual radionuclide activities and detecting efficiency through end-to-end learning paradigms. The model delivers consistent high accuracy across tasks: activity proportions (mean absolute error = 0.009), detection efficiencies (mean absolute error = 0.002), and spectral reconstruction (Structural Similarity Index = 0.9998), validating its physical plausibility for quenched $\beta$ spectroscopy. This AI-driven methodology exhibits significant potential for automated safety-compliant multiradionuclide analysis with robust generalization, real-time processing capabilities, and engineering feasibility, particularly in scenarios where reference materials are unavailable or rapid field analysis is required. |
| 2025-09-03 | [A Hierarchical Deep Reinforcement Learning Framework for Traffic Signal Control with Predictable Cycle Planning](http://arxiv.org/abs/2509.03118v1) | Hankang Gu, Yuli Zhang et al. | Deep reinforcement learning (DRL) has become a popular approach in traffic signal control (TSC) due to its ability to learn adaptive policies from complex traffic environments. Within DRL-based TSC methods, two primary control paradigms are ``choose phase" and ``switch" strategies. Although the agent in the choose phase paradigm selects the next active phase adaptively, this paradigm may result in unexpected phase sequences for drivers, disrupting their anticipation and potentially compromising safety at intersections. Meanwhile, the switch paradigm allows the agent to decide whether to switch to the next predefined phase or extend the current phase. While this structure maintains a more predictable order, it can lead to unfair and inefficient phase allocations, as certain movements may be extended disproportionately while others are neglected. In this paper, we propose a DRL model, named Deep Hierarchical Cycle Planner (DHCP), to allocate the traffic signal cycle duration hierarchically. A high-level agent first determines the split of the total cycle time between the North-South (NS) and East-West (EW) directions based on the overall traffic state. Then, a low-level agent further divides the allocated duration within each major direction between straight and left-turn movements, enabling more flexible durations for the two movements. We test our model on both real and synthetic road networks, along with multiple sets of real and synthetic traffic flows. Empirical results show our model achieves the best performance over all datasets against baselines. |
| 2025-09-03 | [Uncertainty-aware Test-Time Training (UT$^3$) for Efficient On-the-fly Domain Adaptive Dense Regression](http://arxiv.org/abs/2509.03012v1) | Uddeshya Upadhyay | Deep neural networks (DNNs) are increasingly being used in autonomous systems. However, DNNs do not generalize well to domain shift. Adapting to a continuously evolving environment is a safety-critical challenge inevitably faced by all autonomous systems deployed to the real world. Recent work on test-time training proposes methods that adapt to a new test distribution on the fly by optimizing the DNN model for each test input using self-supervision. However, these techniques result in a sharp increase in inference time as multiple forward and backward passes are required for a single test sample (for test-time training) before finally making the prediction based on the fine-tuned features. This is undesirable for real-world robotics applications where these models may be deployed to resource constraint hardware with strong latency requirements. In this work, we propose a new framework (called UT$^3$) that leverages test-time training for improved performance in the presence of continuous domain shift while also decreasing the inference time, making it suitable for real-world applications. Our method proposes an uncertainty-aware self-supervision task for efficient test-time training that leverages the quantified uncertainty to selectively apply the training leading to sharp improvements in the inference time while performing comparably to standard test-time training protocol. Our proposed protocol offers a continuous setting to identify the selected keyframes, allowing the end-user to control how often to apply test-time training. We demonstrate the efficacy of our method on a dense regression task - monocular depth estimation. |
| 2025-09-03 | [StableSleep: Source-Free Test-Time Adaptation for Sleep Staging with Lightweight Safety Rails](http://arxiv.org/abs/2509.02982v1) | Hritik Arasu, Faisal R Jahangiri | Sleep staging models often degrade when deployed on patients with unseen physiology or recording conditions. We propose a streaming, source-free test-time adaptation (TTA) recipe that combines entropy minimization (Tent) with Batch-Norm statistic refresh and two safety rails: an entropy gate to pause adaptation on uncertain windows and an EMA-based reset to reel back drift. On Sleep-EDF Expanded, using single-lead EEG (Fpz-Cz, 100 Hz, 30s epochs; R&K to AASM mapping), we show consistent gains over a frozen baseline at seconds-level latency and minimal memory, reporting per-stage metrics and Cohen's k. The method is model-agnostic, requires no source data or patient calibration, and is practical for on-device or bedside use. |
| 2025-09-03 | [KEPT: Knowledge-Enhanced Prediction of Trajectories from Consecutive Driving Frames with Vision-Language Models](http://arxiv.org/abs/2509.02966v1) | Yujin Wang, Tianyi Wang et al. | Accurate short-horizon trajectory prediction is pivotal for safe and reliable autonomous driving, yet existing vision-language models (VLMs) often fail to effectively ground their reasoning in scene dynamics and domain knowledge. To address this challenge, this paper introduces KEPT, a knowledge-enhanced VLM framework that predicts ego trajectories directly from consecutive front-view driving frames. KEPT couples a temporal frequency-spatial fusion (TFSF) video encoder, trained via self-supervised learning with hard-negative mining, with a scalable k-means + HNSW retrieval stack that supplies scene-aligned exemplars. Retrieved priors are embedded into chain-of-thought (CoT) prompts with explicit planning constraints, while a triple-stage fine-tuning schedule incrementally aligns the language head to metric spatial cues, physically feasible motion, and temporally conditioned front-view planning. Evaluated on nuScenes dataset, KEPT achieves state-of-the-art performance across open-loop protocols: under NoAvg, it achieves 0.70m average L2 with a 0.21\% collision rate; under TemAvg with lightweight ego status, it attains 0.31m average L2 and a 0.07\% collision rate. Ablation studies show that all three fine-tuning stages contribute complementary benefits, and that using Top-2 retrieved exemplars yields the best accuracy-safety trade-off. The k-means-clustered HNSW index delivers sub-millisecond retrieval latency, supporting practical deployment. These results indicate that retrieval-augmented, CoT-guided VLMs offer a promising, data-efficient pathway toward interpretable and trustworthy autonomous driving. |
| 2025-09-03 | [A Data-Driven RetinaNet Model for Small Object Detection in Aerial Images](http://arxiv.org/abs/2509.02928v1) | Zhicheng Tang, Jinwen Tang et al. | In the realm of aerial imaging, the ability to detect small objects is pivotal for a myriad of applications, encompassing environmental surveillance, urban design, and crisis management. Leveraging RetinaNet, this work unveils DDR-Net: a data-driven, deep-learning model devised to enhance the detection of diminutive objects. DDR-Net introduces novel, data-driven techniques to autonomously ascertain optimal feature maps and anchor estimations, cultivating a tailored and proficient training process while maintaining precision. Additionally, this paper presents an innovative sampling technique to bolster model efficacy under limited data training constraints. The model's enhanced detection capabilities support critical applications including wildlife and habitat monitoring, traffic flow optimization, and public safety improvements through accurate identification of small objects like vehicles and pedestrians. DDR-Net significantly reduces the cost and time required for data collection and training, offering efficient performance even with limited data. Empirical assessments over assorted aerial avian imagery datasets demonstrate that DDR-Net markedly surpasses RetinaNet and alternative contemporary models. These innovations advance current aerial image analysis technologies and promise wide-ranging impacts across multiple sectors including agriculture, security, and archaeology. |
| 2025-09-02 | [Safe Sharing of Fast Kernel-Bypass I/O Among Nontrusting Applications](http://arxiv.org/abs/2509.02899v1) | Alan Beadle, Michael L. Scott et al. | Protected user-level libraries have been proposed as a way to allow mutually distrusting applications to safely share kernel-bypass services. In this paper, we identify and solve several previously unaddressed obstacles to realizing this design and identify several optimization opportunities. First, to preserve the kernel's ability to reclaim failed processes, protected library functions must complete in modest, bounded time. We show how to move unbounded waits outside the library itself, enabling synchronous interaction among processes without the need for polling. Second, we show how the bounded time requirement can be leveraged to achieve lower and more stable latency for inter-process interactions. Third, we observe that prior work on protected libraries is vulnerable to a buffer unmapping attack; we prevent this attack by preventing applications from removing pages that they share with the protected library. Fourth, we show how a trusted daemon can respond to asynchronous events and dynamically divide work with application threads in a protected library.   By extending and improving the protected library model, our work provides a new way to structure OS services, combining the advantages of kernel bypass and microkernels. We present a set of safety and performance guidelines for developers of protected libraries, and a set of recommendations for developers of future protected library operating systems. We demonstrate the convenience and performance of our approach with a prototype version of the DDS communication service. To the best of our knowledge, this prototype represents the first successful sharing of a kernel-bypass NIC among mutually untrusting applications. Relative to the commercial FastDDS implementation, we achieve approximately 50\% lower latency and up to 7x throughput, with lower CPU utilization. |
| 2025-09-02 | [Learning from geometry-aware near-misses to real-time COR: A spatiotemporal grouped random GEV framework](http://arxiv.org/abs/2509.02871v1) | Mohammad Anis, Yang Zhou et al. | Real-time prediction of corridor-level crash occurrence risk (COR) remains challenging, as existing near-miss based extreme value models oversimplify collision geometry, exclude vehicle-infrastructure (V-I) interactions, and inadequately capture spatial heterogeneity in vehicle dynamics. This study introduces a geometry-aware two-dimensional time-to-collision (2D-TTC) indicator within a Hierarchical Bayesian spatiotemporal grouped random parameter (HBSGRP) framework using a non-stationary univariate generalized extreme value (UGEV) model to estimate short-term COR in urban corridors. High-resolution trajectories from the Argoverse-2 dataset, covering 28 locations along Miami's Biscayne Boulevard, were analyzed to extract extreme V-V and V-I near misses. The model incorporates dynamic variables and roadway features as covariates, with partial pooling across locations to address unobserved heterogeneity. Results show that the HBSGRP-UGEV framework outperforms fixed-parameter alternatives, reducing DIC by up to 7.5% for V-V and 3.1% for V-I near-misses. Predictive validation using ROC-AUC confirms strong performance: 0.89 for V-V segments, 0.82 for V-V intersections, 0.79 for V-I segments, and 0.75 for V-I intersections. Model interpretation reveals that relative speed and distance dominate V-V risks at intersections and segments, with deceleration critical in segments, while V-I risks are driven by speed, boundary proximity, and steering/heading adjustments. These findings highlight the value of a statistically rigorous, geometry-sensitive, and spatially adaptive modeling approach for proactive corridor-level safety management, supporting real-time interventions and long-term design strategies aligned with Vision Zero. |
| 2025-09-02 | [GPS Spoofing Attacks on Automated Frequency Coordination System in Wi-Fi 6E and Beyond](http://arxiv.org/abs/2509.02824v1) | Yilu Dong, Tianchang Yang et al. | The 6 GHz spectrum, recently opened for unlicensed use under Wi-Fi 6E and Wi-Fi 7, overlaps with frequencies used by mission-critical incumbent systems such as public safety communications and utility infrastructure. To prevent interference, the FCC mandates the use of Automated Frequency Coordination (AFC) systems, which assign safe frequency and power levels based on Wi-Fi Access Point (AP)-reported locations. In this work, we demonstrate that GPS-based location reporting, which Wi-Fi APs use, can be spoofed using inexpensive, off-the-shelf radio equipment. This enables attackers to manipulate AP behavior, gain unauthorized spectrum access, cause harmful interference, or disable APs entirely by spoofing them into foreign locations. We validate these attacks in a controlled lab setting against a commercial AP and evaluate a commercial AFC system under spoofed scenarios. Our findings highlight critical gaps in the security assumptions of AFC and motivate the need for stronger location integrity protections. |
| 2025-09-02 | [Unlearning That Lasts: Utility-Preserving, Robust, and Almost Irreversible Forgetting in LLMs](http://arxiv.org/abs/2509.02820v1) | Naman Deep Singh, Maximilian Müller et al. | Unlearning in large language models (LLMs) involves precisely removing specific information from a pre-trained model. This is crucial to ensure safety of LLMs by deleting private data or harmful knowledge acquired during pre-training. However, existing unlearning methods often fall short when subjected to thorough evaluation. To overcome this, we introduce JensUn, where we leverage the Jensen-Shannon Divergence as the training objective for both forget and retain sets for more stable and effective unlearning dynamics compared to commonly used loss functions. In extensive experiments, JensUn achieves better forget-utility trade-off than competing methods, and even demonstrates strong resilience to benign relearning. Additionally, for a precise unlearning evaluation, we introduce LKF, a curated dataset of lesser-known facts that provides a realistic unlearning scenario. Finally, to comprehensively test unlearning methods, we propose (i) employing an LLM as semantic judge instead of the standard ROUGE score, and (ii) using worst-case unlearning evaluation over various paraphrases and input formats. Our improved evaluation framework reveals that many existing methods are less effective than previously thought. |
| 2025-09-02 | [Improving the Resilience of Quadrotors in Underground Environments by Combining Learning-based and Safety Controllers](http://arxiv.org/abs/2509.02808v1) | Isaac Ronald Ward, Mark Paral et al. | Autonomously controlling quadrotors in large-scale subterranean environments is applicable to many areas such as environmental surveying, mining operations, and search and rescue. Learning-based controllers represent an appealing approach to autonomy, but are known to not generalize well to `out-of-distribution' environments not encountered during training. In this work, we train a normalizing flow-based prior over the environment, which provides a measure of how far out-of-distribution the quadrotor is at any given time. We use this measure as a runtime monitor, allowing us to switch between a learning-based controller and a safe controller when we are sufficiently out-of-distribution. Our methods are benchmarked on a point-to-point navigation task in a simulated 3D cave environment based on real-world point cloud data from the DARPA Subterranean Challenge Final Event Dataset. Our experimental results show that our combined controller simultaneously possesses the liveness of the learning-based controller (completing the task quickly) and the safety of the safety controller (avoiding collision). |
| 2025-09-02 | [A Composite-Loss Graph Neural Network for the Multivariate Post-Processing of Ensemble Weather Forecasts](http://arxiv.org/abs/2509.02784v1) | Mária Lakatos | Ensemble forecasting systems have advanced meteorology by providing probabilistic estimates of future states, supporting applications from renewable energy production to transportation safety. Nonetheless, systematic biases often persist, making statistical post-processing essential. Traditional parametric post-processing techniques and machine learning-based methods can produce calibrated predictive distributions at specific locations and lead times, yet often struggle to capture dependencies across forecast dimensions. To address this, multivariate post-processing methods-such as ensemble copula coupling and the Schaake shuffle-are widely applied in a second step to restore realistic inter-variable or spatio-temporal dependencies. The aim of this study is the multivariate post-processing of ensemble forecasts using a graph neural network (dualGNN) trained with a composite loss function that combines the energy score (ES) and the variogram score (VS). The method is evaluated on two datasets: WRF-based solar irradiance forecasts over northern Chile and ECMWF visibility forecasts for Central Europe. The dualGNN consistently outperforms all empirical copula-based post-processed forecasts and shows significant improvements compared to graph neural networks trained solely on either the continuous ranked probability score (CRPS) or the ES, according to the evaluated multivariate verification metrics. Furthermore, for the WRF forecasts, the rank-order structure of the dualGNN forecasts captures valuable dependency information, enabling a more effective restoration of spatial relationships than either the raw numerical weather prediction ensemble or historical observational rank structures. By contrast, for the visibility forecasts, the GNNs trained on CRPS, ES, or the ES-VS combination outperform the calibrated reference. |
| 2025-09-02 | [Plan Verification for LLM-Based Embodied Task Completion Agents](http://arxiv.org/abs/2509.02761v1) | Ananth Hariharan, Vardhan Dongre et al. | Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI. |

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



