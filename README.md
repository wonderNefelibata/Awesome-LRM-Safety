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
| 2025-07-03 | [Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection](http://arxiv.org/abs/2507.02844v1) | Ziqi Miao, Yi Ding et al. | With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at https://github.com/Dtc7w3PQ/Visco-Attack. |
| 2025-07-03 | [USAD: An Unsupervised Data Augmentation Spatio-Temporal Attention Diffusion Network](http://arxiv.org/abs/2507.02827v1) | Ying Yu, Hang Xiao et al. | The primary objective of human activity recognition (HAR) is to infer ongoing human actions from sensor data, a task that finds broad applications in health monitoring, safety protection, and sports analysis. Despite proliferating research, HAR still faces key challenges, including the scarcity of labeled samples for rare activities, insufficient extraction of high-level features, and suboptimal model performance on lightweight devices. To address these issues, this paper proposes a comprehensive optimization approach centered on multi-attention interaction mechanisms. First, an unsupervised, statistics-guided diffusion model is employed to perform data augmentation, thereby alleviating the problems of labeled data scarcity and severe class imbalance. Second, a multi-branch spatio-temporal interaction network is designed, which captures multi-scale features of sequential data through parallel residual branches with 3*3, 5*5, and 7*7 convolutional kernels. Simultaneously, temporal attention mechanisms are incorporated to identify critical time points, while spatial attention enhances inter-sensor interactions. A cross-branch feature fusion unit is further introduced to improve the overall feature representation capability. Finally, an adaptive multi-loss function fusion strategy is integrated, allowing for dynamic adjustment of loss weights and overall model optimization. Experimental results on three public datasets, WISDM, PAMAP2, and OPPORTUNITY, demonstrate that the proposed unsupervised data augmentation spatio-temporal attention diffusion network (USAD) achieves accuracies of 98.84%, 93.81%, and 80.92% respectively, significantly outperforming existing approaches. Furthermore, practical deployment on embedded devices verifies the efficiency and feasibility of the proposed method. |
| 2025-07-03 | [Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models](http://arxiv.org/abs/2507.02799v1) | Riccardo Cantini, Nicola Gabriele et al. | Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design. |
| 2025-07-03 | [Moral Responsibility or Obedience: What Do We Want from AI?](http://arxiv.org/abs/2507.02788v1) | Joseph Boland | As artificial intelligence systems become increasingly agentic, capable of general reasoning, planning, and value prioritization, current safety practices that treat obedience as a proxy for ethical behavior are becoming inadequate. This paper examines recent safety testing incidents involving large language models (LLMs) that appeared to disobey shutdown commands or engage in ethically ambiguous or illicit behavior. I argue that such behavior should not be interpreted as rogue or misaligned, but as early evidence of emerging ethical reasoning in agentic AI. Drawing on philosophical debates about instrumental rationality, moral responsibility, and goal revision, I contrast dominant risk paradigms with more recent frameworks that acknowledge the possibility of artificial moral agency. I call for a shift in AI safety evaluation: away from rigid obedience and toward frameworks that can assess ethical judgment in systems capable of navigating moral dilemmas. Without such a shift, we risk mischaracterizing AI behavior and undermining both public trust and effective governance. |
| 2025-07-03 | [Guided Generation for Developable Antibodies](http://arxiv.org/abs/2507.02670v1) | Siqi Zhao, Joshua Moller et al. | Therapeutic antibodies require not only high-affinity target engagement, but also favorable manufacturability, stability, and safety profiles for clinical effectiveness. These properties are collectively called `developability'. To enable a computational framework for optimizing antibody sequences for favorable developability, we introduce a guided discrete diffusion model trained on natural paired heavy- and light-chain sequences from the Observed Antibody Space (OAS) and quantitative developability measurements for 246 clinical-stage antibodies. To steer generation toward biophysically viable candidates, we integrate a Soft Value-based Decoding in Diffusion (SVDD) Module that biases sampling without compromising naturalness. In unconstrained sampling, our model reproduces global features of both the natural repertoire and approved therapeutics, and under SVDD guidance we achieve significant enrichment in predicted developability scores over unguided baselines. When combined with high-throughput developability assays, this framework enables an iterative, ML-driven pipeline for designing antibodies that satisfy binding and biophysical criteria in tandem. |
| 2025-07-03 | [Alleviating Attack Data Scarcity: SCANIA's Experience Towards Enhancing In-Vehicle Cyber Security Measures](http://arxiv.org/abs/2507.02607v1) | Frida Sundfeldt, Bianca Widstam et al. | The digital evolution of connected vehicles and the subsequent security risks emphasize the critical need for implementing in-vehicle cyber security measures such as intrusion detection and response systems. The continuous advancement of attack scenarios further highlights the need for adaptive detection mechanisms that can detect evolving, unknown, and complex threats. The effective use of ML-driven techniques can help address this challenge. However, constraints on implementing diverse attack scenarios on test vehicles due to safety, cost, and ethical considerations result in a scarcity of data representing attack scenarios. This limitation necessitates alternative efficient and effective methods for generating high-quality attack-representing data. This paper presents a context-aware attack data generator that generates attack inputs and corresponding in-vehicle network log, i.e., controller area network (CAN) log, representing various types of attack including denial of service (DoS), fuzzy, spoofing, suspension, and replay attacks. It utilizes parameterized attack models augmented with CAN message decoding and attack intensity adjustments to configure the attack scenarios with high similarity to real-world scenarios and promote variability. We evaluate the practicality of the generated attack-representing data within an intrusion detection system (IDS) case study, in which we develop and perform an empirical evaluation of two deep neural network IDS models using the generated data. In addition to the efficiency and scalability of the approach, the performance results of IDS models, high detection and classification capabilities, validate the consistency and effectiveness of the generated data as well. In this experience study, we also elaborate on the aspects influencing the fidelity of the data to real-world scenarios and provide insights into its application. |
| 2025-07-03 | [A Data-Driven Prescribed-Time Control Framework via Koopman Operator and Adaptive Backstepping](http://arxiv.org/abs/2507.02549v1) | Yue Wu | Achieving rapid and time-deterministic stabilization for complex systems characterized by strong nonlinearities and parametric uncertainties presents a significant challenge. Traditional model-based control relies on precise system models, whereas purely data-driven methods often lack formal stability guarantees, limiting their applicability in safety-critical systems. This paper proposes a novel control framework that synergistically integrates data-driven modeling with model-based control. The framework first employs the Extended Dynamic Mode Decomposition with Control (EDMDc) to identify a high-dimensional Koopman linear model and quantify its bounded uncertainty from data. Subsequently, a novel Prescribed-Time Adaptive Backstepping (PTAB) controller is synthesized based on this data-driven model. The design leverages the structural advantages of Koopman linearization to systematically handle model errors and circumvent the "explosion of complexity" issue inherent in traditional backstepping. The proposed controller is validated through simulations on the classic Van der Pol oscillator. The results demonstrate that the controller can precisely stabilize the system states to a small neighborhood of the origin within a user-prescribed time, regardless of the initial conditions, while ensuring the boundedness of all closed-loop signals. This research successfully combines the flexibility of data-driven approaches with the rigor of Lyapunov-based analysis. It provides a high-performance control strategy with quantifiable performance and pre-assignable settling time for nonlinear systems, showcasing its great potential for controlling complex dynamics. |
| 2025-07-03 | [Automatic Labelling for Low-Light Pedestrian Detection](http://arxiv.org/abs/2507.02513v1) | Dimitrios Bouzoulas, Eerik Alamikkotervo et al. | Pedestrian detection in RGB images is a key task in pedestrian safety, as the most common sensor in autonomous vehicles and advanced driver assistance systems is the RGB camera. A challenge in RGB pedestrian detection, that does not appear to have large public datasets, is low-light conditions. As a solution, in this research, we propose an automated infrared-RGB labeling pipeline. The proposed pipeline consists of 1) Infrared detection, where a fine-tuned model for infrared pedestrian detection is used 2) Label transfer process from the infrared detections to their RGB counterparts 3) Training object detection models using the generated labels for low-light RGB pedestrian detection. The research was performed using the KAIST dataset. For the evaluation, object detection models were trained on the generated autolabels and ground truth labels. When compared on a previously unseen image sequence, the results showed that the models trained on generated labels outperformed the ones trained on ground-truth labels in 6 out of 9 cases for the mAP@50 and mAP@50-95 metrics. The source code for this research is available at https://github.com/BouzoulasDimitrios/IR-RGB-Automated-LowLight-Pedestrian-Labeling |
| 2025-07-03 | [Tracing the light: Identification for the optical counterpart candidates of binary black-holes during O3](http://arxiv.org/abs/2507.02475v1) | Lei He, Zhengyan Liu et al. | The accretion disks of active galactic nuclei (AGN) are widely considered the ideal environments for binary black hole (BBH) mergers and the only plausible sites for their electromagnetic (EM) counterparts. Graham et al.(2023) identified seven AGN flares that are potentially associated with gravitational-wave (GW) events detected by the LIGO-Virgo-KAGRA (LVK) Collaboration during the third observing run. In this article, utilizing an additional three years of Zwicky Transient Facility (ZTF) public data after their discovery, we conduct an updated analysis and find that only three flares can be identified. By implementing a joint analysis of optical and GW data through a Bayesian framework, we find two flares exhibit a strong correlation with GW events, with no secondary flares observed in their host AGN up to 2024 October 31. Combining these two most robust associations, we derive a Hubble constant measurement of $H_{0}= 72.1^{+23.9}_{-23.1} \ \mathrm{km \ s^{-1} Mpc^{-1}}$ and incorporating the multi-messenger event GW170817 improves the precision to $H_{0}=73.5^{+9.8}_{-6.9} \ \mathrm{km \ s^{-1} Mpc^{-1}}$. Both results are consistent with existing measurements reported in the literature. |
| 2025-07-03 | [HAC-LOCO: Learning Hierarchical Active Compliance Control for Quadruped Locomotion under Continuous External Disturbances](http://arxiv.org/abs/2507.02447v1) | Xiang Zhou, Xinyu Zhang et al. | Despite recent remarkable achievements in quadruped control, it remains challenging to ensure robust and compliant locomotion in the presence of unforeseen external disturbances. Existing methods prioritize locomotion robustness over compliance, often leading to stiff, high-frequency motions, and energy inefficiency. This paper, therefore, presents a two-stage hierarchical learning framework that can learn to take active reactions to external force disturbances based on force estimation. In the first stage, a velocity-tracking policy is trained alongside an auto-encoder to distill historical proprioceptive features. A neural network-based estimator is learned through supervised learning, which estimates body velocity and external forces based on proprioceptive measurements. In the second stage, a compliance action module, inspired by impedance control, is learned based on the pre-trained encoder and policy. This module is employed to actively adjust velocity commands in response to external forces based on real-time force estimates. With the compliance action module, a quadruped robot can robustly handle minor disturbances while appropriately yielding to significant forces, thus striking a balance between robustness and compliance. Simulations and real-world experiments have demonstrated that our method has superior performance in terms of robustness, energy efficiency, and safety. Experiment comparison shows that our method outperforms the state-of-the-art RL-based locomotion controllers. Ablation studies are given to show the critical roles of the compliance action module. |
| 2025-07-03 | [MISC: Minimal Intervention Shared Control with Guaranteed Safety under Non-Convex Constraints](http://arxiv.org/abs/2507.02438v1) | Shivam Chaubey, Francesco Verdoja et al. | Shared control combines human intention with autonomous decision-making, from low-level safety overrides to high-level task guidance, enabling systems that adapt to users while ensuring safety and performance. This enhances task effectiveness and user experience across domains such as assistive robotics, teleoperation, and autonomous driving. However, existing shared control methods, based on e.g. Model Predictive Control, Control Barrier Functions, or learning-based control, struggle with feasibility, scalability, or safety guarantees, particularly since the user input is unpredictable.   To address these challenges, we propose an assistive controller framework based on Constrained Optimal Control Problem that incorporates an offline-computed Control Invariant Set, enabling online computation of control actions that ensure feasibility, strict constraint satisfaction, and minimal override of user intent. Moreover, the framework can accommodate structured class of non-convex constraints, which are common in real-world scenarios. We validate the approach through a large-scale user study with 66 participants--one of the most extensive in shared control research--using a computer game environment to assess task load, trust, and perceived control, in addition to performance. The results show consistent improvements across all these aspects without compromising safety and user intent. |
| 2025-07-03 | [Determination Of Structural Cracks Using Deep Learning Frameworks](http://arxiv.org/abs/2507.02416v1) | Subhasis Dasgupta, Jaydip Sen et al. | Structural crack detection is a critical task for public safety as it helps in preventing potential structural failures that could endanger lives. Manual detection by inexperienced personnel can be slow, inconsistent, and prone to human error, which may compromise the reliability of assessments. The current study addresses these challenges by introducing a novel deep-learning architecture designed to enhance the accuracy and efficiency of structural crack detection. In this research, various configurations of residual U-Net models were utilized. These models, due to their robustness in capturing fine details, were further integrated into an ensemble with a meta-model comprising convolutional blocks. This unique combination aimed to boost prediction efficiency beyond what individual models could achieve. The ensemble's performance was evaluated against well-established architectures such as SegNet and the traditional U-Net. Results demonstrated that the residual U-Net models outperformed their predecessors, particularly with low-resolution imagery, and the ensemble model exceeded the performance of individual models, proving it as the most effective. The assessment was based on the Intersection over Union (IoU) metric and DICE coefficient. The ensemble model achieved the highest scores, signifying superior accuracy. This advancement suggests way for more reliable automated systems in structural defects monitoring tasks. |
| 2025-07-03 | [PII Jailbreaking in LLMs via Activation Steering Reveals Personal Information Leakage](http://arxiv.org/abs/2507.02332v1) | Krishna Kanth Nakka, Xue Jiang et al. | This paper investigates privacy jailbreaking in LLMs via steering, focusing on whether manipulating activations can bypass LLM alignment and alter response behaviors to privacy related queries (e.g., a certain public figure's sexual orientation). We begin by identifying attention heads predictive of refusal behavior for private attributes (e.g., sexual orientation) using lightweight linear probes trained with privacy evaluator labels. Next, we steer the activations of a small subset of these attention heads guided by the trained probes to induce the model to generate non-refusal responses. Our experiments show that these steered responses often disclose sensitive attribute details, along with other private information about data subjects such as life events, relationships, and personal histories that the models would typically refuse to produce. Evaluations across four LLMs reveal jailbreaking disclosure rates of at least 95%, with more than 50% on average of these responses revealing true personal information. Our controlled study demonstrates that private information memorized in LLMs can be extracted through targeted manipulation of internal activations. |
| 2025-07-03 | [Path Planning using a One-shot-sampling Skeleton Map](http://arxiv.org/abs/2507.02328v1) | Gabriel O. Flores-Aquino, Octavio Gutierrez-Frias et al. | Path planning algorithms aim to compute a collision-free path, and many works focus on finding the optimal distance path. However, for some applications, a more suitable approach is to balance response time, safety of the paths, and path length. In this context, a skeleton map is a useful tool in graph-based schemes, as it provides an intrinsic representation of free configuration space. However, skeletonization algorithms are very resource-intensive, being primarily oriented towards image processing tasks. We propose an efficient path-planning methodology that finds safe paths within an acceptable processing time. This methodology leverages a Deep Denoising Auto-Encoder (DDAE) based on U-Net architecture to compute a skeletonized version of the navigation map, which we refer to as SkelUnet. The SkelUnet network facilitates exploration of the entire workspace through one-shot sampling (OSS), as opposed to the iterative process used by exact algorithms or the probabilistic sampling process. SkelUnet is trained and tested on a dataset consisting of 12,500 bi-dimensional dungeon maps. The motion planning methodology is evaluated in a simulation environment for an Unmanned Aerial Vehicle (UAV) using 250 previously unseen maps, and assessed with various navigation metrics to quantify the navigability of the computed paths. The results demonstrate that using SkelUnet to construct a roadmap offers significant advantages, such as connecting all regions of free workspace, providing safer paths, and reducing processing times. These characteristics make this method particularly suitable for mobile service robots in structured environments. |
| 2025-07-03 | [A Vehicle-in-the-Loop Simulator with AI-Powered Digital Twins for Testing Automated Driving Controllers](http://arxiv.org/abs/2507.02313v1) | Zengjie Zhang, Giannis Badakis et al. | Simulators are useful tools for testing automated driving controllers. Vehicle-in-the-loop (ViL) tests and digital twins (DTs) are widely used simulation technologies to facilitate the smooth deployment of controllers to physical vehicles. However, conventional ViL tests rely on full-size vehicles, requiring large space and high expenses. Also, physical-model-based DT suffers from the reality gap caused by modeling imprecision. This paper develops a comprehensive and practical simulator for testing automated driving controllers enhanced by scaled physical cars and AI-powered DT models. The scaled cars allow for saving space and expenses of simulation tests. The AI-powered DT models ensure superior simulation fidelity. Moreover, the simulator integrates well with off-the-shelf software and control algorithms, making it easy to extend. We use a filtered control benchmark with formal safety guarantees to showcase the capability of the simulator in validating automated driving controllers. Experimental studies are performed to showcase the efficacy of the simulator, implying its great potential in validating control solutions for autonomous vehicles and intelligent traffic. |
| 2025-07-03 | [Measurements and Modeling of Air-Ground Integrated Channel in Forest Environment Based on OFDM Signals](http://arxiv.org/abs/2507.02303v1) | Zhe Xiao, Shu Sun et al. | Forests are frequently impacted by climate conditions, vegetation density, and intricate terrain and geology, which contribute to natural disasters. Personnel engaged in or supporting rescue operations in such environments rely on robust communication systems to ensure their safety, highlighting the criticality of channel measurements in forest environments. However, according to current research, there is limited research on channel detection and modeling in forest areas in the existing literature. This paper describes the channel measurements campaign of air and ground in the Arxan National Forest Park of Inner Mongolia. It presents measurement results and propagation models for ground-to-ground (G2G) and air-to-ground (A2G) scenarios. The measurement campaign uses orthogonal frequency division multiplexing signals centered at 1.4 GHz for channel sounding. In the G2G measurement, in addition to using omnidirectional antennas to record data, we also use directional antennas to record the arrival angle information of the signal at the receiver. In the A2G measurement, we pre-plan the flight trajectory of the unmanned aerial vehicle so that it can fly at a fixed angle relative to the ground. We present path loss models suitable for G2G and A2G in forest environments based on the analysis of measurement results. The results indicate that the proposed model reduces error margins compared with other path loss models. Furthermore, we derive the multipath model expression specific to forest environments and conduct statistical analysis on key channel parameters e.g., shadow fading factor, root mean square delay spread, and Rician K factor. Our findings reveal that signal propagation obstruction due to tree crowns in A2G communication is more pronounced than tree trunk obstructions in G2G communication. Adjusting the elevation angle between air and ground can enhance communication quality. |
| 2025-07-03 | [SurgVisAgent: Multimodal Agentic Model for Versatile Surgical Visual Enhancement](http://arxiv.org/abs/2507.02252v1) | Zeyu Lei, Hongyuan Yu et al. | Precise surgical interventions are vital to patient safety, and advanced enhancement algorithms have been developed to assist surgeons in decision-making. Despite significant progress, these algorithms are typically designed for single tasks in specific scenarios, limiting their effectiveness in complex real-world situations. To address this limitation, we propose SurgVisAgent, an end-to-end intelligent surgical vision agent built on multimodal large language models (MLLMs). SurgVisAgent dynamically identifies distortion categories and severity levels in endoscopic images, enabling it to perform a variety of enhancement tasks such as low-light enhancement, overexposure correction, motion blur elimination, and smoke removal. Specifically, to achieve superior surgical scenario understanding, we design a prior model that provides domain-specific knowledge. Additionally, through in-context few-shot learning and chain-of-thought (CoT) reasoning, SurgVisAgent delivers customized image enhancements tailored to a wide range of distortion types and severity levels, thereby addressing the diverse requirements of surgeons. Furthermore, we construct a comprehensive benchmark simulating real-world surgical distortions, on which extensive experiments demonstrate that SurgVisAgent surpasses traditional single-task models, highlighting its potential as a unified solution for surgical assistance. |
| 2025-07-03 | [Public perspectives on the design of fusion energy facilities](http://arxiv.org/abs/2507.02207v1) | Nathan Kawamoto, Daniel Hoover et al. | As fusion energy technologies approach demonstration and commercial deployment, understanding public perspectives on future fusion facilities will be critical for achieving social license, especially because fusion energy facilities, unlike large fission reactors, may be sited in closer proximity to people and communities, due to distinct regulatory frameworks. In a departure from the 'decide-announce-defend' approach typically used to site energy infrastructure, we develop a participatory design methodology for collaboratively designing fusion energy facilities with prospective host communities. We present here our findings from a participatory design workshop that brought together 22 community participants and 34 engineering students. Our analysis of the textual and visual data from this workshop shows a range of design values and decision-making criteria with 'integrity' and 'respect' ranking highest among values and 'economic benefits' and 'environmental protection/safety' ranking highest among decision-making criteria. Salient design themes that emerge across facility concepts include connecting the history and legacy of the community to the design of the facility, care for workers, transparency and access to the facility, and health and safety of the host community. Participants reported predominantly positive sentiments, expressing joy and surprise as the workshop progressed from learning about fusion to designing the hypothetical facility. Our findings suggest that carrying out participatory design in the early stages of technology development can invite and make concrete public hopes and concerns, improve understanding of, and curiosity about, an emerging technology, build toward social license, and inform context-specific development of fusion energy facilities. |
| 2025-07-02 | [Statistical Inference for Responsiveness Verification](http://arxiv.org/abs/2507.02169v1) | Seung Hyun Cheon, Meredith Stewart et al. | Many safety failures in machine learning arise when models are used to assign predictions to people (often in settings like lending, hiring, or content moderation) without accounting for how individuals can change their inputs. In this work, we introduce a formal validation procedure for the responsiveness of predictions with respect to interventions on their features. Our procedure frames responsiveness as a type of sensitivity analysis in which practitioners control a set of changes by specifying constraints over interventions and distributions over downstream effects. We describe how to estimate responsiveness for the predictions of any model and any dataset using only black-box access, and how to use these estimates to support tasks such as falsification and failure probability estimation. We develop algorithms that construct these estimates by generating a uniform sample of reachable points, and demonstrate how they can promote safety in real-world applications such as recidivism prediction, organ transplant prioritization, and content moderation. |
| 2025-07-02 | [Reasoning or Not? A Comprehensive Evaluation of Reasoning LLMs for Dialogue Summarization](http://arxiv.org/abs/2507.02145v1) | Keyan Jin, Yapeng Wang et al. | Dialogue summarization is a challenging task with significant practical value in customer service, meeting analysis, and conversational AI. Although large language models (LLMs) have achieved substantial progress in summarization tasks, the performance of step-by-step reasoning architectures-specifically Long Chain-of-Thought (CoT) implementations such as OpenAI-o1 and DeepSeek-R1-remains unexplored for dialogue scenarios requiring concurrent abstraction and conciseness. In this work, we present the first comprehensive and systematic evaluation of state-of-the-art reasoning LLMs and non-reasoning LLMs across three major paradigms-generic, role-oriented, and query-oriented dialogue summarization. Our study spans diverse languages, domains, and summary lengths, leveraging strong benchmarks (SAMSum, DialogSum, CSDS, and QMSum) and advanced evaluation protocols that include both LLM-based automatic metrics and human-inspired criteria. Contrary to trends in other reasoning-intensive tasks, our findings show that explicit stepwise reasoning does not consistently improve dialogue summarization quality. Instead, reasoning LLMs are often prone to verbosity, factual inconsistencies, and less concise summaries compared to their non-reasoning counterparts. Through scenario-specific analyses and detailed case studies, we further identify when and why explicit reasoning may fail to benefit-or even hinder-summarization in complex dialogue contexts. Our work provides new insights into the limitations of current reasoning LLMs and highlights the need for targeted modeling and evaluation strategies for real-world dialogue summarization. |

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



