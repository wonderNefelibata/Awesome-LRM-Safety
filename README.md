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
| 2025-08-20 | [Novel Limits on Dark Photon Mixing from Radiation Safety](http://arxiv.org/abs/2508.14885v1) | Wen Yin | I propose a novel laboratory search for dark photons based on radiation-safety monitoring at synchrotron radiation facilities, including NanoTerasu, SPring-8, KEK-PF, and ESRF. Dark photons can be produced parasitically in undulators or via photon-mirror interactions, and subsequently traverse optical systems and shielding. Taking into account quantum effects and the internal structure of undulators, mirrors, and detectors, I show that even a simple Geiger-M\"uller counter, routinely used for radiation-safety monitoring, can detect such dark photons outside the shielding and set competitive limits on the kinetic mixing parameter down to $\chi \lesssim 5\times 10^{-6}$ in the eV mass range, providing some of the strongest bounds among laboratory searches. Because radiation safety is strictly regulated, the resulting limits can be regarded as robust and realistic constraints. |
| 2025-08-20 | [Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent](http://arxiv.org/abs/2508.14853v1) | Sajib Biswas, Mao Nishino et al. | As large language models (LLMs) are increasingly deployed in critical applications, ensuring their robustness and safety alignment remains a major challenge. Despite the overall success of alignment techniques such as reinforcement learning from human feedback (RLHF) on typical prompts, LLMs remain vulnerable to jailbreak attacks enabled by crafted adversarial triggers appended to user prompts. Most existing jailbreak methods either rely on inefficient searches over discrete token spaces or direct optimization of continuous embeddings. While continuous embeddings can be given directly to selected open-source models as input, doing so is not feasible for proprietary models. On the other hand, projecting these embeddings back into valid discrete tokens introduces additional complexity and often reduces attack effectiveness. We propose an intrinsic optimization method which directly optimizes relaxed one-hot encodings of the adversarial suffix tokens using exponentiated gradient descent coupled with Bregman projection, ensuring that the optimized one-hot encoding of each token always remains within the probability simplex. We provide theoretical proof of convergence for our proposed method and implement an efficient algorithm that effectively jailbreaks several widely used LLMs. Our method achieves higher success rates and faster convergence compared to three state-of-the-art baselines, evaluated on five open-source LLMs and four adversarial behavior datasets curated for evaluating jailbreak methods. In addition to individual prompt attacks, we also generate universal adversarial suffixes effective across multiple prompts and demonstrate transferability of optimized suffixes to different LLMs. |
| 2025-08-20 | [Safe and Transparent Robots for Human-in-the-Loop Meat Processing](http://arxiv.org/abs/2508.14763v1) | Sagar Parekh, Casey Grothoff et al. | Labor shortages have severely affected the meat processing sector. Automated technology has the potential to support the meat industry, assist workers, and enhance job quality. However, existing automation in meat processing is highly specialized, inflexible, and cost intensive. Instead of forcing manufacturers to buy a separate device for each step of the process, our objective is to develop general-purpose robotic systems that work alongside humans to perform multiple meat processing tasks. Through a recently conducted survey of industry experts, we identified two main challenges associated with integrating these collaborative robots alongside human workers. First, there must be measures to ensure the safety of human coworkers; second, the coworkers need to understand what the robot is doing. This paper addresses both challenges by introducing a safety and transparency framework for general-purpose meat processing robots. For safety, we implement a hand-detection system that continuously monitors nearby humans. This system can halt the robot in situations where the human comes into close proximity of the operating robot. We also develop an instrumented knife equipped with a force sensor that can differentiate contact between objects such as meat, bone, or fixtures. For transparency, we introduce a method that detects the robot's uncertainty about its performance and uses an LED interface to communicate that uncertainty to the human. Additionally, we design a graphical interface that displays the robot's plans and allows the human to provide feedback on the planned cut. Overall, our framework can ensure safe operation while keeping human workers in-the-loop about the robot's actions which we validate through a user study. |
| 2025-08-20 | [Challenges of Virtual Validation and Verification for Automotive Functions](http://arxiv.org/abs/2508.14747v1) | Beatriz Cabrero-Daniel, Mazen Mohamad | Verification and validation of vehicles is a complex yet critical process, particularly for ensuring safety and coverage through simulations. However, achieving realistic and useful simulations comes with significant challenges. To explore these challenges, we conducted a workshop with experts in the field, allowing them to brainstorm key obstacles. Following this, we distributed a survey to consolidate findings and gain further insights into potential solutions. The experts identified 17 key challenges, along with proposed solutions, an assessment of whether they represent next steps for research, and the roadblocks to their implementation. While a lack of resources was not initially highlighted as a major challenge, utilizing more resources emerged as a critical necessity when experts discussed solutions. Interestingly, we expected some of these challenges to have already been addressed or to have systematic solutions readily available, given the collective expertise in the field. Many of the identified problems already have known solutions, allowing us to shift focus towards unresolved challenges and share the next steps with the broader community. |
| 2025-08-20 | [Multiscale Video Transformers for Class Agnostic Segmentation in Autonomous Driving](http://arxiv.org/abs/2508.14729v1) | Leila Cheshmi, Mennatullah Siam | Ensuring safety in autonomous driving is a complex challenge requiring handling unknown objects and unforeseen driving scenarios. We develop multiscale video transformers capable of detecting unknown objects using only motion cues. Video semantic and panoptic segmentation often relies on known classes seen during training, overlooking novel categories. Recent visual grounding with large language models is computationally expensive, especially for pixel-level output. We propose an efficient video transformer trained end-to-end for class-agnostic segmentation without optical flow. Our method uses multi-stage multiscale query-memory decoding and a scale-specific random drop-token to ensure efficiency and accuracy, maintaining detailed spatiotemporal features with a shared, learnable memory module. Unlike conventional decoders that compress features, our memory-centric design preserves high-resolution information at multiple scales. We evaluate on DAVIS'16, KITTI, and Cityscapes. Our method consistently outperforms multiscale baselines while being efficient in GPU memory and run-time, demonstrating a promising direction for real-time, robust dense prediction in safety-critical robotics. |
| 2025-08-20 | [Emerson-Lei and Manna-Pnueli Games for LTLf+ and PPLTL+ Synthesis](http://arxiv.org/abs/2508.14725v1) | Daniel Hausmann, Shufang Zhu et al. | Recently, the Manna-Pnueli Hierarchy has been used to define the temporal logics LTLfp and PPLTLp, which allow to use finite-trace LTLf/PPLTL techniques in infinite-trace settings while achieving the expressiveness of full LTL. In this paper, we present the first actual solvers for reactive synthesis in these logics. These are based on games on graphs that leverage DFA-based techniques from LTLf/PPLTL to construct the game arena. We start with a symbolic solver based on Emerson-Lei games, which reduces lower-class properties (guarantee, safety) to higher ones (recurrence, persistence) before solving the game. We then introduce Manna-Pnueli games, which natively embed Manna-Pnueli objectives into the arena. These games are solved by composing solutions to a DAG of simpler Emerson-Lei games, resulting in a provably more efficient approach. We implemented the solvers and practically evaluated their performance on a range of representative formulas. The results show that Manna-Pnueli games often offer significant advantages, though not universally, indicating that combining both approaches could further enhance practical performance. |
| 2025-08-20 | [Data-Driven Probabilistic Evaluation of Logic Properties with PAC-Confidence on Mealy Machines](http://arxiv.org/abs/2508.14710v1) | Swantje Plambeck, Ali Salamati et al. | Cyber-Physical Systems (CPS) are complex systems that require powerful models for tasks like verification, diagnosis, or debugging. Often, suitable models are not available and manual extraction is difficult. Data-driven approaches then provide a solution to, e.g., diagnosis tasks and verification problems based on data collected from the system. In this paper, we consider CPS with a discrete abstraction in the form of a Mealy machine. We propose a data-driven approach to determine the safety probability of the system on a finite horizon of n time steps. The approach is based on the Probably Approximately Correct (PAC) learning paradigm. Thus, we elaborate a connection between discrete logic and probabilistic reachability analysis of systems, especially providing an additional confidence on the determined probability. The learning process follows an active learning paradigm, where new learning data is sampled in a guided way after an initial learning set is collected. We validate the approach with a case study on an automated lane-keeping system. |
| 2025-08-20 | [Entropy-Constrained Strategy Optimization in Urban Floods: A Multi-Agent Framework with LLM and Knowledge Graph Integration](http://arxiv.org/abs/2508.14654v1) | Peilin Ji, Xiao Xue et al. | In recent years, the increasing frequency of extreme urban rainfall events has posed significant challenges to emergency scheduling systems. Urban flooding often leads to severe traffic congestion and service disruptions, threatening public safety and mobility. However, effective decision making remains hindered by three key challenges: (1) managing trade-offs among competing goals (e.g., traffic flow, task completion, and risk mitigation) requires dynamic, context-aware strategies; (2) rapidly evolving environmental conditions render static rules inadequate; and (3) LLM-generated strategies frequently suffer from semantic instability and execution inconsistency. Existing methods fail to align perception, global optimization, and multi-agent coordination within a unified framework. To tackle these challenges, we introduce H-J, a hierarchical multi-agent framework that integrates knowledge-guided prompting, entropy-constrained generation, and feedback-driven optimization. The framework establishes a closed-loop pipeline spanning from multi-source perception to strategic execution and continuous refinement. We evaluate H-J on real-world urban topology and rainfall data under three representative conditions: extreme rainfall, intermittent bursts, and daily light rain. Experiments show that H-J outperforms rule-based and reinforcement-learning baselines in traffic smoothness, task success rate, and system robustness. These findings highlight the promise of uncertainty-aware, knowledge-constrained LLM-based approaches for enhancing resilience in urban flood response. |
| 2025-08-20 | [A Fuzzy-Enhanced Explainable AI Framework for Flight Continuous Descent Operations Classification](http://arxiv.org/abs/2508.14618v1) | Amin Noroozi, Sandaruwan K. Sethunge et al. | Continuous Descent Operations (CDO) involve smooth, idle-thrust descents that avoid level-offs, reducing fuel burn, emissions, and noise while improving efficiency and passenger comfort. Despite its operational and environmental benefits, limited research has systematically examined the factors influencing CDO performance. Moreover, many existing methods in related areas, such as trajectory optimization, lack the transparency required in aviation, where explainability is critical for safety and stakeholder trust. This study addresses these gaps by proposing a Fuzzy-Enhanced Explainable AI (FEXAI) framework that integrates fuzzy logic with machine learning and SHapley Additive exPlanations (SHAP) analysis. For this purpose, a comprehensive dataset of 29 features, including 11 operational and 18 weather-related features, was collected from 1,094 flights using Automatic Dependent Surveillance-Broadcast (ADS-B) data. Machine learning models and SHAP were then applied to classify flights' CDO adherence levels and rank features by importance. The three most influential features, as identified by SHAP scores, were then used to construct a fuzzy rule-based classifier, enabling the extraction of interpretable fuzzy rules. All models achieved classification accuracies above 90%, with FEXAI providing meaningful, human-readable rules for operational users. Results indicated that the average descent rate within the arrival route, the number of descent segments, and the average change in directional heading during descent were the strongest predictors of CDO performance. The FEXAI method proposed in this study presents a novel pathway for operational decision support and could be integrated into aviation tools to enable real-time advisories that maintain CDO adherence under varying operational conditions. |
| 2025-08-20 | [Reliable Smoke Detection via Optical Flow-Guided Feature Fusion and Transformer-Based Uncertainty Modeling](http://arxiv.org/abs/2508.14597v1) | Nitish Kumar Mahala, Muzammil Khan et al. | Fire outbreaks pose critical threats to human life and infrastructure, necessitating high-fidelity early-warning systems that detect combustion precursors such as smoke. However, smoke plumes exhibit complex spatiotemporal dynamics influenced by illumination variability, flow kinematics, and environmental noise, undermining the reliability of traditional detectors. To address these challenges without the logistical complexity of multi-sensor arrays, we propose an information-fusion framework by integrating smoke feature representations extracted from monocular imagery. Specifically, a Two-Phase Uncertainty-Aware Shifted Windows Transformer for robust and reliable smoke detection, leveraging a novel smoke segmentation dataset, constructed via optical flow-based motion encoding, is proposed. The optical flow estimation is performed with a four-color-theorem-inspired dual-phase level-set fractional-order variational model, which preserves motion discontinuities. The resulting color-encoded optical flow maps are fused with appearance cues via a Gaussian Mixture Model to generate binary segmentation masks of the smoke regions. These fused representations are fed into the novel Shifted-Windows Transformer, which is augmented with a multi-scale uncertainty estimation head and trained under a two-phase learning regimen. First learning phase optimizes smoke detection accuracy, while during the second phase, the model learns to estimate plausibility confidence in its predictions by jointly modeling aleatoric and epistemic uncertainties. Extensive experiments using multiple evaluation metrics and comparative analysis with state-of-the-art approaches demonstrate superior generalization and robustness, offering a reliable solution for early fire detection in surveillance, industrial safety, and autonomous monitoring applications. |
| 2025-08-20 | [Safety-Critical Learning for Long-Tail Events: The TUM Traffic Accident Dataset](http://arxiv.org/abs/2508.14567v1) | Walter Zimmer, Ross Greer et al. | Even though a significant amount of work has been done to increase the safety of transportation networks, accidents still occur regularly. They must be understood as an unavoidable and sporadic outcome of traffic networks. We present the TUM Traffic Accident (TUMTraf-A) dataset, a collection of real-world highway accidents. It contains ten sequences of vehicle crashes at high-speed driving with 294,924 labeled 2D and 93,012 labeled 3D boxes and track IDs within 48,144 labeled frames recorded from four roadside cameras and LiDARs at 10 Hz. The dataset contains ten object classes and is provided in the OpenLABEL format. We propose Accid3nD, an accident detection model that combines a rule-based approach with a learning-based one. Experiments and ablation studies on our dataset show the robustness of our proposed method. The dataset, model, and code are available on our project website: https://tum-traffic-dataset.github.io/tumtraf-a. |
| 2025-08-20 | [Adversarial Generation and Collaborative Evolution of Safety-Critical Scenarios for Autonomous Vehicles](http://arxiv.org/abs/2508.14527v1) | Jiangfan Liu, Yongkang Guo et al. | The generation of safety-critical scenarios in simulation has become increasingly crucial for safety evaluation in autonomous vehicles prior to road deployment in society. However, current approaches largely rely on predefined threat patterns or rule-based strategies, which limit their ability to expose diverse and unforeseen failure modes. To overcome these, we propose ScenGE, a framework that can generate plentiful safety-critical scenarios by reasoning novel adversarial cases and then amplifying them with complex traffic flows. Given a simple prompt of a benign scene, it first performs Meta-Scenario Generation, where a large language model, grounded in structured driving knowledge, infers an adversarial agent whose behavior poses a threat that is both plausible and deliberately challenging. This meta-scenario is then specified in executable code for precise in-simulator control. Subsequently, Complex Scenario Evolution uses background vehicles to amplify the core threat introduced by Meta-Scenario. It builds an adversarial collaborator graph to identify key agent trajectories for optimization. These perturbations are designed to simultaneously reduce the ego vehicle's maneuvering space and create critical occlusions. Extensive experiments conducted on multiple reinforcement learning based AV models show that ScenGE uncovers more severe collision cases (+31.96%) on average than SoTA baselines. Additionally, our ScenGE can be applied to large model based AV systems and deployed on different simulators; we further observe that adversarial training on our scenarios improves the model robustness. Finally, we validate our framework through real-world vehicle tests and human evaluation, confirming that the generated scenarios are both plausible and critical. We hope our paper can build up a critical step towards building public trust and ensuring their safe deployment. |
| 2025-08-20 | [Great GATsBi: Hybrid, Multimodal, Trajectory Forecasting for Bicycles using Anticipation Mechanism](http://arxiv.org/abs/2508.14523v1) | Kevin Riehl, Shaimaa K. El-Baklish et al. | Accurate prediction of road user movement is increasingly required by many applications ranging from advanced driver assistance systems to autonomous driving, and especially crucial for road safety. Even though most traffic accident fatalities account to bicycles, they have received little attention, as previous work focused mainly on pedestrians and motorized vehicles. In this work, we present the Great GATsBi, a domain-knowledge-based, hybrid, multimodal trajectory prediction framework for bicycles. The model incorporates both physics-based modeling (inspired by motorized vehicles) and social-based modeling (inspired by pedestrian movements) to explicitly account for the dual nature of bicycle movement. The social interactions are modeled with a graph attention network, and include decayed historical, but also anticipated, future trajectory data of a bicycles neighborhood, following recent insights from psychological and social studies. The results indicate that the proposed ensemble of physics models -- performing well in the short-term predictions -- and social models -- performing well in the long-term predictions -- exceeds state-of-the-art performance. We also conducted a controlled mass-cycling experiment to demonstrate the framework's performance when forecasting bicycle trajectories and modeling social interactions with road users. |
| 2025-08-20 | [Constraining Common Envelope Evolution in Binary Neutron Star Formation with Combined Galactic and Gravitational-Wave Observations](http://arxiv.org/abs/2508.14397v1) | Zhiwei Chen, Jihui Zhang et al. | Binary neutron stars (BNSs) are among the most interesting sources for multimessenger studies. A number of recently discovered BNSs in the Milky Way by radio telescopes have added new information to the parameter distribution of the Galactic BNSs. The scarce of BNS mergers during the O4 run of the LIGO-Virgo-Kagra (LVK) suggests a BNS local merger rate six times lower than the previous constraint obtained by O1-O3 runs. With these new multimessenger observations, in this letter, we adopt the compact binary population synthesis model and Bayesian analysis to constrain the formation and evolution of BNSs, especially the common envelope (CE) evolution. We find that it is required: (1) a fraction ($f_{\rm HG}\sim0.8$) but not all of the Hertzsprung gap donors merged with their companions in the CE stage, in order to simultaneously explain the low BNS merger rate density and the existence of the short-orbital-period ($\lesssim 1$ day) Galactic BNSs, different from either all ($f_{\rm HG}=1$) or none ($f_{\rm HG}=0$) adopted in previous studies; (2) a large CE ejection efficiency $\alpha$ ($\sim 5$), in order to explain the existence of the long-orbital-period ($\gtrsim 10$ day) Galactic BNSs. |
| 2025-08-20 | [Fair-CoPlan: Negotiated Flight Planning with Fair Deconfliction for Urban Air Mobility](http://arxiv.org/abs/2508.14380v1) | Nicole Fronda, Phil Smith et al. | Urban Air Mobility (UAM) is an emerging transportation paradigm in which Uncrewed Aerial Systems (UAS) autonomously transport passengers and goods in cities. The UAS have different operators with different, sometimes competing goals, yet must share the airspace. We propose a negotiated, semi-distributed flight planner that optimizes UAS' flight lengths {\em in a fair manner}. Current flight planners might result in some UAS being given disproportionately shorter flight paths at the expense of others. We introduce Fair-CoPlan, a planner in which operators and a Provider of Service to the UAM (PSU) together compute \emph{fair} flight paths. Fair-CoPlan has three steps: First, the PSU constrains take-off and landing choices for flights based on capacity at and around vertiports. Then, operators plan independently under these constraints. Finally, the PSU resolves any conflicting paths, optimizing for path length fairness. By fairly spreading the cost of deconfliction Fair-CoPlan encourages wider participation in UAM, ensures safety of the airspace and the areas below it, and promotes greater operator flexibility. We demonstrate Fair-CoPlan through simulation experiments and find fairer outcomes than a non-fair planner with minor delays as a trade-off. |
| 2025-08-20 | [Deep Learning for Taxol Exposure Analysis: A New Cell Image Dataset and Attention-Based Baseline Model](http://arxiv.org/abs/2508.14349v1) | Sean Fletcher, Gabby Scott et al. | Monitoring the effects of the chemotherapeutic agent Taxol at the cellular level is critical for both clinical evaluation and biomedical research. However, existing detection methods require specialized equipment, skilled personnel, and extensive sample preparation, making them expensive, labor-intensive, and unsuitable for high-throughput or real-time analysis. Deep learning approaches have shown great promise in medical and biological image analysis, enabling automated, high-throughput assessment of cellular morphology. Yet, no publicly available dataset currently exists for automated morphological analysis of cellular responses to Taxol exposure. To address this gap, we introduce a new microscopy image dataset capturing C6 glioma cells treated with varying concentrations of Taxol. To provide an effective solution for Taxol concentration classification and establish a benchmark for future studies on this dataset, we propose a baseline model named ResAttention-KNN, which combines a ResNet-50 with Convolutional Block Attention Modules and uses a k-Nearest Neighbors classifier in the learned embedding space. This model integrates attention-based refinement and non-parametric classification to enhance robustness and interpretability. Both the dataset and implementation are publicly released to support reproducibility and facilitate future research in vision-based biomedical analysis. |
| 2025-08-19 | [Astrophysical or Terrestrial: Machine learning classification of gravitational-wave candidates using multiple-search information](http://arxiv.org/abs/2508.14242v1) | Seiya Tsukamoto, Andrew Toivonen et al. | Low-latency gravitational-wave alerts provide the greater multi-messenger community with information about the candidate events detected by the International Gravitational-Wave Network (IGWN). Prompt release of data products such as the sky localization, false alarm rate (FAR), and $p_\mathrm{astro}$ values allow astronomers to make informed decisions on which candidate gravitational-wave events merit target of opportunity (ToO) follow-up. However, false alarms, often referred to as "glitches", where a gravitational-wave candidate, or trigger, is the result of terrestrial noise, are an inherent part of gravitational-wave searches. In addition, with the presence of multiple gravitational-wave searches, different searches may have varying assessments of the significance of a given trigger. As a complement to quantities such as $p_\mathrm{astro}$, we provide a Machine Learning (ML) based approach to determining whether candidate events are astrophysical or terrestrial in nature, specifically a classifier that utilizes information provided by multiple low-latency search pipelines in its feature space. This classifier has a performance an Area Under the Receiver Operating Characteristic Curve (AUC) of 0.96 and accuracy of 0.90 on the Mock Data Challenge training set and an AUC of 0.93 and accuracy of 0.86 on events from the Advanced LIGO (aLIGO)'s and Advanced Virgo (AdVirgo)'s third observing run (O3). |
| 2025-08-19 | [SLAM-based Safe Indoor Exploration Strategy](http://arxiv.org/abs/2508.14235v1) | Omar Mostafa, Nikolaos Evangeliou et al. | This paper suggests a 2D exploration strategy for a planar space cluttered with obstacles. Rather than using point robots capable of adjusting their position and altitude instantly, this research is tailored to classical agents with circular footprints that cannot control instantly their pose. Inhere, a self-balanced dual-wheeled differential drive system is used to explore the place. The system is equipped with linear accelerometers and angular gyroscopes, a 3D-LiDAR, and a forward-facing RGB-D camera. The system performs RTAB-SLAM using the IMU and the LiDAR, while the camera is used for loop closures. The mobile agent explores the planar space using a safe skeleton approach that places the agent as far as possible from the static obstacles. During the exploration strategy, the heading is towards any offered openings of the space. This space exploration strategy has as its highest priority the agent's safety in avoiding the obstacles followed by the exploration of undetected space. Experimental studies with a ROS-enabled mobile agent are presented indicating the path planning strategy while exploring the space. |
| 2025-08-19 | [Incident Analysis for AI Agents](http://arxiv.org/abs/2508.14231v1) | Carson Ezell, Xavier Roberts-Gaal et al. | As AI agents become more widely deployed, we are likely to see an increasing number of incidents: events involving AI agent use that directly or indirectly cause harm. For example, agents could be prompt-injected to exfiltrate private information or make unauthorized purchases. Structured information about such incidents (e.g., user prompts) can help us understand their causes and prevent future occurrences. However, existing incident reporting processes are not sufficient for understanding agent incidents. In particular, such processes are largely based on publicly available data, which excludes useful, but potentially sensitive, information such as an agent's chain of thought or browser history. To inform the development of new, emerging incident reporting processes, we propose an incident analysis framework for agents. Drawing on systems safety approaches, our framework proposes three types of factors that can cause incidents: system-related (e.g., CBRN training data), contextual (e.g., prompt injections), and cognitive (e.g., misunderstanding a user request). We also identify specific information that could help clarify which factors are relevant to a given incident: activity logs, system documentation and access, and information about the tools an agent uses. We provide recommendations for 1) what information incident reports should include and 2) what information developers and deployers should retain and make available to incident investigators upon request. As we transition to a world with more agents, understanding agent incidents will become increasingly crucial for managing risks. |
| 2025-08-19 | [Reliability comparison of vessel trajectory prediction models via Probability of Detection](http://arxiv.org/abs/2508.14198v1) | Zahra Rastin, Kathrin Donandt et al. | This contribution addresses vessel trajectory prediction (VTP), focusing on the evaluation of different deep learning-based approaches. The objective is to assess model performance in diverse traffic complexities and compare the reliability of the approaches. While previous VTP models overlook the specific traffic situation complexity and lack reliability assessments, this research uses a probability of detection analysis to quantify model reliability in varying traffic scenarios, thus going beyond common error distribution analyses. All models are evaluated on test samples categorized according to their traffic situation during the prediction horizon, with performance metrics and reliability estimates obtained for each category. The results of this comprehensive evaluation provide a deeper understanding of the strengths and weaknesses of the different prediction approaches, along with their reliability in terms of the prediction horizon lengths for which safe forecasts can be guaranteed. These findings can inform the development of more reliable vessel trajectory prediction approaches, enhancing safety and efficiency in future inland waterways navigation. |

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



