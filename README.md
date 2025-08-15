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
| 2025-08-14 | [Psyche-R1: Towards Reliable Psychological LLMs through Unified Empathy, Expertise, and Reasoning](http://arxiv.org/abs/2508.10848v1) | Chongyuan Dai, Jinpeng Hu et al. | Amidst a shortage of qualified mental health professionals, the integration of large language models (LLMs) into psychological applications offers a promising way to alleviate the growing burden of mental health disorders. Recent reasoning-augmented LLMs have achieved remarkable performance in mathematics and programming, while research in the psychological domain has predominantly emphasized emotional support and empathetic dialogue, with limited attention to reasoning mechanisms that are beneficial to generating reliable responses. Therefore, in this paper, we propose Psyche-R1, the first Chinese psychological LLM that jointly integrates empathy, psychological expertise, and reasoning, built upon a novel data curation pipeline. Specifically, we design a comprehensive data synthesis pipeline that produces over 75k high-quality psychological questions paired with detailed rationales, generated through chain-of-thought (CoT) reasoning and iterative prompt-rationale optimization, along with 73k empathetic dialogues. Subsequently, we employ a hybrid training strategy wherein challenging samples are identified through a multi-LLM cross-selection strategy for group relative policy optimization (GRPO) to improve reasoning ability, while the remaining data is used for supervised fine-tuning (SFT) to enhance empathetic response generation and psychological domain knowledge. Extensive experiment results demonstrate the effectiveness of the Psyche-R1 across several psychological benchmarks, where our 7B Psyche-R1 achieves comparable results to 671B DeepSeek-R1. |
| 2025-08-14 | [The SET Perceptual Factors Framework: Towards Assured Perception for Autonomous Systems](http://arxiv.org/abs/2508.10798v1) | Troi Williams | Future autonomous systems promise significant societal benefits, yet their deployment raises concerns about safety and trustworthiness. A key concern is assuring the reliability of robot perception, as perception seeds safe decision-making. Failures in perception are often due to complex yet common environmental factors and can lead to accidents that erode public trust. To address this concern, we introduce the SET (Self, Environment, and Target) Perceptual Factors Framework. We designed the framework to systematically analyze how factors such as weather, occlusion, or sensor limitations negatively impact perception. To achieve this, the framework employs SET State Trees to categorize where such factors originate and SET Factor Trees to model how these sources and factors impact perceptual tasks like object detection or pose estimation. Next, we develop Perceptual Factor Models using both trees to quantify the uncertainty for a given task. Our framework aims to promote rigorous safety assurances and cultivate greater public understanding and trust in autonomous systems by offering a transparent and standardized method for identifying, modeling, and communicating perceptual risks. |
| 2025-08-14 | [Learning Task Execution Hierarchies for Redundant Robots](http://arxiv.org/abs/2508.10780v1) | Alessandro Adami, Aris Synodinos et al. | Modern robotic systems, such as mobile manipulators, humanoids, and aerial robots with arms, often possess high redundancy, enabling them to perform multiple tasks simultaneously. Managing this redundancy is key to achieving reliable and flexible behavior. A widely used approach is the Stack of Tasks (SoT), which organizes control objectives by priority within a unified framework. However, traditional SoTs are manually designed by experts, limiting their adaptability and accessibility. This paper introduces a novel framework that automatically learns both the hierarchy and parameters of a SoT from user-defined objectives. By combining Reinforcement Learning and Genetic Programming, the system discovers task priorities and control strategies without manual intervention. A cost function based on intuitive metrics such as precision, safety, and execution time guides the learning process. We validate our method through simulations and experiments on the mobile-YuMi platform, a dual-arm mobile manipulator with high redundancy. Results show that the learned SoTs enable the robot to dynamically adapt to changing environments and inputs, balancing competing objectives while maintaining robust task execution. This approach provides a general and user-friendly solution for redundancy management in complex robots, advancing human-centered robot programming and reducing the need for expert design. |
| 2025-08-14 | [Synthesis of Deep Neural Networks with Safe Robust Adaptive Control for Reliable Operation of Wheeled Mobile Robots](http://arxiv.org/abs/2508.10634v1) | Mehdi Heydari Shahna, Jouni Mattila | Deep neural networks (DNNs) can enable precise control while maintaining low computational costs by circumventing the need for dynamic modeling. However, the deployment of such black-box approaches remains challenging for heavy-duty wheeled mobile robots (WMRs), which are subject to strict international standards and prone to faults and disturbances. We designed a hierarchical control policy for heavy-duty WMRs, monitored by two safety layers with differing levels of authority. To this end, a DNN policy was trained and deployed as the primary control strategy, providing high-precision performance under nominal operating conditions. When external disturbances arise and reach a level of intensity such that the system performance falls below a predefined threshold, a low-level safety layer intervenes by deactivating the primary control policy and activating a model-free robust adaptive control (RAC) policy. This transition enables the system to continue operating while ensuring stability by effectively managing the inherent trade-off between system robustness and responsiveness. Regardless of the control policy in use, a high-level safety layer continuously monitors system performance during operation. It initiates a shutdown only when disturbances become sufficiently severe such that compensation is no longer viable and continued operation would jeopardize the system or its environment. The proposed synthesis of DNN and RAC policy guarantees uniform exponential stability of the entire WMR system while adhering to safety standards to some extent. The effectiveness of the proposed approach was further validated through real-time experiments using a 6,000 kg WMR. |
| 2025-08-14 | [DEV: A Driver-Environment-Vehicle Closed-Loop Framework for Risk-Aware Adaptive Automation of Driving](http://arxiv.org/abs/2508.10618v1) | Anaïs Halin, Christel Devue et al. | The increasing integration of automation in vehicles aims to enhance both safety and comfort, but it also introduces new risks, including driver disengagement, reduced situation awareness, and mode confusion. In this work, we propose the DEV framework, a closed-loop framework for risk-aware adaptive driving automation that captures the dynamic interplay between the driver, the environment, and the vehicle. The framework promotes to continuously adjusting the operational level of automation based on a risk management strategy. The real-time risk assessment supports smoother transitions and effective cooperation between the driver and the automation system. Furthermore, we introduce a nomenclature of indexes corresponding to each core component, namely driver involvement, environment complexity, and vehicle engagement, and discuss how their interaction influences driving risk. The DEV framework offers a comprehensive perspective to align multidisciplinary research efforts and guide the development of dynamic, risk-aware driving automation systems. |
| 2025-08-14 | [Towards Powerful and Practical Patch Attacks for 2D Object Detection in Autonomous Driving](http://arxiv.org/abs/2508.10600v1) | Yuxin Cao, Yedi Zhang et al. | Learning-based autonomous driving systems remain critically vulnerable to adversarial patches, posing serious safety and security risks in their real-world deployment. Black-box attacks, notable for their high attack success rate without model knowledge, are especially concerning, with their transferability extensively studied to reduce computational costs compared to query-based attacks. Previous transferability-based black-box attacks typically adopt mean Average Precision (mAP) as the evaluation metric and design training loss accordingly. However, due to the presence of multiple detected bounding boxes and the relatively lenient Intersection over Union (IoU) thresholds, the attack effectiveness of these approaches is often overestimated, resulting in reduced success rates in practical attacking scenarios. Furthermore, patches trained on low-resolution data often fail to maintain effectiveness on high-resolution images, limiting their transferability to autonomous driving datasets. To fill this gap, we propose P$^3$A, a Powerful and Practical Patch Attack framework for 2D object detection in autonomous driving, specifically optimized for high-resolution datasets. First, we introduce a novel metric, Practical Attack Success Rate (PASR), to more accurately quantify attack effectiveness with greater relevance for pedestrian safety. Second, we present a tailored Localization-Confidence Suppression Loss (LCSL) to improve attack transferability under PASR. Finally, to maintain the transferability for high-resolution datasets, we further incorporate the Probabilistic Scale-Preserving Padding (PSPP) into the patch attack pipeline as a data preprocessing step. Extensive experiments show that P$^3$A outperforms state-of-the-art attacks on unseen models and unseen high-resolution datasets, both under the proposed practical IoU-based evaluation metric and the previous mAP-based metrics. |
| 2025-08-14 | [SpaRC-AD: A Baseline for Radar-Camera Fusion in End-to-End Autonomous Driving](http://arxiv.org/abs/2508.10567v1) | Philipp Wolters, Johannes Gilg et al. | End-to-end autonomous driving systems promise stronger performance through unified optimization of perception, motion forecasting, and planning. However, vision-based approaches face fundamental limitations in adverse weather conditions, partial occlusions, and precise velocity estimation - critical challenges in safety-sensitive scenarios where accurate motion understanding and long-horizon trajectory prediction are essential for collision avoidance. To address these limitations, we propose SpaRC-AD, a query-based end-to-end camera-radar fusion framework for planning-oriented autonomous driving. Through sparse 3D feature alignment, and doppler-based velocity estimation, we achieve strong 3D scene representations for refinement of agent anchors, map polylines and motion modelling. Our method achieves strong improvements over the state-of-the-art vision-only baselines across multiple autonomous driving tasks, including 3D detection (+4.8% mAP), multi-object tracking (+8.3% AMOTA), online mapping (+1.8% mAP), motion prediction (-4.0% mADE), and trajectory planning (-0.1m L2 and -9% TPC). We achieve both spatial coherence and temporal consistency on multiple challenging benchmarks, including real-world open-loop nuScenes, long-horizon T-nuScenes, and closed-loop simulator Bench2Drive. We show the effectiveness of radar-based fusion in safety-critical scenarios where accurate motion understanding and long-horizon trajectory prediction are essential for collision avoidance. The source code of all experiments is available at https://phi-wol.github.io/sparcad/ |
| 2025-08-14 | [Reproducible Physiological Features in Affective Computing: A Preliminary Analysis on Arousal Modeling](http://arxiv.org/abs/2508.10561v1) | Andrea Gargano, Jasin Machkour et al. | In Affective Computing, a key challenge lies in reliably linking subjective emotional experiences with objective physiological markers. This preliminary study addresses the issue of reproducibility by identifying physiological features from cardiovascular and electrodermal signals that are associated with continuous self-reports of arousal levels. Using the Continuously Annotated Signal of Emotion dataset, we analyzed 164 features extracted from cardiac and electrodermal signals of 30 participants exposed to short emotion-evoking videos. Feature selection was performed using the Terminating-Random Experiments (T-Rex) method, which performs variable selection systematically controlling a user-defined target False Discovery Rate. Remarkably, among all candidate features, only two electrodermal-derived features exhibited reproducible and statistically significant associations with arousal, achieving a 100\% confirmation rate. These results highlight the necessity of rigorous reproducibility assessments in physiological features selection, an aspect often overlooked in Affective Computing. Our approach is particularly promising for applications in safety-critical environments requiring trustworthy and reliable white box models, such as mental disorder recognition and human-robot interaction systems. |
| 2025-08-14 | [eDIF: A European Deep Inference Fabric for Remote Interpretability of LLM](http://arxiv.org/abs/2508.10553v1) | Irma Heithoff. Marc Guggenberger, Sandra Kalogiannis et al. | This paper presents a feasibility study on the deployment of a European Deep Inference Fabric (eDIF), an NDIF-compatible infrastructure designed to support mechanistic interpretability research on large language models. The need for widespread accessibility of LLM interpretability infrastructure in Europe drives this initiative to democratize advanced model analysis capabilities for the research community. The project introduces a GPU-based cluster hosted at Ansbach University of Applied Sciences and interconnected with partner institutions, enabling remote model inspection via the NNsight API. A structured pilot study involving 16 researchers from across Europe evaluated the platform's technical performance, usability, and scientific utility. Users conducted interventions such as activation patching, causal tracing, and representation analysis on models including GPT-2 and DeepSeek-R1-70B. The study revealed a gradual increase in user engagement, stable platform performance throughout, and a positive reception of the remote experimentation capabilities. It also marked the starting point for building a user community around the platform. Identified limitations such as prolonged download durations for activation data as well as intermittent execution interruptions are addressed in the roadmap for future development. This initiative marks a significant step towards widespread accessibility of LLM interpretability infrastructure in Europe and lays the groundwork for broader deployment, expanded tooling, and sustained community collaboration in mechanistic interpretability research. |
| 2025-08-14 | [Virtual Sensing for Solder Layer Degradation and Temperature Monitoring in IGBT Modules](http://arxiv.org/abs/2508.10515v1) | Andrea Urgolo, Monika Stipsitz et al. | Monitoring the degradation state of Insulated Gate Bipolar Transistor (IGBT) modules is essential for ensuring the reliability and longevity of power electronic systems, especially in safety-critical and high-performance applications. However, direct measurement of key degradation indicators - such as junction temperature, solder fatigue or delamination - remains challenging due to the physical inaccessibility of internal components and the harsh environment. In this context, machine learning-based virtual sensing offers a promising alternative by bridging the gap from feasible sensor placement to the relevant but inaccessible locations. This paper explores the feasibility of estimating the degradation state of solder layers, and the corresponding full temperature maps based on a limited number of physical sensors. Based on synthetic data of a specific degradation mode, we obtain a high accuracy in the estimation of the degraded solder area (1.17% mean absolute error), and are able to reproduce the surface temperature of the IGBT with a maximum relative error of 4.56% (corresponding to an average relative error of 0.37%). |
| 2025-08-14 | [A Segmentation-driven Editing Method for Bolt Defect Augmentation and Detection](http://arxiv.org/abs/2508.10509v1) | Yangjie Xiao, Ke Zhang et al. | Bolt defect detection is critical to ensure the safety of transmission lines. However, the scarcity of defect images and imbalanced data distributions significantly limit detection performance. To address this problem, we propose a segmentationdriven bolt defect editing method (SBDE) to augment the dataset. First, a bolt attribute segmentation model (Bolt-SAM) is proposed, which enhances the segmentation of complex bolt attributes through the CLAHE-FFT Adapter (CFA) and Multipart- Aware Mask Decoder (MAMD), generating high-quality masks for subsequent editing tasks. Second, a mask optimization module (MOD) is designed and integrated with the image inpainting model (LaMa) to construct the bolt defect attribute editing model (MOD-LaMa), which converts normal bolts into defective ones through attribute editing. Finally, an editing recovery augmentation (ERA) strategy is proposed to recover and put the edited defect bolts back into the original inspection scenes and expand the defect detection dataset. We constructed multiple bolt datasets and conducted extensive experiments. Experimental results demonstrate that the bolt defect images generated by SBDE significantly outperform state-of-the-art image editing models, and effectively improve the performance of bolt defect detection, which fully verifies the effectiveness and application potential of the proposed method. The code of the project is available at https://github.com/Jay-xyj/SBDE. |
| 2025-08-14 | [PASS: Probabilistic Agentic Supernet Sampling for Interpretable and Adaptive Chest X-Ray Reasoning](http://arxiv.org/abs/2508.10501v1) | Yushi Feng, Junye Du et al. | Existing tool-augmented agentic systems are limited in the real world by (i) black-box reasoning steps that undermine trust of decision-making and pose safety risks, (ii) poor multimodal integration, which is inherently critical for healthcare tasks, and (iii) rigid and computationally inefficient agentic pipelines. We introduce PASS (Probabilistic Agentic Supernet Sampling), the first multimodal framework to address these challenges in the context of Chest X-Ray (CXR) reasoning. PASS adaptively samples agentic workflows over a multi-tool graph, yielding decision paths annotated with interpretable probabilities. Given the complex CXR reasoning task with multimodal medical data, PASS leverages its learned task-conditioned distribution over the agentic supernet. Thus, it adaptively selects the most suitable tool at each supernet layer, offering probability-annotated trajectories for post-hoc audits and directly enhancing medical AI safety. PASS also continuously compresses salient findings into an evolving personalized memory, while dynamically deciding whether to deepen its reasoning path or invoke an early exit for efficiency. To optimize a Pareto frontier balancing performance and cost, we design a novel three-stage training procedure, including expert knowledge warm-up, contrastive path-ranking, and cost-aware reinforcement learning. To facilitate rigorous evaluation, we introduce CAB-E, a comprehensive benchmark for multi-step, safety-critical, free-form CXR reasoning. Experiments across various benchmarks validate that PASS significantly outperforms strong baselines in multiple metrics (e.g., accuracy, AUC, LLM-J.) while balancing computational costs, pushing a new paradigm shift towards interpretable, adaptive, and multimodal medical agentic systems. |
| 2025-08-14 | [A Structured Framework for Prioritizing Unsafe Control Actions in STPA: Case Study on eVTOL Operations](http://arxiv.org/abs/2508.10446v1) | Halima El Badaoui | Systems Theoretic Process Analysis (STPA) is a widely recommended method for analysing complex system safety. STPA can identify numerous Unsafe Control Actions (UCAs) and requirements depending on the level of granularity of the analysis and the complexity of the system being analysed. Managing numerous results is challenging, especially during a fast-paced development lifecycle. Extensive research has been done to optimize the efficiency of managing and prioritising the STPA results. However, maintaining the objectivity of prioritisation and communicating the prioritised results have become common challenges. In this paper, the authors present a complementary approach that incorporates inputs from both the safety analysts and domain experts to more objectively prioritise UCAs. This is done by evaluating the severity of each UCA, the impact factor of each controller or decision maker that issues the UCA, and the ranking provided by the subject matter experts who assess the UCA criticalities based on different factors. In addition, a Monte Carlo simulation is introduced to reduce subjectivity and relativity, thus enabling more objective prioritisation of the UCAs. As part of the approach to better communicate the prioritisation results and plan the next steps of system development, a dynamic-scaling prioritisation matrix was developed to capture different sets of prioritised UCAs. The approach was applied to a real project to improve the safe operations of Electric Vertical Take-off and Landing (eVTOL). The results highlighted critical UCAs that need to be prioritised for safer eVTOL operation. 318 UCAs were identified in total. Based on the application of the prioritisation methodology, 110 were recognized as high-priority UCAs to strengthen the system design. |
| 2025-08-14 | [STRIDE-QA: Visual Question Answering Dataset for Spatiotemporal Reasoning in Urban Driving Scenes](http://arxiv.org/abs/2508.10427v1) | Keishi Ishihara, Kento Sasaki et al. | Vision-Language Models (VLMs) have been applied to autonomous driving to support decision-making in complex real-world scenarios. However, their training on static, web-sourced image-text pairs fundamentally limits the precise spatiotemporal reasoning required to understand and predict dynamic traffic scenes. We address this critical gap with STRIDE-QA, a large-scale visual question answering (VQA) dataset for physically grounded reasoning from an ego-centric perspective. Constructed from 100 hours of multi-sensor driving data in Tokyo, capturing diverse and challenging conditions, STRIDE-QA is the largest VQA dataset for spatiotemporal reasoning in urban driving, offering 16 million QA pairs over 285K frames. Grounded by dense, automatically generated annotations including 3D bounding boxes, segmentation masks, and multi-object tracks, the dataset uniquely supports both object-centric and ego-centric reasoning through three novel QA tasks that require spatial localization and temporal prediction. Our benchmarks demonstrate that existing VLMs struggle significantly, achieving near-zero scores on prediction consistency. In contrast, VLMs fine-tuned on STRIDE-QA exhibit dramatic performance gains, achieving 55% success in spatial localization and 28% consistency in future motion prediction, compared to near-zero scores from general-purpose VLMs. Therefore, STRIDE-QA establishes a comprehensive foundation for developing more reliable VLMs for safety-critical autonomous systems. |
| 2025-08-14 | [Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation](http://arxiv.org/abs/2508.10404v1) | Huizhen Shu, Xuying Li et al. | With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP systems.However, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated. |
| 2025-08-14 | [PQ-DAF: Pose-driven Quality-controlled Data Augmentation for Data-scarce Driver Distraction Detection](http://arxiv.org/abs/2508.10397v1) | Haibin Sun, Xinghui Song | Driver distraction detection is essential for improving traffic safety and reducing road accidents. However, existing models often suffer from degraded generalization when deployed in real-world scenarios. This limitation primarily arises from the few-shot learning challenge caused by the high cost of data annotation in practical environments, as well as the substantial domain shift between training datasets and target deployment conditions. To address these issues, we propose a Pose-driven Quality-controlled Data Augmentation Framework (PQ-DAF) that leverages a vision-language model for sample filtering to cost-effectively expand training data and enhance cross-domain robustness. Specifically, we employ a Progressive Conditional Diffusion Model (PCDMs) to accurately capture key driver pose features and synthesize diverse training examples. A sample quality assessment module, built upon the CogVLM vision-language model, is then introduced to filter out low-quality synthetic samples based on a confidence threshold, ensuring the reliability of the augmented dataset. Extensive experiments demonstrate that PQ-DAF substantially improves performance in few-shot driver distraction detection, achieving significant gains in model generalization under data-scarce conditions. |
| 2025-08-14 | [Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts](http://arxiv.org/abs/2508.10390v1) | Chiyu Zhang, Lu Zhou et al. | Evaluating jailbreak attacks is challenging when prompts are not overtly harmful or fail to induce harmful outputs. Unfortunately, many existing red-teaming datasets contain such unsuitable prompts. To evaluate attacks accurately, these datasets need to be assessed and cleaned for maliciousness. However, existing malicious content detection methods rely on either manual annotation, which is labor-intensive, or large language models (LLMs), which have inconsistent accuracy in harmful types. To balance accuracy and efficiency, we propose a hybrid evaluation framework named MDH (Malicious content Detection based on LLMs with Human assistance) that combines LLM-based annotation with minimal human oversight, and apply it to dataset cleaning and detection of jailbroken responses. Furthermore, we find that well-crafted developer messages can significantly boost jailbreak success, leading us to propose two new strategies: D-Attack, which leverages context simulation, and DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets, judgements, and detection results will be released in github repository: https://github.com/AlienZhang1996/DH-CoT. |
| 2025-08-14 | [A Semantic-Aware Framework for Safe and Intent-Integrative Assistance in Upper-Limb Exoskeletons](http://arxiv.org/abs/2508.10378v1) | Yu Chen, Shu Miao et al. | Upper-limb exoskeletons are primarily designed to provide assistive support by accurately interpreting and responding to human intentions. In home-care scenarios, exoskeletons are expected to adapt their assistive configurations based on the semantic information of the task, adjusting appropriately in accordance with the nature of the object being manipulated. However, existing solutions often lack the ability to understand task semantics or collaboratively plan actions with the user, limiting their generalizability. To address this challenge, this paper introduces a semantic-aware framework that integrates large language models into the task planning framework, enabling the delivery of safe and intent-integrative assistance. The proposed approach begins with the exoskeleton operating in transparent mode to capture the wearer's intent during object grasping. Once semantic information is extracted from the task description, the system automatically configures appropriate assistive parameters. In addition, a diffusion-based anomaly detector is used to continuously monitor the state of human-robot interaction and trigger real-time replanning in response to detected anomalies. During task execution, online trajectory refinement and impedance control are used to ensure safety and regulate human-robot interaction. Experimental results demonstrate that the proposed method effectively aligns with the wearer's cognition, adapts to semantically varying tasks, and responds reliably to anomalies. |
| 2025-08-14 | [A Curriculum Learning Approach to Reinforcement Learning: Leveraging RAG for Multimodal Question Answering](http://arxiv.org/abs/2508.10337v1) | Chenliang Zhang, Lin Wang et al. | This paper describes the solutions of the Dianping-Trust-Safety team for the META CRAG-MM challenge. The challenge requires building a comprehensive retrieval-augmented generation system capable for multi-modal multi-turn question answering. The competition consists of three tasks: (1) answering questions using structured data retrieved from an image-based mock knowledge graph, (2) synthesizing information from both knowledge graphs and web search results, and (3) handling multi-turn conversations that require context understanding and information aggregation from multiple sources. For Task 1, our solution is based on the vision large language model, enhanced by supervised fine-tuning with knowledge distilled from GPT-4.1. We further applied curriculum learning strategies to guide reinforcement learning, resulting in improved answer accuracy and reduced hallucination. For Task 2 and Task 3, we additionally leveraged web search APIs to incorporate external knowledge, enabling the system to better handle complex queries and multi-turn conversations. Our approach achieved 1st place in Task 1 with a significant lead of 52.38\%, and 3rd place in Task 3, demonstrating the effectiveness of the integration of curriculum learning with reinforcement learning in our training pipeline. |
| 2025-08-14 | [From Pixel to Mask: A Survey of Out-of-Distribution Segmentation](http://arxiv.org/abs/2508.10309v1) | Wenjie Zhao, Jia Li et al. | Out-of-distribution (OoD) detection and segmentation have attracted growing attention as concerns about AI security rise. Conventional OoD detection methods identify the existence of OoD objects but lack spatial localization, limiting their usefulness in downstream tasks. OoD segmentation addresses this limitation by localizing anomalous objects at pixel-level granularity. This capability is crucial for safety-critical applications such as autonomous driving, where perception modules must not only detect but also precisely segment OoD objects, enabling targeted control actions and enhancing overall system robustness. In this survey, we group current OoD segmentation approaches into four categories: (i) test-time OoD segmentation, (ii) outlier exposure for supervised training, (iii) reconstruction-based methods, (iv) and approaches that leverage powerful models. We systematically review recent advances in OoD segmentation for autonomous-driving scenarios, identify emerging challenges, and discuss promising future research directions. |

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



