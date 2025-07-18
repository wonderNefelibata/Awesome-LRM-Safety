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
| 2025-07-17 | [FormulaOne: Measuring the Depth of Algorithmic Reasoning Beyond Competitive Programming](http://arxiv.org/abs/2507.13337v1) | Gal Beniamini, Yuval Dor et al. | Frontier AI models demonstrate formidable breadth of knowledge. But how close are they to true human -- or superhuman -- expertise? Genuine experts can tackle the hardest problems and push the boundaries of scientific understanding. To illuminate the limits of frontier model capabilities, we turn away from contrived competitive programming puzzles, and instead focus on real-life research problems.   We construct FormulaOne, a benchmark that lies at the intersection of graph theory, logic, and algorithms, all well within the training distribution of frontier models. Our problems are incredibly demanding, requiring an array of reasoning steps. The dataset has three key properties. First, it is of commercial interest and relates to practical large-scale optimisation problems, such as those arising in routing, scheduling, and network design. Second, it is generated from the highly expressive framework of Monadic Second-Order (MSO) logic on graphs, paving the way toward automatic problem generation at scale; ideal for building RL environments. Third, many of our problems are intimately related to the frontier of theoretical computer science, and to central conjectures therein, such as the Strong Exponential Time Hypothesis (SETH). As such, any significant algorithmic progress on our dataset, beyond known results, could carry profound theoretical implications.   Remarkably, state-of-the-art models like OpenAI's o3 fail entirely on FormulaOne, solving less than 1% of the questions, even when given 10 attempts and explanatory fewshot examples -- highlighting how far they remain from expert-level understanding in some domains. To support further research, we additionally curate FormulaOne-Warmup, offering a set of simpler tasks, from the same distribution. We release the full corpus along with a comprehensive evaluation framework. |
| 2025-07-17 | [The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner](http://arxiv.org/abs/2507.13332v1) | Zhouqi Hua, Wenwei Zhang et al. | Length generalization, the ability to solve problems of longer sequences than those observed during training, poses a core challenge of Transformer-based large language models (LLM). Although existing studies have predominantly focused on data-driven approaches for arithmetic operations and symbolic manipulation tasks, these approaches tend to be task-specific with limited overall performance. To pursue a more general solution, this paper focuses on a broader case of reasoning problems that are computable, i.e., problems that algorithms can solve, thus can be solved by the Turing Machine. From this perspective, this paper proposes Turing MAchine Imitation Learning (TAIL) to improve the length generalization ability of LLMs. TAIL synthesizes chain-of-thoughts (CoT) data that imitate the execution process of a Turing Machine by computer programs, which linearly expands the reasoning steps into atomic states to alleviate shortcut learning and explicit memory fetch mechanism to reduce the difficulties of dynamic and long-range data access in elementary operations. To validate the reliability and universality of TAIL, we construct a challenging synthetic dataset covering 8 classes of algorithms and 18 tasks. Without bells and whistles, TAIL significantly improves the length generalization ability as well as the performance of Qwen2.5-7B on various tasks using only synthetic data, surpassing previous methods and DeepSeek-R1. The experimental results reveal that the key concepts in the Turing Machine, instead of the thinking styles, are indispensable for TAIL for length generalization, through which the model exhibits read-and-write behaviors consistent with the properties of the Turing Machine in their attention layers. This work provides a promising direction for future research in the learning of LLM reasoning from synthetic data. |
| 2025-07-17 | [AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](http://arxiv.org/abs/2507.13300v1) | Yilun Zhao, Weiyuan Chen et al. | We introduce AbGen, the first benchmark designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. AbGen consists of 1,500 expert-annotated examples derived from 807 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. Our evaluation of leading LLMs, such as DeepSeek-R1-0528 and o4-mini, highlights a significant performance gap between these models and human experts in terms of the importance, faithfulness, and soundness of the ablation study designs. Moreover, we demonstrate that current automated evaluation methods are not reliable for our task, as they show a significant discrepancy when compared to human assessment. To better investigate this, we develop AbGen-Eval, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task. We investigate various LLM-as-Judge systems on AbGen-Eval, providing insights for future research on developing more effective and reliable LLM-based evaluation systems for complex scientific tasks. |
| 2025-07-17 | [Automating Steering for Safe Multimodal Large Language Models](http://arxiv.org/abs/2507.13255v1) | Lyucheng Wu, Mengru Wang et al. | Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems. |
| 2025-07-17 | [Channel-wise Motion Features for Efficient Motion Segmentation](http://arxiv.org/abs/2507.13082v1) | Riku Inoue, Masamitsu Tsuchiya et al. | For safety-critical robotics applications such as autonomous driving, it is important to detect all required objects accurately in real-time. Motion segmentation offers a solution by identifying dynamic objects from the scene in a class-agnostic manner. Recently, various motion segmentation models have been proposed, most of which jointly use subnetworks to estimate Depth, Pose, Optical Flow, and Scene Flow. As a result, the overall computational cost of the model increases, hindering real-time performance.   In this paper, we propose a novel cost-volume-based motion feature representation, Channel-wise Motion Features. By extracting depth features of each instance in the feature map and capturing the scene's 3D motion information, it offers enhanced efficiency. The only subnetwork used to build Channel-wise Motion Features is the Pose Network, and no others are required. Our method not only achieves about 4 times the FPS of state-of-the-art models in the KITTI Dataset and Cityscapes of the VCAS-Motion Dataset, but also demonstrates equivalent accuracy while reducing the parameters to about 25$\%$. |
| 2025-07-17 | [Dual LiDAR-Based Traffic Movement Count Estimation at a Signalized Intersection: Deployment, Data Collection, and Preliminary Analysis](http://arxiv.org/abs/2507.13073v1) | Saswat Priyadarshi Nayak, Guoyuan Wu et al. | Traffic Movement Count (TMC) at intersections is crucial for optimizing signal timings, assessing the performance of existing traffic control measures, and proposing efficient lane configurations to minimize delays, reduce congestion, and promote safety. Traditionally, methods such as manual counting, loop detectors, pneumatic road tubes, and camera-based recognition have been used for TMC estimation. Although generally reliable, camera-based TMC estimation is prone to inaccuracies under poor lighting conditions during harsh weather and nighttime. In contrast, Light Detection and Ranging (LiDAR) technology is gaining popularity in recent times due to reduced costs and its expanding use in 3D object detection, tracking, and related applications. This paper presents the authors' endeavor to develop, deploy and evaluate a dual-LiDAR system at an intersection in the city of Rialto, California, for TMC estimation. The 3D bounding box detections from the two LiDARs are used to classify vehicle counts based on traffic directions, vehicle movements, and vehicle classes. This work discusses the estimated TMC results and provides insights into the observed trends and irregularities. Potential improvements are also discussed that could enhance not only TMC estimation, but also trajectory forecasting and intent prediction at intersections. |
| 2025-07-17 | [History-dependent and frequency-dependent dielectric nonlinearities induced by polar nanoregions in a thin film of 0.5 ( Ba 0.7 Ca 0.3 TiO 3 ) -- 0.5 ( BaZr 0.2 Ti 0.8 O 3 )](http://arxiv.org/abs/2507.13021v1) | Kevin Nadaud, Guillaume Nataf et al. | In this article, dielectric nonlinearities in 0.5(Ba0.7 Ca0.3 TiO3 ) -- 0.5(BaZr 0.2 Ti0.8 O3 ) thin film are studied using impedance spectroscopy and harmonic measurements, as a function of the AC measuring field, at different frequencies and upon cycling. The measurements reveal that the pinching of the hysteresis loop, characterized by a phase angle of the third harmonic close to --270 deg, is stronger for low frequencies. This confirms that the pinching is induced by the presence of polar nanoregions (PNRs), whose responses are also strongly dependent on frequency. When repeating the measurement, the PNR contribution changes since the phase angle of the third-harmonic response evolves from pinched to conventional relaxor. This shows that strong changes in the PNR configuration can be induced, even for low AC fields. First-order reversal curves (FORCs) confirm the presence of a pinched hysteresis loop. When repeating the FORC measurement a second time, the distribution drastically changes and corresponds to a soft ferroelectric. The asymmetry of the Preisach plane measured using FORCs is confirmed by a proposed measurement strategy: unipolar impedance measurements. |
| 2025-07-17 | [Bridging Boundaries: How to Foster Effective Research Collaborations Across Affiliations in the Field of Trust and Safety](http://arxiv.org/abs/2507.13008v1) | Amanda Menking, Mona Elswah et al. | As the field of Trust and Safety in digital spaces continues to grow, it has become increasingly necessary - but also increasingly complex - to collaborate on research across the academic, industry, governmental and non-governmental sectors. This paper examines how cross-affiliation research partnerships can be structured to overcome misaligned incentives, timelines and constraints while delivering on the unique strengths of each stakeholder. Drawing on our own experience of cross-sector collaboration, we define the main types of affiliation and highlight the common differences in research priorities, operational pressures and evaluation metrics across sectors. We then propose a practical, step-by-step framework for initiating and managing effective collaborations, including strategies for building trust, aligning goals, and distributing roles. We emphasize the critical yet often invisible work of articulation and argue that cross-sector partnerships are essential for developing more ethical, equitable and impactful research in trust and safety. Ultimately, we advocate collaborative models that prioritize inclusivity, transparency and real-world relevance in order to meet the interdisciplinary demands of this emerging field. |
| 2025-07-17 | [Robustness Requirement Coverage using a Situation Coverage Approach for Vision-based AI Systems](http://arxiv.org/abs/2507.12986v1) | Sepeedeh Shahbeigi, Nawshin Mannan Proma et al. | AI-based robots and vehicles are expected to operate safely in complex and dynamic environments, even in the presence of component degradation. In such systems, perception relies on sensors such as cameras to capture environmental data, which is then processed by AI models to support decision-making. However, degradation in sensor performance directly impacts input data quality and can impair AI inference. Specifying safety requirements for all possible sensor degradation scenarios leads to unmanageable complexity and inevitable gaps. In this position paper, we present a novel framework that integrates camera noise factor identification with situation coverage analysis to systematically elicit robustness-related safety requirements for AI-based perception systems. We focus specifically on camera degradation in the automotive domain. Building on an existing framework for identifying degradation modes, we propose involving domain, sensor, and safety experts, and incorporating Operational Design Domain specifications to extend the degradation model by incorporating noise factors relevant to AI performance. Situation coverage analysis is then applied to identify representative operational contexts. This work marks an initial step toward integrating noise factor analysis and situational coverage to support principled formulation and completeness assessment of robustness requirements for camera-based AI perception. |
| 2025-07-17 | [Non-differentiable Reward Optimization for Diffusion-based Autonomous Motion Planning](http://arxiv.org/abs/2507.12977v1) | Giwon Lee, Daehee Park et al. | Safe and effective motion planning is crucial for autonomous robots. Diffusion models excel at capturing complex agent interactions, a fundamental aspect of decision-making in dynamic environments. Recent studies have successfully applied diffusion models to motion planning, demonstrating their competence in handling complex scenarios and accurately predicting multi-modal future trajectories. Despite their effectiveness, diffusion models have limitations in training objectives, as they approximate data distributions rather than explicitly capturing the underlying decision-making dynamics. However, the crux of motion planning lies in non-differentiable downstream objectives, such as safety (collision avoidance) and effectiveness (goal-reaching), which conventional learning algorithms cannot directly optimize. In this paper, we propose a reinforcement learning-based training scheme for diffusion motion planning models, enabling them to effectively learn non-differentiable objectives that explicitly measure safety and effectiveness. Specifically, we introduce a reward-weighted dynamic thresholding algorithm to shape a dense reward signal, facilitating more effective training and outperforming models trained with differentiable objectives. State-of-the-art performance on pedestrian datasets (CrowdNav, ETH-UCY) compared to various baselines demonstrates the versatility of our approach for safe and effective motion planning. |
| 2025-07-17 | [Insights into a radiology-specialised multimodal large language model with sparse autoencoders](http://arxiv.org/abs/2507.12950v1) | Kenza Bouzid, Shruthi Bannur et al. | Interpretability can improve the safety, transparency and trust of AI models, which is especially important in healthcare applications where decisions often carry significant consequences. Mechanistic interpretability, particularly through the use of sparse autoencoders (SAEs), offers a promising approach for uncovering human-interpretable features within large transformer-based models. In this study, we apply Matryoshka-SAE to the radiology-specialised multimodal large language model, MAIRA-2, to interpret its internal representations. Using large-scale automated interpretability of the SAE features, we identify a range of clinically relevant concepts - including medical devices (e.g., line and tube placements, pacemaker presence), pathologies such as pleural effusion and cardiomegaly, longitudinal changes and textual features. We further examine the influence of these features on model behaviour through steering, demonstrating directional control over generations with mixed success. Our results reveal practical and methodological challenges, yet they offer initial insights into the internal concepts learned by MAIRA-2 - marking a step toward deeper mechanistic understanding and interpretability of a radiology-adapted multimodal large language model, and paving the way for improved model transparency. We release the trained SAEs and interpretations: https://huggingface.co/microsoft/maira-2-sae. |
| 2025-07-17 | [LanePerf: a Performance Estimation Framework for Lane Detection](http://arxiv.org/abs/2507.12894v1) | Yin Wu, Daniel Slieter et al. | Lane detection is a critical component of Advanced Driver-Assistance Systems (ADAS) and Automated Driving System (ADS), providing essential spatial information for lateral control. However, domain shifts often undermine model reliability when deployed in new environments. Ensuring the robustness and safety of lane detection models typically requires collecting and annotating target domain data, which is resource-intensive. Estimating model performance without ground-truth labels offers a promising alternative for efficient robustness assessment, yet remains underexplored in lane detection. While previous work has addressed performance estimation in image classification, these methods are not directly applicable to lane detection tasks. This paper first adapts five well-performing performance estimation methods from image classification to lane detection, building a baseline. Addressing the limitations of prior approaches that solely rely on softmax scores or lane features, we further propose a new Lane Performance Estimation Framework (LanePerf), which integrates image and lane features using a pretrained image encoder and a DeepSets-based architecture, effectively handling zero-lane detection scenarios and large domain-shift cases. Extensive experiments on the OpenLane dataset, covering diverse domain shifts (scenes, weather, hours), demonstrate that our LanePerf outperforms all baselines, achieving a lower MAE of 0.117 and a higher Spearman's rank correlation coefficient of 0.727. These findings pave the way for robust, label-free performance estimation in ADAS, supporting more efficient testing and improved safety in challenging driving scenarios. |
| 2025-07-17 | [Manipulation Attacks by Misaligned AI: Risk Analysis and Safety Case Framework](http://arxiv.org/abs/2507.12872v1) | Rishane Dassanayake, Mario Demetroudi et al. | Frontier AI systems are rapidly advancing in their capabilities to persuade, deceive, and influence human behaviour, with current models already demonstrating human-level persuasion and strategic deception in specific contexts. Humans are often the weakest link in cybersecurity systems, and a misaligned AI system deployed internally within a frontier company may seek to undermine human oversight by manipulating employees. Despite this growing threat, manipulation attacks have received little attention, and no systematic framework exists for assessing and mitigating these risks. To address this, we provide a detailed explanation of why manipulation attacks are a significant threat and could lead to catastrophic outcomes. Additionally, we present a safety case framework for manipulation risk, structured around three core lines of argument: inability, control, and trustworthiness. For each argument, we specify evidence requirements, evaluation methodologies, and implementation considerations for direct application by AI companies. This paper provides the first systematic methodology for integrating manipulation risk into AI safety governance, offering AI companies a concrete foundation to assess and mitigate these threats before deployment. |
| 2025-07-17 | [AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation](http://arxiv.org/abs/2507.12768v1) | Hengkai Tan, Yao Feng et al. | Vision-language-action (VLA) models have shown promise on task-conditioned control in complex settings such as bimanual manipulation. However, the heavy reliance on task-specific human demonstrations limits their generalization and incurs high data acquisition costs. In this work, we present a new notion of task-agnostic action paradigm that decouples action execution from task-specific conditioning, enhancing scalability, efficiency, and cost-effectiveness. To address the data collection challenges posed by this paradigm -- such as low coverage density, behavioral redundancy, and safety risks -- we introduce ATARA (Automated Task-Agnostic Random Actions), a scalable self-supervised framework that accelerates collection by over $ 30\times $ compared to human teleoperation. To further enable effective learning from task-agnostic data, which often suffers from distribution mismatch and irrelevant trajectories, we propose AnyPos, an inverse dynamics model equipped with Arm-Decoupled Estimation and a Direction-Aware Decoder (DAD). We additionally integrate a video-conditioned action validation module to verify the feasibility of learned policies across diverse manipulation tasks. Extensive experiments show that the AnyPos-ATARA pipeline yields a 51% improvement in test accuracy and achieves 30-40% higher success rates in downstream tasks such as lifting, pick-and-place, and clicking, using replay-based video validation. Project Page: https://embodiedfoundation.github.io/vidar_anypos |
| 2025-07-17 | [World Model-Based End-to-End Scene Generation for Accident Anticipation in Autonomous Driving](http://arxiv.org/abs/2507.12762v1) | Yanchen Guan, Haicheng Liao et al. | Reliable anticipation of traffic accidents is essential for advancing autonomous driving systems. However, this objective is limited by two fundamental challenges: the scarcity of diverse, high-quality training data and the frequent absence of crucial object-level cues due to environmental disruptions or sensor deficiencies. To tackle these issues, we propose a comprehensive framework combining generative scene augmentation with adaptive temporal reasoning. Specifically, we develop a video generation pipeline that utilizes a world model guided by domain-informed prompts to create high-resolution, statistically consistent driving scenarios, particularly enriching the coverage of edge cases and complex interactions. In parallel, we construct a dynamic prediction model that encodes spatio-temporal relationships through strengthened graph convolutions and dilated temporal operators, effectively addressing data incompleteness and transient visual noise. Furthermore, we release a new benchmark dataset designed to better capture diverse real-world driving risks. Extensive experiments on public and newly released datasets confirm that our framework enhances both the accuracy and lead time of accident anticipation, offering a robust solution to current data and modeling limitations in safety-critical autonomous driving applications. |
| 2025-07-17 | [Invariance Guarantees using Continuously Parametrized Control Barrier Functions](http://arxiv.org/abs/2507.12743v1) | Inkyu Jang, H. Jin Kim | Constructing a control invariant set with an appropriate shape that fits within a given state constraint is a fundamental problem in safety-critical control but is known to be difficult, especially for large or complex spaces. This paper introduces a safe control framework of utilizing PCBF: continuously parametrized control barrier functions (CBFs). In PCBF, each choice of parameter corresponds to a control invariant set of relatively simple shape. Invariance-preserving control is done by dynamically selecting a parameter whose corresponding invariant set lies within the safety bound. This eliminates the need for synthesizing a single complex CBF that matches the entire free space. It also enables easier adaptation to diverse environments. By assigning a differentiable dynamics on the parameter space, we derive a lightweight feedback controller based on quadratic programming (QP), namely PCBF-QP. We also discuss on how to build a valid PCBF for a class of systems and how to constrain the parameter so that the invariant set does not exceed the safety bound. The concept is also extended to cover continuously parametrized high-order CBFs, which is called high-order PCBF. Finally, simulation experiments are conducted to validate the proposed approach. |
| 2025-07-17 | [On the Properties of Optimal-Decay Control Barrier Functions](http://arxiv.org/abs/2507.12717v1) | Pio Ong, Max H. Cohen et al. | Control barrier functions provide a powerful means for synthesizing safety filters that ensure safety framed as forward set invariance. Key to CBFs' effectiveness is the simple inequality on the system dynamics: $\dot{h} \geq - \alpha(h)$. Yet determining the class $\mathcal{K}^e$ function $\alpha$ is a user defined choice that can have a dramatic effect on the resulting system behavior. This paper formalizes the process of choosing $\alpha$ using optimal-decay control barrier functions (OD-CBFs). These modify the traditional CBF inequality to: $\dot{h} \geq - \omega \alpha(h)$, where $\omega \geq 0$ is automatically determined by the safety filter. A comprehensive characterization of this framework is elaborated, including tractable conditions on OD-CBF validity, control invariance of the underlying sets in the state space, forward invariance conditions for safe sets, and discussion on optimization-based safe controllers in terms of their feasibility, Lipschitz continuity, and closed-form expressions. The framework also extends existing higher-order CBF techniques, addressing safety constraints with vanishing relative degrees. The proposed method is demonstrated on a satellite control problem in simulation. |
| 2025-07-16 | [Reasoning-Finetuning Repurposes Latent Representations in Base Models](http://arxiv.org/abs/2507.12638v1) | Jake Ward, Chuqiao Lin et al. | Backtracking, an emergent behavior elicited by reasoning fine-tuning, has been shown to be a key mechanism in reasoning models' enhanced capabilities. Prior work has succeeded in manipulating this behavior via steering vectors, but the underlying mechanism remains poorly understood. In this work, we show that the emergence of backtracking in DeepSeek-R1-Distill-Llama-8B is in part driven by a repurposed direction already present in base model activations. Specifically, we identify a direction in base Llama-3.1-8B's residual stream which systematically induces backtracking when used to steer the distilled reasoning model, and find that the effects of steering with this direction cannot be trivially explained by token-level attributes. We further find that this direction does not induce backtracking in the base model, suggesting that the reasoning finetuning process repurposes pre-existing representations to form new behavioral circuits. Additionally, we hypothesize that this direction is one of several which may work together to mediate backtracking. Our findings offer a compelling picture that reasoning-finetuned models repurpose pre-existing base model representations, rather than learn new capabilities from scratch. |
| 2025-07-16 | [Safeguarding Federated Learning-based Road Condition Classification](http://arxiv.org/abs/2507.12568v1) | Sheng Liu, Panos Papadimitratos | Federated Learning (FL) has emerged as a promising solution for privacy-preserving autonomous driving, specifically camera-based Road Condition Classification (RCC) systems, harnessing distributed sensing, computing, and communication resources on board vehicles without sharing sensitive image data. However, the collaborative nature of FL-RCC frameworks introduces new vulnerabilities: Targeted Label Flipping Attacks (TLFAs), in which malicious clients (vehicles) deliberately alter their training data labels to compromise the learned model inference performance. Such attacks can, e.g., cause a vehicle to mis-classify slippery, dangerous road conditions as pristine and exceed recommended speed. However, TLFAs for FL-based RCC systems are largely missing. We address this challenge with a threefold contribution: 1) we disclose the vulnerability of existing FL-RCC systems to TLFAs; 2) we introduce a novel label-distance-based metric to precisely quantify the safety risks posed by TLFAs; and 3) we propose FLARE, a defensive mechanism leveraging neuron-wise analysis of the output layer to mitigate TLFA effects. Extensive experiments across three RCC tasks, four evaluation metrics, six baselines, and three deep learning models demonstrate both the severity of TLFAs on FL-RCC systems and the effectiveness of FLARE in mitigating the attack impact. |
| 2025-07-16 | [MMHU: A Massive-Scale Multimodal Benchmark for Human Behavior Understanding](http://arxiv.org/abs/2507.12463v1) | Renjie Li, Ruijie Ye et al. | Humans are integral components of the transportation ecosystem, and understanding their behaviors is crucial to facilitating the development of safe driving systems. Although recent progress has explored various aspects of human behavior$\unicode{x2014}$such as motion, trajectories, and intention$\unicode{x2014}$a comprehensive benchmark for evaluating human behavior understanding in autonomous driving remains unavailable. In this work, we propose $\textbf{MMHU}$, a large-scale benchmark for human behavior analysis featuring rich annotations, such as human motion and trajectories, text description for human motions, human intention, and critical behavior labels relevant to driving safety. Our dataset encompasses 57k human motion clips and 1.73M frames gathered from diverse sources, including established driving datasets such as Waymo, in-the-wild videos from YouTube, and self-collected data. A human-in-the-loop annotation pipeline is developed to generate rich behavior captions. We provide a thorough dataset analysis and benchmark multiple tasks$\unicode{x2014}$ranging from motion prediction to motion generation and human behavior question answering$\unicode{x2014}$thereby offering a broad evaluation suite. Project page : https://MMHU-Benchmark.github.io. |

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



