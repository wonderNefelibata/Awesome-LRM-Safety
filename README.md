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
| 2025-09-23 | [Residual Off-Policy RL for Finetuning Behavior Cloning Policies](http://arxiv.org/abs/2509.19301v1) | Lars Ankile, Zhenyu Jiang et al. | Recent advances in behavior cloning (BC) have enabled impressive visuomotor control policies. However, these approaches are limited by the quality of human demonstrations, the manual effort required for data collection, and the diminishing returns from increasing offline data. In comparison, reinforcement learning (RL) trains an agent through autonomous interaction with the environment and has shown remarkable success in various domains. Still, training RL policies directly on real-world robots remains challenging due to sample inefficiency, safety concerns, and the difficulty of learning from sparse rewards for long-horizon tasks, especially for high-degree-of-freedom (DoF) systems. We present a recipe that combines the benefits of BC and RL through a residual learning framework. Our approach leverages BC policies as black-box bases and learns lightweight per-step residual corrections via sample-efficient off-policy RL. We demonstrate that our method requires only sparse binary reward signals and can effectively improve manipulation policies on high-degree-of-freedom (DoF) systems in both simulation and the real world. In particular, we demonstrate, to the best of our knowledge, the first successful real-world RL training on a humanoid robot with dexterous hands. Our results demonstrate state-of-the-art performance in various vision-based tasks, pointing towards a practical pathway for deploying RL in the real world. Project website: https://residual-offpolicy-rl.github.io |
| 2025-09-23 | [SOE: Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration](http://arxiv.org/abs/2509.19292v1) | Yang Jin, Jun Lv et al. | Intelligent agents progress by continually refining their capabilities through actively exploring environments. Yet robot policies often lack sufficient exploration capability due to action mode collapse. Existing methods that encourage exploration typically rely on random perturbations, which are unsafe and induce unstable, erratic behaviors, thereby limiting their effectiveness. We propose Self-Improvement via On-Manifold Exploration (SOE), a framework that enhances policy exploration and improvement in robotic manipulation. SOE learns a compact latent representation of task-relevant factors and constrains exploration to the manifold of valid actions, ensuring safety, diversity, and effectiveness. It can be seamlessly integrated with arbitrary policy models as a plug-in module, augmenting exploration without degrading the base policy performance. Moreover, the structured latent space enables human-guided exploration, further improving efficiency and controllability. Extensive experiments in both simulation and real-world tasks demonstrate that SOE consistently outperforms prior methods, achieving higher task success rates, smoother and safer exploration, and superior sample efficiency. These results establish on-manifold exploration as a principled approach to sample-efficient policy self-improvement. Project website: https://ericjin2002.github.io/SOE |
| 2025-09-23 | [MsFIN: Multi-scale Feature Interaction Network for Traffic Accident Anticipation](http://arxiv.org/abs/2509.19227v1) | Tongshuai Wu, Chao Lu et al. | With the widespread deployment of dashcams and advancements in computer vision, developing accident prediction models from the dashcam perspective has become critical for proactive safety interventions. However, two key challenges persist: modeling feature-level interactions among traffic participants (often occluded in dashcam views) and capturing complex, asynchronous multi-temporal behavioral cues preceding accidents. To deal with these two challenges, a Multi-scale Feature Interaction Network (MsFIN) is proposed for early-stage accident anticipation from dashcam videos. MsFIN has three layers for multi-scale feature aggregation, temporal feature processing and multi-scale feature post fusion, respectively. For multi-scale feature aggregation, a Multi-scale Module is designed to extract scene representations at short-term, mid-term and long-term temporal scales. Meanwhile, the Transformer architecture is leveraged to facilitate comprehensive feature interactions. Temporal feature processing captures the sequential evolution of scene and object features under causal constraints. In the multi-scale feature post fusion stage, the network fuses scene and object features across multiple temporal scales to generate a comprehensive risk representation. Experiments on DAD and DADA datasets show that MsFIN significantly outperforms state-of-the-art models with single-scale feature extraction in both prediction correctness and earliness. Ablation studies validate the effectiveness of each module in MsFIN, highlighting how the network achieves superior performance through multi-scale feature fusion and contextual interaction modeling. |
| 2025-09-23 | [Steering Multimodal Large Language Models Decoding for Context-Aware Safety](http://arxiv.org/abs/2509.19212v1) | Zheyuan Liu, Zhangchen Xu et al. | Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet their ability to make context-aware safety decisions remains limited. Existing methods often fail to balance oversensitivity (unjustified refusals of benign queries) and undersensitivity (missed detection of visually grounded risks), leaving a persistent gap in safety alignment. To address this issue, we introduce Safety-aware Contrastive Decoding (SafeCoDe), a lightweight and model-agnostic decoding framework that dynamically adjusts token generation based on multimodal context. SafeCoDe operates in two stages: (1) a contrastive decoding mechanism that highlights tokens sensitive to visual context by contrasting real and Gaussian-noised images, and (2) a global-aware token modulation strategy that integrates scene-level reasoning with token-level adjustment to adapt refusals according to the predicted safety verdict. Extensive experiments across diverse MLLM architectures and safety benchmarks, covering undersensitivity, oversensitivity, and general safety evaluations, show that SafeCoDe consistently improves context-sensitive refusal behaviors while preserving model helpfulness. |
| 2025-09-23 | [Algorithms for Adversarially Robust Deep Learning](http://arxiv.org/abs/2509.19100v1) | Alexander Robey | Given the widespread use of deep learning models in safety-critical applications, ensuring that the decisions of such models are robust against adversarial exploitation is of fundamental importance. In this thesis, we discuss recent progress toward designing algorithms that exhibit desirable robustness properties. First, we discuss the problem of adversarial examples in computer vision, for which we introduce new technical results, training paradigms, and certification algorithms. Next, we consider the problem of domain generalization, wherein the task is to train neural networks to generalize from a family of training distributions to unseen test distributions. We present new algorithms that achieve state-of-the-art generalization in medical imaging, molecular identification, and image classification. Finally, we study the setting of jailbreaking large language models (LLMs), wherein an adversarial user attempts to design prompts that elicit objectionable content from an LLM. We propose new attacks and defenses, which represent the frontier of progress toward designing robust language-based agents. |
| 2025-09-23 | [Investigating Traffic Accident Detection Using Multimodal Large Language Models](http://arxiv.org/abs/2509.19096v1) | Ilhan Skender, Kailin Tong et al. | Traffic safety remains a critical global concern, with timely and accurate accident detection essential for hazard reduction and rapid emergency response. Infrastructure-based vision sensors offer scalable and efficient solutions for continuous real-time monitoring, facilitating automated detection of acci- dents directly from captured images. This research investigates the zero-shot capabilities of multimodal large language models (MLLMs) for detecting and describing traffic accidents using images from infrastructure cameras, thus minimizing reliance on extensive labeled datasets. Main contributions include: (1) Evaluation of MLLMs using the simulated DeepAccident dataset from CARLA, explicitly addressing the scarcity of diverse, realistic, infrastructure-based accident data through controlled simulations; (2) Comparative performance analysis between Gemini 1.5 and 2.0, Gemma 3 and Pixtral models in acci- dent identification and descriptive capabilities without prior fine-tuning; and (3) Integration of advanced visual analytics, specifically YOLO for object detection, Deep SORT for multi- object tracking, and Segment Anything (SAM) for instance segmentation, into enhanced prompts to improve model accuracy and explainability. Key numerical results show Pixtral as the top performer with an F1-score of 0.71 and 83% recall, while Gemini models gained precision with enhanced prompts (e.g., Gemini 1.5 rose to 90%) but suffered notable F1 and recall losses. Gemma 3 offered the most balanced performance with minimal metric fluctuation. These findings demonstrate the substantial potential of integrating MLLMs with advanced visual analytics techniques, enhancing their applicability in real-world automated traffic monitoring systems. |
| 2025-09-23 | [Adaptive Override Control under High-Relative-Degree Nonovershooting Constraints](http://arxiv.org/abs/2509.18988v1) | Ziliang Lyu, Miroslav Krstic et al. | This paper considers the problem of adaptively overriding unsafe actions of a nominal controller in the presence of high-relative-degree nonovershooting constraints and parametric uncertainties. To prevent the design from being coupled with high-order derivatives of the parameter estimation error, we adopt a modular design approach in which the controller and the parameter identifier are designed separately. The controller module ensures that any safety violations caused by parametric uncertainties remain bounded, provided that the parameter estimation error and its first-order derivative are either bounded or square-integrable. The identifier module, in turn, guarantees that these requirements on the parameter estimation error are satisfied. Both theoretical analysis and simulation results demonstrate that the closed-loop safety violation is bounded by a tunable function of the initial estimation error. Moreover, as time increases, the parameter estimate converges to the true value, and the amount of safety violation decreases accordingly. |
| 2025-09-23 | [LiDAR Point Cloud Image-based Generation Using Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2509.18917v1) | Amirhesam Aghanouri, Cristina Olaverri-Monreal | Autonomous vehicles (AVs) are expected to revolutionize transportation by improving efficiency and safety. Their success relies on 3D vision systems that effectively sense the environment and detect traffic agents. Among sensors AVs use to create a comprehensive view of surroundings, LiDAR provides high-resolution depth data enabling accurate object detection, safe navigation, and collision avoidance. However, collecting real-world LiDAR data is time-consuming and often affected by noise and sparsity due to adverse weather or sensor limitations. This work applies a denoising diffusion probabilistic model (DDPM), enhanced with novel noise scheduling and time-step embedding techniques to generate high-quality synthetic data for augmentation, thereby improving performance across a range of computer vision tasks, particularly in AV perception. These modifications impact the denoising process and the model's temporal awareness, allowing it to produce more realistic point clouds based on the projection. The proposed method was extensively evaluated under various configurations using the IAMCV and KITTI-360 datasets, with four performance metrics compared against state-of-the-art (SOTA) methods. The results demonstrate the model's superior performance over most existing baselines and its effectiveness in mitigating the effects of noisy and sparse LiDAR data, producing diverse point clouds with rich spatial relationships and structural detail. |
| 2025-09-23 | [Beyond the Leaderboard: Understanding Performance Disparities in Large Language Models via Model Diffing](http://arxiv.org/abs/2509.18792v1) | Sabri Boughorbel, Fahim Dalvi et al. | As fine-tuning becomes the dominant paradigm for improving large language models (LLMs), understanding what changes during this process is increasingly important. Traditional benchmarking often fails to explain why one model outperforms another. In this work, we use model diffing, a mechanistic interpretability approach, to analyze the specific capability differences between Gemma-2-9b-it and a SimPO-enhanced variant. Using crosscoders, we identify and categorize latent representations that differentiate the two models. We find that SimPO acquired latent concepts predominantly enhance safety mechanisms (+32.8%), multilingual capabilities (+43.8%), and instruction-following (+151.7%), while its additional training also reduces emphasis on model self-reference (-44.1%) and hallucination management (-68.5%). Our analysis shows that model diffing can yield fine-grained insights beyond leaderboard metrics, attributing performance gaps to concrete mechanistic capabilities. This approach offers a transparent and targeted framework for comparing LLMs. |
| 2025-09-23 | [Real-time Deer Detection and Warning in Connected Vehicles via Thermal Sensing and Deep Learning](http://arxiv.org/abs/2509.18779v1) | Hemanth Puppala, Wayne Sarasua et al. | Deer-vehicle collisions represent a critical safety challenge in the United States, causing nearly 2.1 million incidents annually and resulting in approximately 440 fatalities, 59,000 injuries, and 10 billion USD in economic damages. These collisions also contribute significantly to declining deer populations. This paper presents a real-time detection and driver warning system that integrates thermal imaging, deep learning, and vehicle-to-everything communication to help mitigate deer-vehicle collisions. Our system was trained and validated on a custom dataset of over 12,000 thermal deer images collected in Mars Hill, North Carolina. Experimental evaluation demonstrates exceptional performance with 98.84 percent mean average precision, 95.44 percent precision, and 95.96 percent recall. The system was field tested during a follow-up visit to Mars Hill and readily sensed deer providing the driver with advanced warning. Field testing validates robust operation across diverse weather conditions, with thermal imaging maintaining between 88 and 92 percent detection accuracy in challenging scenarios where conventional visible light based cameras achieve less than 60 percent effectiveness. When a high probability threshold is reached sensor data sharing messages are broadcast to surrounding vehicles and roadside units via cellular vehicle to everything (CV2X) communication devices. Overall, our system achieves end to end latency consistently under 100 milliseconds from detection to driver alert. This research establishes a viable technological pathway for reducing deer-vehicle collisions through thermal imaging and connected vehicles. |
| 2025-09-23 | [AECBench: A Hierarchical Benchmark for Knowledge Evaluation of Large Language Models in the AEC Field](http://arxiv.org/abs/2509.18776v1) | Chen Liang, Zhaoqi Huang et al. | Large language models (LLMs), as a novel information technology, are seeing increasing adoption in the Architecture, Engineering, and Construction (AEC) field. They have shown their potential to streamline processes throughout the building lifecycle. However, the robustness and reliability of LLMs in such a specialized and safety-critical domain remain to be evaluated. To address this challenge, this paper establishes AECBench, a comprehensive benchmark designed to quantify the strengths and limitations of current LLMs in the AEC domain. The benchmark defines 23 representative tasks within a five-level cognition-oriented evaluation framework encompassing Knowledge Memorization, Understanding, Reasoning, Calculation, and Application. These tasks were derived from authentic AEC practice, with scope ranging from codes retrieval to specialized documents generation. Subsequently, a 4,800-question dataset encompassing diverse formats, including open-ended questions, was crafted primarily by engineers and validated through a two-round expert review. Furthermore, an LLM-as-a-Judge approach was introduced to provide a scalable and consistent methodology for evaluating complex, long-form responses leveraging expert-derived rubrics. Through the evaluation of nine LLMs, a clear performance decline across five cognitive levels was revealed. Despite demonstrating proficiency in foundational tasks at the Knowledge Memorization and Understanding levels, the models showed significant performance deficits, particularly in interpreting knowledge from tables in building codes, executing complex reasoning and calculation, and generating domain-specific documents. Consequently, this study lays the groundwork for future research and development aimed at the robust and reliable integration of LLMs into safety-critical engineering practices. |
| 2025-09-23 | [Guaranteed Robust Nonlinear MPC via Disturbance Feedback](http://arxiv.org/abs/2509.18760v1) | Antoine P. Leeman, Johannes Köhler et al. | Robots must satisfy safety-critical state and input constraints despite disturbances and model mismatch. We introduce a robust model predictive control (RMPC) formulation that is fast, scalable, and compatible with real-time implementation. Our formulation guarantees robust constraint satisfaction, input-to-state stability (ISS) and recursive feasibility. The key idea is to decompose the uncertain nonlinear system into (i) a nominal nonlinear dynamic model, (ii) disturbance-feedback controllers, and (iii) bounds on the model error. These components are optimized jointly using sequential convex programming. The resulting convex subproblems are solved efficiently using a recent disturbance-feedback MPC solver. The approach is validated across multiple dynamics, including a rocket-landing problem with steerable thrust. An open-source implementation is available at https://github.com/antoineleeman/robust-nonlinear-mpc. |
| 2025-09-23 | [Lightweight Vision Transformer with Window and Spatial Attention for Food Image Classification](http://arxiv.org/abs/2509.18692v1) | Xinle Gao, Linghui Ye et al. | With the rapid development of society and continuous advances in science and technology, the food industry increasingly demands higher production quality and efficiency. Food image classification plays a vital role in enabling automated quality control on production lines, supporting food safety supervision, and promoting intelligent agricultural production. However, this task faces challenges due to the large number of parameters and high computational complexity of Vision Transformer models. To address these issues, we propose a lightweight food image classification algorithm that integrates a Window Multi-Head Attention Mechanism (WMHAM) and a Spatial Attention Mechanism (SAM). The WMHAM reduces computational cost by capturing local and global contextual features through efficient window partitioning, while the SAM adaptively emphasizes key spatial regions to improve discriminative feature representation. Experiments conducted on the Food-101 and Vireo Food-172 datasets demonstrate that our model achieves accuracies of 95.24% and 94.33%, respectively, while significantly reducing parameters and FLOPs compared with baseline methods. These results confirm that the proposed approach achieves an effective balance between computational efficiency and classification performance, making it well-suited for deployment in resource-constrained environments. |
| 2025-09-23 | [Verification and Synthesis of Discrete-Time Control Barrier Functions](http://arxiv.org/abs/2509.18685v1) | Erfan Shakhesi, W. P. M. H. Heemels et al. | Discrete-time Control Barrier Functions (DTCBFs) have recently attracted interest for guaranteeing safety and synthesizing safe controllers for discrete-time dynamical systems. This paper addresses the open challenges of verifying candidate DTCBFs and synthesizing DTCBFs for general nonlinear discrete-time systems with input constraints and arbitrary safe sets. In particular, we propose a branch-and-bound method, inspired by the $\alpha$BB algorithm, for the verification of candidate DTCBFs in both cases, whether a corresponding control policy is known or unknown. We prove that this method, in a finite number of iterations, either verifies a given candidate function as a valid DTCBF or falsifies it by providing a counterexample (within predefined tolerances). As a second main contribution, we propose a novel bilevel optimization approach to synthesize a DTCBF and a corresponding control policy in finite time. This involves determining the unknown coefficients of a parameterized DTCBF and a parameterized control policy. Furthermore, we introduce various strategies to reduce the computational burden of the bilevel approach. We also demonstrate our methods using numerical case studies. |
| 2025-09-23 | [Implementation of airborne ML models with semantics preservation](http://arxiv.org/abs/2509.18681v1) | Nicolas Valot, Louis Fabre et al. | Machine Learning (ML) may offer new capabilities in airborne systems. However, as any piece of airborne systems, ML-based systems will be required to guarantee their safe operation. Thus, their development will have to be demonstrated to be compliant with the adequate guidance. So far, the European Union Aviation Safety Agency (EASA) has published a concept paper and an EUROCAE/SAE group is preparing ED-324. Both approaches delineate high-level objectives to confirm the ML model achieves its intended function and maintains training performance in the target environment. The paper aims to clarify the difference between an ML model and its corresponding unambiguous description, referred to as the Machine Learning Model Description (MLMD). It then refines the essential notion of semantics preservation to ensure the accurate replication of the model. We apply our contributions to several industrial use cases to build and compare several target models. |
| 2025-09-23 | [SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer](http://arxiv.org/abs/2509.18648v1) | Yarden As, Chengrui Qu et al. | Safety remains a major concern for deploying reinforcement learning (RL) in real-world applications. Simulators provide safe, scalable training environments, but the inevitable sim-to-real gap introduces additional safety concerns, as policies must satisfy constraints in real-world conditions that differ from simulation. To address this challenge, robust safe RL techniques offer principled methods, but are often incompatible with standard scalable training pipelines. In contrast, domain randomization, a simple and popular sim-to-real technique, stands out as a promising alternative, although it often results in unsafe behaviors in practice. We present SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance. |
| 2025-09-23 | [Number Adaptive Formation Flight Planning via Affine Deformable Guidance in Narrow Environments](http://arxiv.org/abs/2509.18636v1) | Yuan Zhou, Jialiang Hou et al. | Formation maintenance with varying number of drones in narrow environments hinders the convergence of planning to the desired configurations. To address this challenge, this paper proposes a formation planning method guided by Deformable Virtual Structures (DVS) with continuous spatiotemporal transformation. Firstly, to satisfy swarm safety distance and preserve formation shape filling integrity for irregular formation geometries, we employ Lloyd algorithm for uniform $\underline{PA}$rtitioning and Hungarian algorithm for $\underline{AS}$signment (PAAS) in DVS. Subsequently, a spatiotemporal trajectory involving DVS is planned using primitive-based path search and nonlinear trajectory optimization. The DVS trajectory achieves adaptive transitions with respect to a varying number of drones while ensuring adaptability to narrow environments through affine transformation. Finally, each agent conducts distributed trajectory planning guided by desired spatiotemporal positions within the DVS, while incorporating collision avoidance and dynamic feasibility requirements. Our method enables up to 15\% of swarm numbers to join or leave in cluttered environments while rapidly restoring the desired formation shape in simulation. Compared to cutting-edge formation planning method, we demonstrate rapid formation recovery capacity and environmental adaptability. Real-world experiments validate the effectiveness and resilience of our formation planning method. |
| 2025-09-23 | [The Case for Negative Data: From Crash Reports to Counterfactuals for Reasonable Driving](http://arxiv.org/abs/2509.18626v1) | Jay Patrikar, Apoorva Sharma et al. | Learning-based autonomous driving systems are trained mostly on incident-free data, offering little guidance near safety-performance boundaries. Real crash reports contain precisely the contrastive evidence needed, but they are hard to use: narratives are unstructured, third-person, and poorly grounded to sensor views. We address these challenges by normalizing crash narratives to ego-centric language and converting both logs and crashes into a unified scene-action representation suitable for retrieval. At decision time, our system adjudicates proposed actions by retrieving relevant precedents from this unified index; an agentic counterfactual extension proposes plausible alternatives, retrieves for each, and reasons across outcomes before deciding. On a nuScenes benchmark, precedent retrieval substantially improves calibration, with recall on contextually preferred actions rising from 24% to 53%. The counterfactual variant preserves these gains while sharpening decisions near risk. |
| 2025-09-23 | [Interaction-aware Lane-Changing Early Warning System in Congested Traffic](http://arxiv.org/abs/2509.18624v1) | Yue Zhang, Xinzhi Zhong et al. | Lane changes (LCs) in congested traffic are complex, multi-vehicle interactive events that pose significant safety concerns. Providing early warnings can enable more proactive driver assistance system and support more informed decision-making for drivers under LCs. This paper presents an interaction-aware Lane-Changing Early Warning (LCEW) system designed to issue reliable early warning signals based on future trajectory predictions. We first investigate the stochastic nature of LCs, characterized by (i) variable-size multi-vehicle interactions and (ii) the direct and indirect risks resulting from these interactions. To model these stochastic interactions, a Social Spatio-Temporal Graph Convolutional Neural Network framework informed by mutual information (STGCNN-MI) is introduced to predict multi-vehicle trajectories. By leveraging a MI-based adjacency matrix, the framework enhances trajectory prediction accuracy while providing interpretable representations of vehicle interactions. Then, potential collisions between the LC vehicle and adjacent vehicles (direct risks) or among the non-adjacent vehicles (indirect risks) are identified using oriented bounding box detection applied to the predicted trajectories. Finally, a warning signal is generated to inform the LC driver of location of potential collisions within the predicted time window. Traffic simulation experiments conducted in SUMO demonstrate that the proposed interaction-aware LCEW improves both vehicle-level safety and overall traffic efficiency, while also promoting more natural behavioral adaptation. |
| 2025-09-23 | [Enhancing Video Object Segmentation in TrackRAD Using XMem Memory Network](http://arxiv.org/abs/2509.18591v1) | Pengchao Deng, Shengqi Chen | This paper presents an advanced tumor segmentation framework for real-time MRI-guided radiotherapy, designed for the TrackRAD2025 challenge. Our method leverages the XMem model, a memory-augmented architecture, to segment tumors across long cine-MRI sequences. The proposed system efficiently integrates memory mechanisms to track tumor motion in real-time, achieving high segmentation accuracy even under challenging conditions with limited annotated data. Unfortunately, the detailed experimental records have been lost, preventing us from reporting precise quantitative results at this stage. Nevertheless, From our preliminary impressions during development, the XMem-based framework demonstrated reasonable segmentation performance and satisfied the clinical real-time requirement. Our work contributes to improving the precision of tumor tracking during MRI-guided radiotherapy, which is crucial for enhancing the accuracy and safety of cancer treatments. |

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



