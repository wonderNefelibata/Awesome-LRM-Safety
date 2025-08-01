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
| 2025-07-31 | [Deep Learning-based Prediction of Clinical Trial Enrollment with Uncertainty Estimates](http://arxiv.org/abs/2507.23607v1) | Tien Huu Do, Antoine Masquelier et al. | Clinical trials are a systematic endeavor to assess the safety and efficacy of new drugs or treatments. Conducting such trials typically demands significant financial investment and meticulous planning, highlighting the need for accurate predictions of trial outcomes. Accurately predicting patient enrollment, a key factor in trial success, is one of the primary challenges during the planning phase. In this work, we propose a novel deep learning-based method to address this critical challenge. Our method, implemented as a neural network model, leverages pre-trained language models (PLMs) to capture the complexities and nuances of clinical documents, transforming them into expressive representations. These representations are then combined with encoded tabular features via an attention mechanism. To account for uncertainties in enrollment prediction, we enhance the model with a probabilistic layer based on the Gamma distribution, which enables range estimation. We apply the proposed model to predict clinical trial duration, assuming site-level enrollment follows a Poisson-Gamma process. We carry out extensive experiments on real-world clinical trial data, and show that the proposed method can effectively predict the number of patients enrolled at a number of sites for a given clinical trial, outperforming established baseline models. |
| 2025-07-31 | [Explanations for Unrealizability of Infinite-State Safety Shields](http://arxiv.org/abs/2507.23603v1) | Andoni Rodriguez, Irfansha Shaik et al. | Safe Reinforcement Learning focuses on developing optimal policies while ensuring safety. A popular method to address such task is shielding, in which a correct-by-construction safety component is synthesized from logical specifications. Recently, shield synthesis has been extended to infinite-state domains, such as continuous environments. This makes shielding more applicable to realistic scenarios. However, often shields might be unrealizable because the specification is inconsistent (e.g., contradictory). In order to address this gap, we present a method to obtain simple unconditional and conditional explanations that witness unrealizability, which goes by temporal formula unrolling. In this paper, we show different variants of the technique and its applicability. |
| 2025-07-31 | [A Unified Perception-Language-Action Framework for Adaptive Autonomous Driving](http://arxiv.org/abs/2507.23540v1) | Yi Zhang, Erik Leo Haß et al. | Autonomous driving systems face significant challenges in achieving human-like adaptability, robustness, and interpretability in complex, open-world environments. These challenges stem from fragmented architectures, limited generalization to novel scenarios, and insufficient semantic extraction from perception. To address these limitations, we propose a unified Perception-Language-Action (PLA) framework that integrates multi-sensor fusion (cameras, LiDAR, radar) with a large language model (LLM)-augmented Vision-Language-Action (VLA) architecture, specifically a GPT-4.1-powered reasoning core. This framework unifies low-level sensory processing with high-level contextual reasoning, tightly coupling perception with natural language-based semantic understanding and decision-making to enable context-aware, explainable, and safety-bounded autonomous driving. Evaluations on an urban intersection scenario with a construction zone demonstrate superior performance in trajectory tracking, speed prediction, and adaptive planning. The results highlight the potential of language-augmented cognitive frameworks for advancing the safety, interpretability, and scalability of autonomous driving systems. |
| 2025-07-31 | [Distributionally Robust Cascading Risk Quantification in Multi-Agent Rendezvous: Effects of Time Delay and Network Connectivity](http://arxiv.org/abs/2507.23489v1) | Vivek Pandey, Nader Motee | Achieving safety in autonomous multi-agent systems, particularly in time-critical tasks like rendezvous, is a critical challenge. In this paper, we propose a distributionally robust risk framework for analyzing cascading failures in multi-agent rendezvous. To capture the complex interactions between network connectivity, system dynamics, and communication delays, we use a time-delayed network model as a benchmark. We introduce a conditional distributionally robust functional to quantify cascading effects between agents, utilizing a bi-variate normal distribution. Our approach yields closed-form risk expressions that reveal the impact of time delay, noise statistics, communication topology, and failure modes on rendezvous risk. The insights derived inform the design of resilient networks that mitigate the risk of cascading failures. We validate our theoretical results through extensive simulations, demonstrating the effectiveness of our framework. |
| 2025-07-31 | [A Novel Evaluation Benchmark for Medical LLMs: Illuminating Safety and Effectiveness in Clinical Domains](http://arxiv.org/abs/2507.23486v1) | Shirui Wang, Zhihui Tang et al. | Large language models (LLMs) hold promise in clinical decision support but face major challenges in safety evaluation and effectiveness validation. We developed the Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB), a multidimensional framework built on clinical expert consensus, encompassing 30 criteria covering critical areas like critical illness recognition, guideline adherence, and medication safety, with weighted consequence measures. Thirty-two specialist physicians developed and reviewed 2,069 open-ended Q&A items aligned with these criteria, spanning 26 clinical departments to simulate real-world scenarios. Benchmark testing of six LLMs revealed moderate overall performance (average total score 57.2%, safety 54.7%, effectiveness 62.3%), with a significant 13.3% performance drop in high-risk scenarios (p < 0.0001). Domain-specific medical LLMs showed consistent performance advantages over general-purpose models, with relatively higher top scores in safety (0.912) and effectiveness (0.861). The findings of this study not only provide a standardized metric for evaluating the clinical application of medical LLMs, facilitating comparative analyses, risk exposure identification, and improvement directions across different scenarios, but also hold the potential to promote safer and more effective deployment of large language models in healthcare environments. |
| 2025-07-31 | [CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV Tracking in Complex Scenes](http://arxiv.org/abs/2507.23473v1) | Bin Xie, Congxuan Zhang et al. | The widespread application of Unmanned Aerial Vehicles (UAVs) has raised serious public safety and privacy concerns, making UAV perception crucial for anti-UAV tasks. However, existing UAV tracking datasets predominantly feature conspicuous objects and lack diversity in scene complexity and attribute representation, limiting their applicability to real-world scenarios. To overcome these limitations, we present the CST Anti-UAV, a new thermal infrared dataset specifically designed for Single Object Tracking (SOT) in Complex Scenes with Tiny UAVs (CST). It contains 220 video sequences with over 240k high-quality bounding box annotations, highlighting two key properties: a significant number of tiny-sized UAV targets and the diverse and complex scenes. To the best of our knowledge, CST Anti-UAV is the first dataset to incorporate complete manual frame-level attribute annotations, enabling precise evaluations under varied challenges. To conduct an in-depth performance analysis for CST Anti-UAV, we evaluate 20 existing SOT methods on the proposed dataset. Experimental results demonstrate that tracking tiny UAVs in complex environments remains a challenge, as the state-of-the-art method achieves only 35.92% state accuracy, much lower than the 67.69% observed on the Anti-UAV410 dataset. These findings underscore the limitations of existing benchmarks and the need for further advancements in UAV tracking research. The CST Anti-UAV benchmark is about to be publicly released, which not only fosters the development of more robust SOT methods but also drives innovation in anti-UAV systems. |
| 2025-07-31 | [Role-Aware Language Models for Secure and Contextualized Access Control in Organizations](http://arxiv.org/abs/2507.23465v1) | Saeed Almheiri, Yerulan Kongrat et al. | As large language models (LLMs) are increasingly deployed in enterprise settings, controlling model behavior based on user roles becomes an essential requirement. Existing safety methods typically assume uniform access and focus on preventing harmful or toxic outputs, without addressing role-specific access constraints. In this work, we investigate whether LLMs can be fine-tuned to generate responses that reflect the access privileges associated with different organizational roles. We explore three modeling strategies: a BERT-based classifier, an LLM-based classifier, and role-conditioned generation. To evaluate these approaches, we construct two complementary datasets. The first is adapted from existing instruction-tuning corpora through clustering and role labeling, while the second is synthetically generated to reflect realistic, role-sensitive enterprise scenarios. We assess model performance across varying organizational structures and analyze robustness to prompt injection, role mismatch, and jailbreak attempts. |
| 2025-07-31 | [Breaking the mould of Social Mixed Reality -- State-of-the-Art and Glossary](http://arxiv.org/abs/2507.23454v1) | Marta Bieńkiewicz, Julia Ayache et al. | This article explores a critical gap in Mixed Reality (MR) technology: while advances have been made, MR still struggles to authentically replicate human embodiment and socio-motor interaction. For MR to enable truly meaningful social experiences, it needs to incorporate multi-modal data streams and multi-agent interaction capabilities. To address this challenge, we present a comprehensive glossary covering key topics such as Virtual Characters and Autonomisation, Responsible AI, Ethics by Design, and the Scientific Challenges of Social MR within Neuroscience, Embodiment, and Technology. Our aim is to drive the transformative evolution of MR technologies that prioritize human-centric innovation, fostering richer digital connections. We advocate for MR systems that enhance social interaction and collaboration between humans and virtual autonomous agents, ensuring inclusivity, ethical design and psychological safety in the process. |
| 2025-07-31 | [Policy Learning from Large Vision-Language Model Feedback without Reward Modeling](http://arxiv.org/abs/2507.23391v1) | Tung M. Luu, Donghoon Lee et al. | Offline reinforcement learning (RL) provides a powerful framework for training robotic agents using pre-collected, suboptimal datasets, eliminating the need for costly, time-consuming, and potentially hazardous online interactions. This is particularly useful in safety-critical real-world applications, where online data collection is expensive and impractical. However, existing offline RL algorithms typically require reward labeled data, which introduces an additional bottleneck: reward function design is itself costly, labor-intensive, and requires significant domain expertise. In this paper, we introduce PLARE, a novel approach that leverages large vision-language models (VLMs) to provide guidance signals for agent training. Instead of relying on manually designed reward functions, PLARE queries a VLM for preference labels on pairs of visual trajectory segments based on a language task description. The policy is then trained directly from these preference labels using a supervised contrastive preference learning objective, bypassing the need to learn explicit reward models. Through extensive experiments on robotic manipulation tasks from the MetaWorld, PLARE achieves performance on par with or surpassing existing state-of-the-art VLM-based reward generation methods. Furthermore, we demonstrate the effectiveness of PLARE in real-world manipulation tasks with a physical robot, further validating its practical applicability. |
| 2025-07-31 | [Contrastive Learning-Driven Traffic Sign Perception: Multi-Modal Fusion of Text and Vision](http://arxiv.org/abs/2507.23331v1) | Qiang Lu, Waikit Xiu et al. | Traffic sign recognition, as a core component of autonomous driving perception systems, directly influences vehicle environmental awareness and driving safety. Current technologies face two significant challenges: first, the traffic sign dataset exhibits a pronounced long-tail distribution, resulting in a substantial decline in recognition performance of traditional convolutional networks when processing low-frequency and out-of-distribution classes; second, traffic signs in real-world scenarios are predominantly small targets with significant scale variations, making it difficult to extract multi-scale features.To overcome these issues, we propose a novel two-stage framework combining open-vocabulary detection and cross-modal learning. For traffic sign detection, our NanoVerse YOLO model integrates a reparameterizable vision-language path aggregation network (RepVL-PAN) and an SPD-Conv module to specifically enhance feature extraction for small, multi-scale targets. For traffic sign classification, we designed a Traffic Sign Recognition Multimodal Contrastive Learning model (TSR-MCL). By contrasting visual features from a Vision Transformer with semantic features from a rule-based BERT, TSR-MCL learns robust, frequency-independent representations, effectively mitigating class confusion caused by data imbalance. On the TT100K dataset, our method achieves a state-of-the-art 78.4% mAP in the long-tail detection task for all-class recognition. The model also obtains 91.8% accuracy and 88.9% recall, significantly outperforming mainstream algorithms and demonstrating superior accuracy and generalization in complex, open-world scenarios. |
| 2025-07-31 | [A Framework for Ethical Decision-Making in Automated Vehicles through Human Reasons-based Supervision](http://arxiv.org/abs/2507.23308v1) | Lucas Elbert Suryana, Saeed Rahmani et al. | Ethical dilemmas are a common challenge in everyday driving, requiring human drivers to balance competing priorities such as safety, efficiency, and rule compliance. However, much of the existing research in automated vehicles (AVs) has focused on high-stakes "trolley problems," which involve extreme and rare situations. Such scenarios, though rich in ethical implications, are rarely applicable in real-world AV decision-making. In practice, when AVs confront everyday ethical dilemmas, they often appear to prioritise strict adherence to traffic rules. By contrast, human drivers may bend the rules in context-specific situations, using judgement informed by practical concerns such as safety and efficiency. According to the concept of meaningful human control, AVs should respond to human reasons, including those of drivers, vulnerable road users, and policymakers. This work introduces a novel human reasons-based supervision framework that detects when AV behaviour misaligns with expected human reasons to trigger trajectory reconsideration. The framework integrates with motion planning and control systems to support real-time adaptation, enabling decisions that better reflect safety, efficiency, and regulatory considerations. Simulation results demonstrate that this approach could help AVs respond more effectively to ethical challenges in dynamic driving environments by prompting replanning when the current trajectory fails to align with human reasons. These findings suggest that our approach offers a path toward more adaptable, human-centered decision-making in AVs. |
| 2025-07-31 | [Data-Driven Stochastic Control via Non-i.i.d. Trajectories: Foundations and Guarantees](http://arxiv.org/abs/2507.23280v1) | Abolfazl Lavaei | This work establishes a crucial step toward advancing data-driven trajectory-based methods for stochastic systems with unknown mathematical dynamics. In contrast to scenario-based approaches that rely on independent and identically distributed (i.i.d.) trajectories, this work develops a data-driven framework where each trajectory is gathered over a finite horizon and exhibits temporal dependence-referred to as a non-i.i.d. trajectory. To ensure safety of dynamical systems using such trajectories, the current body of literature primarily considers dynamics subject to unknown-but-bounded disturbances, which facilitates robust analysis. While promising, such bounds may be violated in practice and the resulting worst-case robust analysis tends to be overly conservative. To overcome these fundamental challenges, this paper considers stochastic systems with unknown mathematical dynamics, influenced by process noise with unknown distributions. In the proposed framework, data is collected from stochastic systems under multiple realizations within a finite-horizon experiment, where each realization generates a non-i.i.d. trajectory. Leveraging the concept of stochastic control barrier certificates constructed from data, this work quantifies probabilistic safety guarantees with a certified confidence level. To achieve this, the proposed conditions are formulated as sum-of-squares (SOS) optimization problems, relying solely on empirical average of the collected trajectories and statistical features of the process noise. The efficacy of the approach has been validated on three stochastic benchmarks with both unknown models and noise distributions. In one case study, it is shown that while no safety controller exists for the robust analysis of the system under bounded disturbances, the proposed stochastic framework offers a safety controller with guaranteed probabilistic satisfaction. |
| 2025-07-31 | [Toward Safe, Trustworthy and Realistic Augmented Reality User Experience](http://arxiv.org/abs/2507.23226v1) | Yanming Xiu | As augmented reality (AR) becomes increasingly integrated into everyday life, ensuring the safety and trustworthiness of its virtual content is critical. Our research addresses the risks of task-detrimental AR content, particularly that which obstructs critical information or subtly manipulates user perception. We developed two systems, ViDDAR and VIM-Sense, to detect such attacks using vision-language models (VLMs) and multimodal reasoning modules. Building on this foundation, we propose three future directions: automated, perceptually aligned quality assessment of virtual content; detection of multimodal attacks; and adaptation of VLMs for efficient and user-centered deployment on AR devices. Overall, our work aims to establish a scalable, human-aligned framework for safeguarding AR experiences and seeks feedback on perceptual modeling, multimodal AR content implementation, and lightweight model adaptation. |
| 2025-07-31 | [YOLO-ROC: A High-Precision and Ultra-Lightweight Model for Real-Time Road Damage Detection](http://arxiv.org/abs/2507.23225v1) | Zicheng Lin, Weichao Pan | Road damage detection is a critical task for ensuring traffic safety and maintaining infrastructure integrity. While deep learning-based detection methods are now widely adopted, they still face two core challenges: first, the inadequate multi-scale feature extraction capabilities of existing networks for diverse targets like cracks and potholes, leading to high miss rates for small-scale damage; and second, the substantial parameter counts and computational demands of mainstream models, which hinder their deployment for efficient, real-time detection in practical applications. To address these issues, this paper proposes a high-precision and lightweight model, YOLO - Road Orthogonal Compact (YOLO-ROC). We designed a Bidirectional Multi-scale Spatial Pyramid Pooling Fast (BMS-SPPF) module to enhance multi-scale feature extraction and implemented a hierarchical channel compression strategy to reduce computational complexity. The BMS-SPPF module leverages a bidirectional spatial-channel attention mechanism to improve the detection of small targets. Concurrently, the channel compression strategy reduces the parameter count from 3.01M to 0.89M and GFLOPs from 8.1 to 2.6. Experiments on the RDD2022_China_Drone dataset demonstrate that YOLO-ROC achieves a mAP50 of 67.6%, surpassing the baseline YOLOv8n by 2.11%. Notably, the mAP50 for the small-target D40 category improved by 16.8%, and the final model size is only 2.0 MB. Furthermore, the model exhibits excellent generalization performance on the RDD2022_China_Motorbike dataset. |
| 2025-07-30 | [Lightweight Language Models are Prone to Reasoning Errors for Complex Computational Phenotyping Tasks](http://arxiv.org/abs/2507.23146v1) | Sarah Pungitore, Shashank Yadav et al. | Objective: Although computational phenotyping is a central informatics activity with resulting cohorts supporting a wide variety of applications, it is time-intensive because of manual data review. We previously assessed the ability of LLMs to perform computational phenotyping tasks using computable phenotypes for ARF respiratory support therapies. They successfully performed concept classification and classification of single-therapy phenotypes, but underperformed on multiple-therapy phenotypes. To understand issues with these complex tasks, we expanded PHEONA, a generalizable framework for evaluation of LLMs, to include methods specifically for evaluating faulty reasoning. Materials and Methods: We assessed the responses of three lightweight LLMs (DeepSeek-r1 32 billion, Mistral Small 24 billion, and Phi-4 14 billion) both with and without prompt modifications to identify explanation correctness and unfaithfulness errors for phenotyping. Results: For experiments without prompt modifications, both errors were present across all models although more responses had explanation correctness errors than unfaithfulness errors. For experiments assessing accuracy impact after prompt modifications, DeepSeek, a reasoning model, had the smallest overall accuracy impact when compared to Mistral and Phi. Discussion: Since reasoning errors were ubiquitous across models, our enhancement of PHEONA to include a component for assessing faulty reasoning provides critical support for LLM evaluation and evidence for reasoning errors for complex tasks. While insights from reasoning errors can help prompt refinement, a deeper understanding of why LLM reasoning errors occur will likely require further development and refinement of interpretability methods. Conclusion: Reasoning errors were pervasive across LLM responses for computational phenotyping, a complex reasoning task |
| 2025-07-30 | [Observational Multiplicity](http://arxiv.org/abs/2507.23136v1) | Erin George, Deanna Needell et al. | Many prediction tasks can admit multiple models that can perform almost equally well. This phenomenon can can undermine interpretability and safety when competing models assign conflicting predictions to individuals. In this work, we study how arbitrariness can arise in probabilistic classification tasks as a result of an effect that we call \emph{observational multiplicity}. We discuss how this effect arises in a broad class of practical applications where we learn a classifier to predict probabilities $p_i \in [0,1]$ but are given a dataset of observations $y_i \in \{0,1\}$. We propose to evaluate the arbitrariness of individual probability predictions through the lens of \emph{regret}. We introduce a measure of regret for probabilistic classification tasks, which measures how the predictions of a model could change as a result of different training labels change. We present a general-purpose method to estimate the regret in a probabilistic classification task. We use our measure to show that regret is higher for certain groups in the dataset and discuss potential applications of regret. We demonstrate how estimating regret promote safety in real-world applications by abstention and data collection. |
| 2025-07-30 | [Stability-Constrained AC Optimal Power Flow -- A Gaussian Process-Based Approach](http://arxiv.org/abs/2507.23094v1) | Vincenzo Di Vito, Kaarthik Sundar et al. | The Alternating Current Optimal Power Flow (ACOPF) problem is a core task in power system operations, aimed at determining cost-effective generation dispatch while satisfying physical and operational constraints. However, conventional ACOPF formulations rely on steady-state models and neglect the dynamic behavior of generators, which can lead to operating points that are economically optimal but dynamically unstable. This paper proposes a novel, data-driven approach to incorporate generator dynamics into the ACOPF using Gaussian Process (GP) models. Specifically, it introduces an exponential surrogate function to characterize the stability of solutions to the differential equations governing synchronous generator dynamics. The exponent, which indicates whether system trajectories decay (stable) or grow (unstable), is learned as a function of the bus voltage using GP regression. Crucially, the framework enables probabilistic stability assessment to be integrated directly into the optimization process. The resulting dynamics-aware ACOPF formulation identifies operating points that satisfy both operational safety and dynamic stability criteria. Numerical experiments on the IEEE 39-bus, 57-bus, and 118-bus systems demonstrate that the proposed method efficiently captures generator dynamics using limited training data, leading to more reliable and robust decisions across a wide range of operating conditions. |
| 2025-07-30 | [Reference-Guided Diffusion Inpainting For Multimodal Counterfactual Generation](http://arxiv.org/abs/2507.23058v1) | Alexandru Buburuzan | Safety-critical applications, such as autonomous driving and medical image analysis, require extensive multimodal data for rigorous testing. Synthetic data methods are gaining prominence due to the cost and complexity of gathering real-world data, but they demand a high degree of realism and controllability to be useful. This work introduces two novel methods for synthetic data generation in autonomous driving and medical image analysis, namely MObI and AnydoorMed, respectively. MObI is a first-of-its-kind framework for Multimodal Object Inpainting that leverages a diffusion model to produce realistic and controllable object inpaintings across perceptual modalities, demonstrated simultaneously for camera and lidar. Given a single reference RGB image, MObI enables seamless object insertion into existing multimodal scenes at a specified 3D location, guided by a bounding box, while maintaining semantic consistency and multimodal coherence. Unlike traditional inpainting methods that rely solely on edit masks, this approach uses 3D bounding box conditioning to ensure accurate spatial positioning and realistic scaling. AnydoorMed extends this paradigm to the medical imaging domain, focusing on reference-guided inpainting for mammography scans. It leverages a diffusion-based model to inpaint anomalies with impressive detail preservation, maintaining the reference anomaly's structural integrity while semantically blending it with the surrounding tissue. Together, these methods demonstrate that foundation models for reference-guided inpainting in natural images can be readily adapted to diverse perceptual modalities, paving the way for the next generation of systems capable of constructing highly realistic, controllable and multimodal counterfactual scenarios. |
| 2025-07-30 | [Early Goal-Guided Multi-Scale Fusion for Real-Time Vision-Language Driving](http://arxiv.org/abs/2507.23042v1) | Santosh Patapati, Trisanth Srinivasan | Autonomous vehicles must react in milliseconds while reasoning about road geometry and traffic intent to navigate complex situations. We introduce NovaDrive, a single-branch vision-language architecture that processes front-camera images, HD-map tiles, LiDAR depth, and textual waypoints in a single branch. A lightweight, two-stage cross-attention block first aligns waypoint tokens with the HD map, then refines attention over fine-grained image and depth patches. Coupled with a novel smoothness loss that discourages abrupt steering and speed changes, this design eliminates the need for recurrent memory. We fine-tune the top 15 layers of an 11B LLaMA-3.2 vision-language backbone, enabling real-time inference. On the nuScenes / Waymo subset of the MD-NEX Outdoor benchmark, NovaDrive raises success rate to 84% (+4%), boosts path-efficiency (SPL) to 0.66 (+0.11), and reduces collision frequency from 2.6% to 1.2% (-1.4%) relative to the previous state-of-the-art. Our ablations confirm that waypoint tokens, partial VLM fine-tuning, and the cross-attention fusion each contribute the most to these gains. Beyond safety, NovaDrive's shorter routes (resulting from the novel smoothness loss) translate to lower fuel or battery usage, pointing toward leaner, more easily updated driving stacks. NovaDrive can be extended to other embodied-AI domains as well. |
| 2025-07-30 | [DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](http://arxiv.org/abs/2507.22813v1) | Hossein Mirzaei, Zeinab Taghavi et al. | Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL. |

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



