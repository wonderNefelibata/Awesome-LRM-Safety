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
| 2025-07-16 | [MMHU: A Massive-Scale Multimodal Benchmark for Human Behavior Understanding](http://arxiv.org/abs/2507.12463v1) | Renjie Li, Ruijie Ye et al. | Humans are integral components of the transportation ecosystem, and understanding their behaviors is crucial to facilitating the development of safe driving systems. Although recent progress has explored various aspects of human behavior$\unicode{x2014}$such as motion, trajectories, and intention$\unicode{x2014}$a comprehensive benchmark for evaluating human behavior understanding in autonomous driving remains unavailable. In this work, we propose $\textbf{MMHU}$, a large-scale benchmark for human behavior analysis featuring rich annotations, such as human motion and trajectories, text description for human motions, human intention, and critical behavior labels relevant to driving safety. Our dataset encompasses 57k human motion clips and 1.73M frames gathered from diverse sources, including established driving datasets such as Waymo, in-the-wild videos from YouTube, and self-collected data. A human-in-the-loop annotation pipeline is developed to generate rich behavior captions. We provide a thorough dataset analysis and benchmark multiple tasks$\unicode{x2014}$ranging from motion prediction to motion generation and human behavior question answering$\unicode{x2014}$thereby offering a broad evaluation suite. Project page : https://MMHU-Benchmark.github.io. |
| 2025-07-16 | [Vision-based Perception for Autonomous Vehicles in Obstacle Avoidance Scenarios](http://arxiv.org/abs/2507.12449v1) | Van-Hoang-Anh Phan, Chi-Tam Nguyen et al. | Obstacle avoidance is essential for ensuring the safety of autonomous vehicles. Accurate perception and motion planning are crucial to enabling vehicles to navigate complex environments while avoiding collisions. In this paper, we propose an efficient obstacle avoidance pipeline that leverages a camera-only perception module and a Frenet-Pure Pursuit-based planning strategy. By integrating advancements in computer vision, the system utilizes YOLOv11 for object detection and state-of-the-art monocular depth estimation models, such as Depth Anything V2, to estimate object distances. A comparative analysis of these models provides valuable insights into their accuracy, efficiency, and robustness in real-world conditions. The system is evaluated in diverse scenarios on a university campus, demonstrating its effectiveness in handling various obstacles and enhancing autonomous navigation. The video presenting the results of the obstacle avoidance experiments is available at: https://www.youtube.com/watch?v=FoXiO5S_tA8 |
| 2025-07-16 | [Design and Development of an Automated Contact Angle Tester (ACAT) for Surface Wettability Measurement](http://arxiv.org/abs/2507.12431v1) | Connor Burgess, Kyle Douin et al. | The Automated Contact Angle Tester (ACAT) is a fully integrated robotic work cell developed to automate the measurement of surface wettability on 3D-printed materials. Designed for precision, repeatability, and safety, ACAT addresses the limitations of manual contact angle testing by combining programmable robotics, precise liquid dispensing, and a modular software-hardware architecture. The system is composed of three core subsystems: (1) an electrical system including power, control, and safety circuits compliant with industrial standards such as NEC 70, NFPA 79, and UL 508A; (2) a software control system based on a Raspberry Pi and Python, featuring fault detection, GPIO logic, and operator interfaces; and (3) a mechanical system that includes a 3-axis Cartesian robot, pneumatic actuation, and a precision liquid dispenser enclosed within a safety-certified frame. The ACAT enables high-throughput, automated surface characterization and provides a robust platform for future integration into smart manufacturing and materials discovery workflows. This paper details the design methodology, implementation strategies, and system integration required to develop the ACAT platform. |
| 2025-07-16 | [Can We Predict Alignment Before Models Finish Thinking? Towards Monitoring Misaligned Reasoning Models](http://arxiv.org/abs/2507.12428v1) | Yik Siu Chan, Zheng-Xin Yong et al. | Open-weights reasoning language models generate long chains-of-thought (CoTs) before producing a final response, which improves performance but introduces additional alignment risks, with harmful content often appearing in both the CoTs and the final outputs. In this work, we investigate if we can use CoTs to predict final response misalignment. We evaluate a range of monitoring approaches, including humans, highly-capable large language models, and text classifiers, using either CoT text or activations. First, we find that a simple linear probe trained on CoT activations can significantly outperform all text-based methods in predicting whether a final response will be safe or unsafe. CoT texts are often unfaithful and can mislead humans and classifiers, while model latents (i.e., CoT activations) offer a more reliable predictive signal. Second, the probe makes accurate predictions before reasoning completes, achieving strong performance even when applied to early CoT segments. These findings generalize across model sizes, families, and safety benchmarks, suggesting that lightweight probes could enable real-time safety monitoring and early intervention during generation. |
| 2025-07-16 | [OD-VIRAT: A Large-Scale Benchmark for Object Detection in Realistic Surveillance Environments](http://arxiv.org/abs/2507.12396v1) | Hayat Ullah, Abbas Khan et al. | Realistic human surveillance datasets are crucial for training and evaluating computer vision models under real-world conditions, facilitating the development of robust algorithms for human and human-interacting object detection in complex environments. These datasets need to offer diverse and challenging data to enable a comprehensive assessment of model performance and the creation of more reliable surveillance systems for public safety. To this end, we present two visual object detection benchmarks named OD-VIRAT Large and OD-VIRAT Tiny, aiming at advancing visual understanding tasks in surveillance imagery. The video sequences in both benchmarks cover 10 different scenes of human surveillance recorded from significant height and distance. The proposed benchmarks offer rich annotations of bounding boxes and categories, where OD-VIRAT Large has 8.7 million annotated instances in 599,996 images and OD-VIRAT Tiny has 288,901 annotated instances in 19,860 images. This work also focuses on benchmarking state-of-the-art object detection architectures, including RETMDET, YOLOX, RetinaNet, DETR, and Deformable-DETR on this object detection-specific variant of VIRAT dataset. To the best of our knowledge, it is the first work to examine the performance of these recently published state-of-the-art object detection architectures on realistic surveillance imagery under challenging conditions such as complex backgrounds, occluded objects, and small-scale objects. The proposed benchmarking and experimental settings will help in providing insights concerning the performance of selected object detection models and set the base for developing more efficient and robust object detection architectures. |
| 2025-07-16 | [Thought Purity: Defense Paradigm For Chain-of-Thought Attack](http://arxiv.org/abs/2507.12314v1) | Zihao Xue, Zhen Bi et al. | While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures. |
| 2025-07-16 | [All-sky search for long-duration gravitational-wave transients in the first part of the fourth LIGO-Virgo-KAGRA Observing run](http://arxiv.org/abs/2507.12282v1) | The LIGO Scientific Collaboration, the Virgo Collaboration et al. | We present an all-sky search for long-duration gravitational waves (GWs) from the first part of the LIGO-Virgo-KAGRA fourth observing run (O4), called O4a and comprising data taken between 24 May 2023 and 16 January 2024. The GW signals targeted by this search are the so-called "long-duration" (> 1 s) transients expected from a variety of astrophysical processes, including non-axisymmetric deformations in magnetars or eccentric binary coalescences. We make minimal assumptions on the emitted GW waveforms in terms of morphologies and durations. Overall, our search targets signals with durations ~1-1000 s and frequency content in the range 16-2048 Hz. In the absence of significant detections, we report the sensitivity limits of our search in terms of root-sum-square signal amplitude (hrss) of reference waveforms. These limits improve upon the results from the third LIGO-Virgo-KAGRA observing run (O3) by about 30% on average. Moreover, this analysis demonstrates substantial progress in our ability to search for long-duration GW signals owing to enhancements in pipeline detection efficiencies. As detector sensitivities continue to advance and observational runs grow longer, unmodeled long-duration searches will increasingly be able to explore a range of compelling astrophysical scenarios involving neutron stars and black holes. |
| 2025-07-16 | [Exploiting Jailbreaking Vulnerabilities in Generative AI to Bypass Ethical Safeguards for Facilitating Phishing Attacks](http://arxiv.org/abs/2507.12185v1) | Rina Mishra, Gaurav Varshney | The advent of advanced Generative AI (GenAI) models such as DeepSeek and ChatGPT has significantly reshaped the cybersecurity landscape, introducing both promising opportunities and critical risks. This study investigates how GenAI powered chatbot services can be exploited via jailbreaking techniques to bypass ethical safeguards, enabling the generation of phishing content, recommendation of hacking tools, and orchestration of phishing campaigns. In ethically controlled experiments, we used ChatGPT 4o Mini selected for its accessibility and status as the latest publicly available model at the time of experimentation, as a representative GenAI system. Our findings reveal that the model could successfully guide novice users in executing phishing attacks across various vectors, including web, email, SMS (smishing), and voice (vishing). Unlike automated phishing campaigns that typically follow detectable patterns, these human-guided, AI assisted attacks are capable of evading traditional anti phishing mechanisms, thereby posing a growing security threat. We focused on DeepSeek and ChatGPT due to their widespread adoption and technical relevance in 2025. The study further examines common jailbreaking techniques and the specific vulnerabilities exploited in these models. Finally, we evaluate a range of mitigation strategies such as user education, advanced authentication mechanisms, and regulatory policy measures and discuss emerging trends in GenAI facilitated phishing, outlining future research directions to strengthen cybersecurity defenses in the age of artificial intelligence. |
| 2025-07-16 | [Probabilistic Safety Verification for an Autonomous Ground Vehicle: A Situation Coverage Grid Approach](http://arxiv.org/abs/2507.12158v1) | Nawshin Mannan Proma, Gricel Vázquez et al. | As industrial autonomous ground vehicles are increasingly deployed in safety-critical environments, ensuring their safe operation under diverse conditions is paramount. This paper presents a novel approach for their safety verification based on systematic situation extraction, probabilistic modelling and verification. We build upon the concept of a situation coverage grid, which exhaustively enumerates environmental configurations relevant to the vehicle's operation. This grid is augmented with quantitative probabilistic data collected from situation-based system testing, capturing probabilistic transitions between situations. We then generate a probabilistic model that encodes the dynamics of both normal and unsafe system behaviour. Safety properties extracted from hazard analysis and formalised in temporal logic are verified through probabilistic model checking against this model. The results demonstrate that our approach effectively identifies high-risk situations, provides quantitative safety guarantees, and supports compliance with regulatory standards, thereby contributing to the robust deployment of autonomous systems. |
| 2025-07-16 | [Topology Enhanced MARL for Multi-Vehicle Cooperative Decision-Making of CAVs](http://arxiv.org/abs/2507.12110v1) | Ye Han, Lijun Zhang et al. | The exploration-exploitation trade-off constitutes one of the fundamental challenges in reinforcement learning (RL), which is exacerbated in multi-agent reinforcement learning (MARL) due to the exponential growth of joint state-action spaces. This paper proposes a topology-enhanced MARL (TPE-MARL) method for optimizing cooperative decision-making of connected and autonomous vehicles (CAVs) in mixed traffic. This work presents two primary contributions: First, we construct a game topology tensor for dynamic traffic flow, effectively compressing high-dimensional traffic state information and decrease the search space for MARL algorithms. Second, building upon the designed game topology tensor and using QMIX as the backbone RL algorithm, we establish a topology-enhanced MARL framework incorporating visit counts and agent mutual information. Extensive simulations across varying traffic densities and CAV penetration rates demonstrate the effectiveness of TPE-MARL. Evaluations encompassing training dynamics, exploration patterns, macroscopic traffic performance metrics, and microscopic vehicle behaviors reveal that TPE-MARL successfully balances exploration and exploitation. Consequently, it exhibits superior performance in terms of traffic efficiency, safety, decision smoothness, and task completion. Furthermore, the algorithm demonstrates decision-making rationality comparable to or exceeding that of human drivers in both mixed-autonomy and fully autonomous traffic scenarios. Code of our work is available at \href{https://github.com/leoPub/tpemarl}{https://github.com/leoPub/tpemarl}. |
| 2025-07-16 | [Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](http://arxiv.org/abs/2507.12083v1) | Muleilan Pei, Shaoshuai Shi et al. | Motion forecasting for on-road traffic agents presents both a significant challenge and a critical necessity for ensuring safety in autonomous driving systems. In contrast to most existing data-driven approaches that directly predict future trajectories, we rethink this task from a planning perspective, advocating a "First Reasoning, Then Forecasting" strategy that explicitly incorporates behavior intentions as spatial guidance for trajectory prediction. To achieve this, we introduce an interpretable, reward-driven intention reasoner grounded in a novel query-centric Inverse Reinforcement Learning (IRL) scheme. Our method first encodes traffic agents and scene elements into a unified vectorized representation, then aggregates contextual features through a query-centric paradigm. This enables the derivation of a reward distribution, a compact yet informative representation of the target agent's behavior within the given scene context via IRL. Guided by this reward heuristic, we perform policy rollouts to reason about multiple plausible intentions, providing valuable priors for subsequent trajectory generation. Finally, we develop a hierarchical DETR-like decoder integrated with bidirectional selective state space models to produce accurate future trajectories along with their associated probabilities. Extensive experiments on the large-scale Argoverse and nuScenes motion forecasting datasets demonstrate that our approach significantly enhances trajectory prediction confidence, achieving highly competitive performance relative to state-of-the-art methods. |
| 2025-07-16 | [Towards Ultra-Reliable 6G in-X Subnetworks: Dynamic Link Adaptation by Deep Reinforcement Learning](http://arxiv.org/abs/2507.12031v1) | Fateme Salehi, Aamir Mahmood et al. | 6G networks are composed of subnetworks expected to meet ultra-reliable low-latency communication (URLLC) requirements for mission-critical applications such as industrial control and automation. An often-ignored aspect in URLLC is consecutive packet outages, which can destabilize control loops and compromise safety in in-factory environments. Hence, the current work proposes a link adaptation framework to support extreme reliability requirements using the soft actor-critic (SAC)-based deep reinforcement learning (DRL) algorithm that jointly optimizes energy efficiency (EE) and reliability under dynamic channel and interference conditions. Unlike prior work focusing on average reliability, our method explicitly targets reducing burst/consecutive outages through adaptive control of transmit power and blocklength based solely on the observed signal-to-interference-plus-noise ratio (SINR). The joint optimization problem is formulated under finite blocklength and quality of service constraints, balancing reliability and EE. Simulation results show that the proposed method significantly outperforms the baseline algorithms, reducing outage bursts while consuming only 18\% of the transmission cost required by a full/maximum resource allocation policy in the evaluated scenario. The framework also supports flexible trade-off tuning between EE and reliability by adjusting reward weights, making it adaptable to diverse industrial requirements. |
| 2025-07-16 | [Robust Planning for Autonomous Vehicles with Diffusion-Based Failure Samplers](http://arxiv.org/abs/2507.11991v1) | Juanran Wang, Marc R. Schlichting et al. | High-risk traffic zones such as intersections are a major cause of collisions. This study leverages deep generative models to enhance the safety of autonomous vehicles in an intersection context. We train a 1000-step denoising diffusion probabilistic model to generate collision-causing sensor noise sequences for an autonomous vehicle navigating a four-way intersection based on the current relative position and velocity of an intruder. Using the generative adversarial architecture, the 1000-step model is distilled into a single-step denoising diffusion model which demonstrates fast inference speed while maintaining similar sampling quality. We demonstrate one possible application of the single-step model in building a robust planner for the autonomous vehicle. The planner uses the single-step model to efficiently sample potential failure cases based on the currently measured traffic state to inform its decision-making. Through simulation experiments, the robust planner demonstrates significantly lower failure rate and delay rate compared with the baseline Intelligent Driver Model controller. |
| 2025-07-16 | [Formal Verification of Neural Certificates Done Dynamically](http://arxiv.org/abs/2507.11987v1) | Thomas A. Henzinger, Konstantin Kueffner et al. | Neural certificates have emerged as a powerful tool in cyber-physical systems control, providing witnesses of correctness. These certificates, such as barrier functions, often learned alongside control policies, once verified, serve as mathematical proofs of system safety. However, traditional formal verification of their defining conditions typically faces scalability challenges due to exhaustive state-space exploration. To address this challenge, we propose a lightweight runtime monitoring framework that integrates real-time verification and does not require access to the underlying control policy. Our monitor observes the system during deployment and performs on-the-fly verification of the certificate over a lookahead region to ensure safety within a finite prediction horizon. We instantiate this framework for ReLU-based control barrier functions and demonstrate its practical effectiveness in a case study. Our approach enables timely detection of safety violations and incorrect certificates with minimal overhead, providing an effective but lightweight alternative to the static verification of the certificates. |
| 2025-07-16 | [Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation](http://arxiv.org/abs/2507.11968v1) | Sahid Hossain Mustakim, S M Jishanul Islam et al. | Multimodal Large Language Models (MLLMs) are increasingly used for content moderation, yet their robustness in short-form video contexts remains underexplored. Current safety evaluations often rely on unimodal attacks, failing to address combined attack vulnerabilities. In this paper, we introduce a comprehensive framework for evaluating the tri-modal safety of MLLMs. First, we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising diverse short-form videos with human-guided synthetic adversarial attacks. Second, we propose ChimeraBreak, a novel tri-modal attack strategy that simultaneously challenges visual, auditory, and semantic reasoning pathways. Extensive experiments on state-of-the-art MLLMs reveal significant vulnerabilities with high Attack Success Rates (ASR). Our findings uncover distinct failure modes, showing model biases toward misclassifying benign or policy-violating content. We assess results using LLM-as-a-judge, demonstrating attack reasoning efficacy. Our dataset and findings provide crucial insights for developing more robust and safe MLLMs. |
| 2025-07-16 | [Toxicity-Aware Few-Shot Prompting for Low-Resource Singlish Translation](http://arxiv.org/abs/2507.11966v1) | Ziyu Ge, Gabriel Chua et al. | As online communication increasingly incorporates under-represented languages and colloquial dialects, standard translation systems often fail to preserve local slang, code-mixing, and culturally embedded markers of harmful speech. Translating toxic content between low-resource language pairs poses additional challenges due to scarce parallel data and safety filters that sanitize offensive expressions. In this work, we propose a reproducible, two-stage framework for toxicity-preserving translation, demonstrated on a code-mixed Singlish safety corpus. First, we perform human-verified few-shot prompt engineering: we iteratively curate and rank annotator-selected Singlish-target examples to capture nuanced slang, tone, and toxicity. Second, we optimize model-prompt pairs by benchmarking several large language models using semantic similarity via direct and back-translation. Quantitative human evaluation confirms the effectiveness and efficiency of our pipeline. Beyond improving translation quality, our framework contributes to the safety of multicultural LLMs by supporting culturally sensitive moderation and benchmarking in low-resource contexts. By positioning Singlish as a testbed for inclusive NLP, we underscore the importance of preserving sociolinguistic nuance in real-world applications such as content moderation and regional platform governance. |
| 2025-07-16 | [Parallel-plate chambers as radiation-hard detectors for time-based beam diagnostics in carbon-ion radiotherapy](http://arxiv.org/abs/2507.11963v1) | Na Hye Kwon, Sung Woon Choi et al. | Accurate range verification of carbon ion beams is critical for the precision and safety of charged particle radiotherapy. In this study, we evaluated the feasibility of using a parallel-plate ionization chamber for real-time, time-based diagnostic monitoring of carbon ion beams. The chamber featured a 0.4 mm gas gap defined by metallic electrodes and was filled with carbon dioxide (CO$_2$), a non-polymerizing gas suitable for high-rate applications. Timing precision was assessed via self-correlation analysis, yielding a precision approaching one picosecond for one-second acquisitions under clinically relevant beam conditions. This level of timing accuracy translates to a water-equivalent range uncertainty of approximately 1 mm, which meets the recommended clinical tolerance for carbon ion therapy. Furthermore, the kinetic energy of the beam at the synchrotron extraction point was determined from the measured orbital period, with results consistently within 1 MeV/nucleon of the nominal energy. These findings demonstrate the potential of parallel-plate chambers for precise, real-time energy and range verification in clinical carbon ion beam quality assurance. |
| 2025-07-16 | [Hybrid Conformal Prediction-based Risk-Aware Model Predictive Planning in Dense, Uncertain Environments](http://arxiv.org/abs/2507.11920v1) | Jeongyong Yang, KwangBin Lee et al. | Real-time path planning in dense, uncertain environments remains a challenging problem, as predicting the future motions of numerous dynamic obstacles is computationally burdensome and unrealistic. To address this, we introduce Hybrid Prediction-based Risk-Aware Planning (HyPRAP), a prediction-based risk-aware path-planning framework which uses a hybrid combination of models to predict local obstacle movement. HyPRAP uses a novel Prediction-based Collision Risk Index (P-CRI) to evaluate the risk posed by each obstacle, enabling the selective use of predictors based on whether the agent prioritizes high predictive accuracy or low computational prediction overhead. This selective routing enables the agent to focus on high-risk obstacles while ignoring or simplifying low-risk ones, making it suitable for environments with a large number of obstacles. Moreover, HyPRAP incorporates uncertainty quantification through hybrid conformal prediction by deriving confidence bounds simultaneously achieved by multiple predictions across different models. Theoretical analysis demonstrates that HyPRAP effectively balances safety and computational efficiency by leveraging the diversity of prediction models. Extensive simulations validate these insights for more general settings, confirming that HyPRAP performs better compared to single predictor methods, and P-CRI performs better over naive proximity-based risk assessment. |
| 2025-07-16 | [SEPose: A Synthetic Event-based Human Pose Estimation Dataset for Pedestrian Monitoring](http://arxiv.org/abs/2507.11910v1) | Kaustav Chanda, Aayush Atul Verma et al. | Event-based sensors have emerged as a promising solution for addressing challenging conditions in pedestrian and traffic monitoring systems. Their low-latency and high dynamic range allow for improved response time in safety-critical situations caused by distracted walking or other unusual movements. However, the availability of data covering such scenarios remains limited. To address this gap, we present SEPose -- a comprehensive synthetic event-based human pose estimation dataset for fixed pedestrian perception generated using dynamic vision sensors in the CARLA simulator. With nearly 350K annotated pedestrians with body pose keypoints from the perspective of fixed traffic cameras, SEPose is a comprehensive synthetic multi-person pose estimation dataset that spans busy and light crowds and traffic across diverse lighting and weather conditions in 4-way intersections in urban, suburban, and rural environments. We train existing state-of-the-art models such as RVT and YOLOv8 on our dataset and evaluate them on real event-based data to demonstrate the sim-to-real generalization capabilities of the proposed dataset. |
| 2025-07-16 | [LLMs Encode Harmfulness and Refusal Separately](http://arxiv.org/abs/2507.11878v1) | Jiachen Zhao, Jing Huang et al. | LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety |

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



