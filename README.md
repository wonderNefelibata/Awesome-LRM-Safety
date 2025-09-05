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
| 2025-09-04 | [SAFE--MA--RRT: Multi-Agent Motion Planning with Data-Driven Safety Certificates](http://arxiv.org/abs/2509.04413v1) | Babak Esmaeili, Hamidreza Modares | This paper proposes a fully data-driven motion-planning framework for homogeneous linear multi-agent systems that operate in shared, obstacle-filled workspaces without access to explicit system models. Each agent independently learns its closed-loop behavior from experimental data by solving convex semidefinite programs that generate locally invariant ellipsoids and corresponding state-feedback gains. These ellipsoids, centered along grid-based waypoints, certify the dynamic feasibility of short-range transitions and define safe regions of operation. A sampling-based planner constructs a tree of such waypoints, where transitions are allowed only when adjacent ellipsoids overlap, ensuring invariant-to-invariant transitions and continuous safety. All agents expand their trees simultaneously and are coordinated through a space-time reservation table that guarantees inter-agent safety by preventing simultaneous occupancy and head-on collisions. Each successful edge in the tree is equipped with its own local controller, enabling execution without re-solving optimization problems at runtime. The resulting trajectories are not only dynamically feasible but also provably safe with respect to both environmental constraints and inter-agent collisions. Simulation results demonstrate the effectiveness of the approach in synthesizing synchronized, safe trajectories for multiple agents under shared dynamics and constraints, using only data and convex optimization tools. |
| 2025-09-04 | [Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios](http://arxiv.org/abs/2509.04403v1) | Jingen Qu, Lijun Li et al. | Multimodal large language models (MLLMs) are rapidly evolving, presenting increasingly complex safety challenges. However, current dataset construction methods, which are risk-oriented, fail to cover the growing complexity of real-world multimodal safety scenarios (RMS). And due to the lack of a unified evaluation metric, their overall effectiveness remains unproven. This paper introduces a novel image-oriented self-adaptive dataset construction method for RMS, which starts with images and end constructing paired text and guidance responses. Using the image-oriented method, we automatically generate an RMS dataset comprising 35k image-text pairs with guidance responses. Additionally, we introduce a standardized safety dataset evaluation metric: fine-tuning a safety judge model and evaluating its capabilities on other safety datasets.Extensive experiments on various tasks demonstrate the effectiveness of the proposed image-oriented pipeline. The results confirm the scalability and effectiveness of the image-oriented approach, offering a new perspective for the construction of real-world multimodal safety datasets. |
| 2025-09-04 | [AnomalyLMM: Bridging Generative Knowledge and Discriminative Retrieval for Text-Based Person Anomaly Search](http://arxiv.org/abs/2509.04376v1) | Hao Ju, Hu Zhang et al. | With growing public safety demands, text-based person anomaly search has emerged as a critical task, aiming to retrieve individuals with abnormal behaviors via natural language descriptions. Unlike conventional person search, this task presents two unique challenges: (1) fine-grained cross-modal alignment between textual anomalies and visual behaviors, and (2) anomaly recognition under sparse real-world samples. While Large Multi-modal Models (LMMs) excel in multi-modal understanding, their potential for fine-grained anomaly retrieval remains underexplored, hindered by: (1) a domain gap between generative knowledge and discriminative retrieval, and (2) the absence of efficient adaptation strategies for deployment. In this work, we propose AnomalyLMM, the first framework that harnesses LMMs for text-based person anomaly search. Our key contributions are: (1) A novel coarse-to-fine pipeline integrating LMMs to bridge generative world knowledge with retrieval-centric anomaly detection; (2) A training-free adaptation cookbook featuring masked cross-modal prompting, behavioral saliency prediction, and knowledge-aware re-ranking, enabling zero-shot focus on subtle anomaly cues. As the first study to explore LMMs for this task, we conduct a rigorous evaluation on the PAB dataset, the only publicly available benchmark for text-based person anomaly search, with its curated real-world anomalies covering diverse scenarios (e.g., falling, collision, and being hit). Experiments show the effectiveness of the proposed method, surpassing the competitive baseline by +0.96% Recall@1 accuracy. Notably, our method reveals interpretable alignment between textual anomalies and visual behaviors, validated via qualitative analysis. Our code and models will be released for future research. |
| 2025-09-04 | [Reinforcement Learning for Robust Ageing-Aware Control of Li-ion Battery Systems with Data-Driven Formal Verification](http://arxiv.org/abs/2509.04288v1) | Rudi Coppola, Hovsep Touloujian et al. | Rechargeable lithium-ion (Li-ion) batteries are a ubiquitous element of modern technology. In the last decades, the production and design of such batteries and their adjacent embedded charging and safety protocols, denoted by Battery Management Systems (BMS), has taken central stage. A fundamental challenge to be addressed is the trade-off between the speed of charging and the ageing behavior, resulting in the loss of capacity in the battery cell. We rely on a high-fidelity physics-based battery model and propose an approach to data-driven charging and safety protocol design. Following a Counterexample-Guided Inductive Synthesis scheme, we combine Reinforcement Learning (RL) with recent developments in data-driven formal methods to obtain a hybrid control strategy: RL is used to synthesise the individual controllers, and a data-driven abstraction guides their partitioning into a switched structure, depending on the initial output measurements of the battery. The resulting discrete selection among RL-based controllers, coupled with the continuous battery dynamics, realises a hybrid system. When a design meets the desired criteria, the abstraction provides probabilistic guarantees on the closed-loop performance of the cell. |
| 2025-09-04 | [When Lifetimes Liberate: A Type System for Arenas with Higher-Order Reachability Tracking](http://arxiv.org/abs/2509.04253v1) | Siyuan He, Songlin Jia et al. | Static resource management in higher-order functional languages remains elusive due to tensions between control, expressiveness, and flexibility. Region-based systems [Grossman et al. 2002; Tofte et al. 2001] offer control over lifetimes and expressive in-region sharing, but restrict resources to lexical scopes. Rust, an instance of ownership types [Clarke et al. 2013], offers non-lexical lifetimes and robust safety guarantees, yet its global invariants make common sharing patterns hard to express. Reachability types [Wei et al. 2024] enable reasoning about sharing and separation, but lack practical tools for controlling resource lifetimes.   In this work, we try to unify their strengths. Our solution enables grouping resources as arenas for arbitrary sharing and static guarantees of lexically scoped lifetimes. Crucially, arenas and lexical lifetimes are not the only choice: users may also manage resources individually, with non-lexical lifetimes. Regardless of mode, resources share the same type, preserving the higher-order parametric nature of the language.   Obtaining static safety guarantee in a higher-order language with flexible sharing is nontrivial. To this end, we propose two new extensions atop reachability types [Wei et al. 2024]. First, A<: features a novel two-dimensional store model to enable coarse-grained reachability tracking for arbitrarily shared resources within arenas. Building on this, {A}<: establishes lexical lifetime control with static guarantees. As the first reachability formalism presented for lifetime control, {A}<: avoids the complication of flow-sensitive reasoning and retains expressive power and simplicity. Both calculi are formalized and proven type safe in Rocq. |
| 2025-09-04 | [How many patients could we save with LLM priors?](http://arxiv.org/abs/2509.04250v1) | Shota Arai, David Selby et al. | Imagine a world where clinical trials need far fewer patients to achieve the same statistical power, thanks to the knowledge encoded in large language models (LLMs). We present a novel framework for hierarchical Bayesian modeling of adverse events in multi-center clinical trials, leveraging LLM-informed prior distributions. Unlike data augmentation approaches that generate synthetic data points, our methodology directly obtains parametric priors from the model. Our approach systematically elicits informative priors for hyperparameters in hierarchical Bayesian models using a pre-trained LLM, enabling the incorporation of external clinical expertise directly into Bayesian safety modeling. Through comprehensive temperature sensitivity analysis and rigorous cross-validation on real-world clinical trial data, we demonstrate that LLM-derived priors consistently improve predictive performance compared to traditional meta-analytical approaches. This methodology paves the way for more efficient and expert-informed clinical trial design, enabling substantial reductions in the number of patients required to achieve robust safety assessment and with the potential to transform drug safety monitoring and regulatory decision making. |
| 2025-09-04 | [Learning Active Perception via Self-Evolving Preference Optimization for GUI Grounding](http://arxiv.org/abs/2509.04243v1) | Wanfu Wang, Qipeng Huang et al. | Vision Language Models (VLMs) have recently achieved significant progress in bridging visual perception and linguistic reasoning. Recently, OpenAI o3 model introduced a zoom-in search strategy that effectively elicits active perception capabilities in VLMs, improving downstream task performance. However, enabling VLMs to reason effectively over appropriate image regions remains a core challenge in GUI grounding, particularly under high-resolution inputs and complex multi-element visual interactions. In this work, we propose LASER, a self-evolving framework that progressively endows VLMs with multi-step perception capabilities, enabling precise coordinate prediction. Specifically, our approach integrate Monte Carlo quality estimation with Intersection-over-Union (IoU)-based region quality evaluation to jointly encourage both accuracy and diversity in constructing high-quality preference data. This combination explicitly guides the model to focus on instruction-relevant key regions while adaptively allocating reasoning steps based on task complexity. Comprehensive experiments on the ScreenSpot Pro and ScreenSpot-v2 benchmarks demonstrate consistent performance gains, validating the effectiveness of our method. Furthermore, when fine-tuned on GTA1-7B, LASER achieves a score of 55.7 on the ScreenSpot-Pro benchmark, establishing a new state-of-the-art (SoTA) among 7B-scale models. |
| 2025-09-04 | [Compatibility of Multiple Control Barrier Functions for Constrained Nonlinear Systems](http://arxiv.org/abs/2509.04220v1) | Max H. Cohen, Eugene Lavretsky et al. | Control barrier functions (CBFs) are a powerful tool for the constrained control of nonlinear systems; however, the majority of results in the literature focus on systems subject to a single CBF constraint, making it challenging to synthesize provably safe controllers that handle multiple state constraints. This paper presents a framework for constrained control of nonlinear systems subject to box constraints on the systems' vector-valued outputs using multiple CBFs. Our results illustrate that when the output has a vector relative degree, the CBF constraints encoding these box constraints are compatible, and the resulting optimization-based controller is locally Lipschitz continuous and admits a closed-form expression. Additional results are presented to characterize the degradation of nominal tracking objectives in the presence of safety constraints. Simulations of a planar quadrotor are presented to demonstrate the efficacy of the proposed framework. |
| 2025-09-04 | [Real-time Object Detection and Associated Hardware Accelerators Targeting Autonomous Vehicles: A Review](http://arxiv.org/abs/2509.04173v1) | Safa Sali, Anis Meribout et al. | The efficiency of object detectors depends on factors like detection accuracy, processing time, and computational resources. Processing time is crucial for real-time applications, particularly for autonomous vehicles (AVs), where instantaneous responses are vital for safety. This review paper provides a concise yet comprehensive survey of real-time object detection (OD) algorithms for autonomous cars delving into their hardware accelerators (HAs). Non-neural network-based algorithms, which use statistical image processing, have been entirely substituted by AI algorithms, such as different models of convolutional neural networks (CNNs). Their intrinsically parallel features led them to be deployable into edge-based HAs of various types, where GPUs and, to a lesser extent, ASIC (application-specific integrated circuit) remain the most widely used. Throughputs of hundreds of frames/s (fps) could be reached; however, handling object detection for all the cameras available in a typical AV requires further hardware and algorithmic improvements. The intensive competition between AV providers has limited the disclosure of algorithms, firmware, and even hardware platform details. This remains a hurdle for researchers, as commercial systems provide valuable insights while academics undergo lengthy training and testing on restricted datasets and road scenarios. Consequently, many AV research papers may not be reflected in end products, being developed under limited conditions. This paper surveys state-of-the-art OD algorithms and aims to bridge the gap with technologies in commercial AVs. To our knowledge, this aspect has not been addressed in earlier surveys. Hence, the paper serves as a tangible reference for researchers designing future generations of vehicles, expected to be fully autonomous for comfort and safety. |
| 2025-09-04 | [Active Dual-Gated Graphene Transistors for Low-Noise, Drift-Stable, and Tunable Chemical Sensing](http://arxiv.org/abs/2509.04137v1) | Vinay Kammarchedu, Heshmat Asgharian et al. | Graphene field-effect transistors (GFETs) are among the most promising platforms for ultrasensitive chemical and biological sensing due to their high carrier mobility, large surface area, and low intrinsic noise. However, conventional single-gate GFET sensors in liquid environments suffer from severe limitations, including signal drift, charge trapping, and insufficient signal amplification. Here, we introduce a dual-gate GFET architecture that integrates a high-k hafnium dioxide local back gate with an electrolyte top gate, coupled with real-time feedback biasing. This design enables capacitive signal amplification while simultaneously suppressing gate leakage and low-frequency noise. By systematically evaluating seven distinct operational modes, we identify the Dual Mode Fixed configuration as optimal, achieving up to 20x signal gain, > 15x lower drift compared with gate-swept methods, and up to 7x higher signal to noise ratio across a diverse range of analytes, including neurotransmitters, volatile organic compounds, environmental contaminants, and proteins. We further demonstrate robust, multiplexed detection using a PCB-integrated GFET sensor array, underscoring the scalability and practicality of the platform for portable, high-throughput sensing in complex environments. Together, these advances establish a versatile and stable sensing technology capable of real-time, label-free detection of molecular targets under ambient and physiological conditions, with broad applicability in health monitoring, food safety, agriculture, and environmental screening. |
| 2025-09-04 | [DVS-PedX: Synthetic-and-Real Event-Based Pedestrian Dataset](http://arxiv.org/abs/2509.04117v1) | Mustafa Sakhai, Kaung Sithu et al. | Event cameras like Dynamic Vision Sensors (DVS) report micro-timed brightness changes instead of full frames, offering low latency, high dynamic range, and motion robustness. DVS-PedX (Dynamic Vision Sensor Pedestrian eXploration) is a neuromorphic dataset designed for pedestrian detection and crossing-intention analysis in normal and adverse weather conditions across two complementary sources: (1) synthetic event streams generated in the CARLA simulator for controlled "approach-cross" scenes under varied weather and lighting; and (2) real-world JAAD dash-cam videos converted to event streams using the v2e tool, preserving natural behaviors and backgrounds. Each sequence includes paired RGB frames, per-frame DVS "event frames" (33 ms accumulations), and frame-level labels (crossing vs. not crossing). We also provide raw AEDAT 2.0/AEDAT 4.0 event files and AVI DVS video files and metadata for flexible re-processing. Baseline spiking neural networks (SNNs) using SpikingJelly illustrate dataset usability and reveal a sim-to-real gap, motivating domain adaptation and multimodal fusion. DVS-PedX aims to accelerate research in event-based pedestrian safety, intention prediction, and neuromorphic perception. |
| 2025-09-04 | [Low-Power Impact Detection and Localization on Forklifts Using Wireless IMU Sensors](http://arxiv.org/abs/2509.04096v1) | Lyssa Ramaut, Chesney Buyle et al. | Forklifts are essential for transporting goods in industrial environments. These machines face wear and tear during field operations, along with rough terrain, tight spaces and complex handling scenarios. This increases the likelihood of unintended impacts, such as collisions with goods, infrastructure, or other machinery. In addition, deliberate misuse has been stated, compromising safety and equipment integrity. This paper presents a low-cost and low-power impact detection system based on multiple wireless sensor nodes measuring 3D accelerations. These were deployed in a measurement campaign covering realworld operational scenarios. An algorithm was developed, based on this collected data, to differentiate high-impact events from normal usage and to localize detected collisions on the forklift. The solution successfully detects and localizes impacts, while maintaining low power consumption, enabling reliable forklift monitoring with multi-year sensor autonomy. |
| 2025-09-04 | [TriLiteNet: Lightweight Model for Multi-Task Visual Perception](http://arxiv.org/abs/2509.04092v1) | Quang-Huy Che, Duc-Khai Lam | Efficient perception models are essential for Advanced Driver Assistance Systems (ADAS), as these applications require rapid processing and response to ensure safety and effectiveness in real-world environments. To address the real-time execution needs of such perception models, this study introduces the TriLiteNet model. This model can simultaneously manage multiple tasks related to panoramic driving perception. TriLiteNet is designed to optimize performance while maintaining low computational costs. Experimental results on the BDD100k dataset demonstrate that the model achieves competitive performance across three key tasks: vehicle detection, drivable area segmentation, and lane line segmentation. Specifically, the TriLiteNet_{base} demonstrated a recall of 85.6% for vehicle detection, a mean Intersection over Union (mIoU) of 92.4% for drivable area segmentation, and an Acc of 82.3% for lane line segmentation with only 2.35M parameters and a computational cost of 7.72 GFLOPs. Our proposed model includes a tiny configuration with just 0.14M parameters, which provides a multi-task solution with minimal computational demand. Evaluated for latency and power consumption on embedded devices, TriLiteNet in both configurations shows low latency and reasonable power during inference. By balancing performance, computational efficiency, and scalability, TriLiteNet offers a practical and deployable solution for real-world autonomous driving applications. Code is available at https://github.com/chequanghuy/TriLiteNet. |
| 2025-09-04 | [ICSLure: A Very High Interaction Honeynet for PLC-based Industrial Control Systems](http://arxiv.org/abs/2509.04080v1) | Francesco Aurelio Pironti, Angelo Furfaro et al. | The security of Industrial Control Systems (ICSs) is critical to ensuring the safety of industrial processes and personnel. The rapid adoption of Industrial Internet of Things (IIoT) technologies has expanded system functionality but also increased the attack surface, exposing ICSs to a growing range of cyber threats. Honeypots provide a means to detect and analyze such threats by emulating target systems and capturing attacker behavior. However, traditional ICS honeypots, often limited to software-based simulations of a single Programmable Logic Controller (PLC), lack the realism required to engage sophisticated adversaries. In this work, we introduce a modular honeynet framework named ICSLure. The framework has been designed to emulate realistic ICS environments. Our approach integrates physical PLCs interacting with live data sources via industrial protocols such as Modbus and Profinet RTU, along with virtualized network components including routers, switches, and Remote Terminal Units (RTUs). The system incorporates comprehensive monitoring capabilities to collect detailed logs of attacker interactions. We demonstrate that our framework enables coherent and high-fidelity emulation of real-world industrial plants. This high-interaction environment significantly enhances the quality of threat data collected and supports advanced analysis of ICS-specific attack strategies, contributing to more effective detection and mitigation techniques. |
| 2025-09-04 | [Integrated Wheel Sensor Communication using ESP32 -- A Contribution towards a Digital Twin of the Road System](http://arxiv.org/abs/2509.04061v1) | Ventseslav Yordanov, Simon Schäfer et al. | While current onboard state estimation methods are adequate for most driving and safety-related applications, they do not provide insights into the interaction between tires and road surfaces. This paper explores a novel communication concept for efficiently transmitting integrated wheel sensor data from an ESP32 microcontroller. Our proposed approach utilizes a publish-subscribe system, surpassing comparable solutions in the literature regarding data transmission volume. We tested this approach on a drum tire test rig with our prototype sensors system utilizing a diverse selection of sample frequencies between 1 Hz and 32 000 Hz to demonstrate the efficacy of our communication concept. The implemented prototype sensor showcases minimal data loss, approximately 0.1 % of the sampled data, validating the reliability of our developed communication system. This work contributes to advancing real-time data acquisition, providing insights into optimizing integrated wheel sensor communication. |
| 2025-09-04 | [Safeguarding Patient Trust in the Age of AI: Tackling Health Misinformation with Explainable AI](http://arxiv.org/abs/2509.04052v1) | Sueun Hong, Shuojie Fu et al. | AI-generated health misinformation poses unprecedented threats to patient safety and healthcare system trust globally. This white paper presents an explainable AI framework developed through the EPSRC INDICATE project to combat medical misinformation while enhancing evidence-based healthcare delivery. Our systematic review of 17 studies reveals the urgent need for transparent AI systems in healthcare. The proposed solution demonstrates 95% recall in clinical evidence retrieval and integrates novel trustworthiness classifiers achieving 76% F1 score in detecting biomedical misinformation. Results show that explainable AI can transform traditional 6-month expert review processes into real-time, automated evidence synthesis while maintaining clinical rigor. This approach offers a critical intervention to preserve healthcare integrity in the AI era. |
| 2025-09-04 | [NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models](http://arxiv.org/abs/2509.03985v1) | Chuhan Zhang, Ye Zhang et al. | In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks. |
| 2025-09-04 | [Sample Efficient Certification of Discrete-Time Control Barrier Functions](http://arxiv.org/abs/2509.03899v1) | Sampath Kumar Mulagaleti, Andrea Del Prete | Control Invariant (CI) sets are instrumental in certifying the safety of dynamical systems. Control Barrier Functions (CBFs) are effective tools to compute such sets, since the zero sublevel sets of CBFs are CI sets. However, computing CBFs generally involves addressing a complex robust optimization problem, which can be intractable. Scenario-based methods have been proposed to simplify this computation. Then, one needs to verify if the CBF actually satisfies the robust constraints. We present an approach to perform this verification that relies on Lipschitz arguments, and forms the basis of a certification algorithm designed for sample efficiency. Through a numerical example, we validated the efficiency of the proposed procedure. |
| 2025-09-04 | [False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize](http://arxiv.org/abs/2509.03888v1) | Cheng Wang, Zeming Wei et al. | Large Language Models (LLMs) can comply with harmful instructions, raising serious safety concerns despite their impressive capabilities. Recent work has leveraged probing-based approaches to study the separability of malicious and benign inputs in LLMs' internal representations, and researchers have proposed using such probing methods for safety detection. We systematically re-examine this paradigm. Motivated by poor out-of-distribution performance, we hypothesize that probes learn superficial patterns rather than semantic harmfulness. Through controlled experiments, we confirm this hypothesis and identify the specific patterns learned: instructional patterns and trigger words. Our investigation follows a systematic approach, progressing from demonstrating comparable performance of simple n-gram methods, to controlled experiments with semantically cleaned datasets, to detailed analysis of pattern dependencies. These results reveal a false sense of security around current probing-based approaches and highlight the need to redesign both models and evaluation protocols, for which we provide further discussions in the hope of suggesting responsible further research in this direction. We have open-sourced the project at https://github.com/WangCheng0116/Why-Probe-Fails. |
| 2025-09-04 | [A Comprehensive Survey on Trustworthiness in Reasoning with Large Language Models](http://arxiv.org/abs/2509.03871v1) | Yanbo Wang, Yongcan Yu et al. | The development of Long-CoT reasoning has advanced LLM performance across various tasks, including language understanding, complex problem solving, and code generation. This paradigm enables models to generate intermediate reasoning steps, thereby improving both accuracy and interpretability. However, despite these advancements, a comprehensive understanding of how CoT-based reasoning affects the trustworthiness of language models remains underdeveloped. In this paper, we survey recent work on reasoning models and CoT techniques, focusing on five core dimensions of trustworthy reasoning: truthfulness, safety, robustness, fairness, and privacy. For each aspect, we provide a clear and structured overview of recent studies in chronological order, along with detailed analyses of their methodologies, findings, and limitations. Future research directions are also appended at the end for reference and discussion. Overall, while reasoning techniques hold promise for enhancing model trustworthiness through hallucination mitigation, harmful content detection, and robustness improvement, cutting-edge reasoning models themselves often suffer from comparable or even greater vulnerabilities in safety, robustness, and privacy. By synthesizing these insights, we hope this work serves as a valuable and timely resource for the AI safety community to stay informed on the latest progress in reasoning trustworthiness. A full list of related papers can be found at \href{https://github.com/ybwang119/Awesome-reasoning-safety}{https://github.com/ybwang119/Awesome-reasoning-safety}. |

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



