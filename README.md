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
| 2025-09-08 | [From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers](http://arxiv.org/abs/2509.06938v1) | Praneet Suresh, Jack Stanley et al. | As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk. |
| 2025-09-08 | [Tackling the Noisy Elephant in the Room: Label Noise-robust Out-of-Distribution Detection via Loss Correction and Low-rank Decomposition](http://arxiv.org/abs/2509.06918v1) | Tarhib Al Azad, Shahana Ibrahim | Robust out-of-distribution (OOD) detection is an indispensable component of modern artificial intelligence (AI) systems, especially in safety-critical applications where models must identify inputs from unfamiliar classes not seen during training. While OOD detection has been extensively studied in the machine learning literature--with both post hoc and training-based approaches--its effectiveness under noisy training labels remains underexplored. Recent studies suggest that label noise can significantly degrade OOD performance, yet principled solutions to this issue are lacking. In this work, we demonstrate that directly combining existing label noise-robust methods with OOD detection strategies is insufficient to address this critical challenge. To overcome this, we propose a robust OOD detection framework that integrates loss correction techniques from the noisy label learning literature with low-rank and sparse decomposition methods from signal processing. Extensive experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms the state-of-the-art OOD detection techniques, particularly under severe noisy label settings. |
| 2025-09-08 | [mmBERT: A Modern Multilingual Encoder with Annealed Language Learning](http://arxiv.org/abs/2509.06888v1) | Marc Marone, Orion Weller et al. | Encoder-only languages models are frequently used for a variety of standard machine learning tasks, including classification and retrieval. However, there has been a lack of recent research for encoder models, especially with respect to multilingual models. We introduce mmBERT, an encoder-only language model pretrained on 3T tokens of multilingual text in over 1800 languages. To build mmBERT we introduce several novel elements, including an inverse mask ratio schedule and an inverse temperature sampling ratio. We add over 1700 low-resource languages to the data mix only during the decay phase, showing that it boosts performance dramatically and maximizes the gains from the relatively small amount of training data. Despite only including these low-resource languages in the short decay phase we achieve similar classification performance to models like OpenAI's o3 and Google's Gemini 2.5 Pro. Overall, we show that mmBERT significantly outperforms the previous generation of models on classification and retrieval tasks -- on both high and low-resource languages. |
| 2025-09-08 | [EPT Benchmark: Evaluation of Persian Trustworthiness in Large Language Models](http://arxiv.org/abs/2509.06838v1) | Mohammad Reza Mirbagheri, Mohammad Mahdi Mirkamali et al. | Large Language Models (LLMs), trained on extensive datasets using advanced deep learning architectures, have demonstrated remarkable performance across a wide range of language tasks, becoming a cornerstone of modern AI technologies. However, ensuring their trustworthiness remains a critical challenge, as reliability is essential not only for accurate performance but also for upholding ethical, cultural, and social values. Careful alignment of training data and culturally grounded evaluation criteria are vital for developing responsible AI systems. In this study, we introduce the EPT (Evaluation of Persian Trustworthiness) metric, a culturally informed benchmark specifically designed to assess the trustworthiness of LLMs across six key aspects: truthfulness, safety, fairness, robustness, privacy, and ethical alignment. We curated a labeled dataset and evaluated the performance of several leading models - including ChatGPT, Claude, DeepSeek, Gemini, Grok, LLaMA, Mistral, and Qwen - using both automated LLM-based and human assessments. Our results reveal significant deficiencies in the safety dimension, underscoring the urgent need for focused attention on this critical aspect of model behavior. Furthermore, our findings offer valuable insights into the alignment of these models with Persian ethical-cultural values and highlight critical gaps and opportunities for advancing trustworthy and culturally responsible AI. The dataset is publicly available at: https://github.com/Rezamirbagheri110/EPT-Benchmark. |
| 2025-09-08 | [Anchoring Refusal Direction: Mitigating Safety Risks in Tuning via Projection Constraint](http://arxiv.org/abs/2509.06795v1) | Yanrui Du, Fenglei Fan et al. | Instruction Fine-Tuning (IFT) has been widely adopted as an effective post-training strategy to enhance various abilities of Large Language Models (LLMs). However, prior studies have shown that IFT can significantly compromise LLMs' safety, particularly their ability to refuse malicious instructions, raising significant concerns. Recent research into the internal mechanisms of LLMs has identified the refusal direction (r-direction) in the hidden states, which plays a pivotal role in governing refusal behavior. Building on this insight, our study reveals that the r-direction tends to drift during training, which we identify as one of the causes of the associated safety risks. To mitigate such drift, our proposed ProCon method introduces a projection-constrained loss term that regularizes the projection magnitude of each training sample's hidden state onto the r-direction. Our initial analysis shows that applying an appropriate constraint can effectively mitigate the refusal direction drift and associated safety risks, but remains limited by overall performance barriers. To overcome this barrier, informed by our observation of early-stage sharp drift and a data-driven perspective, we introduce a warm-up strategy that emphasizes early-stage strong constraints and broaden the data distribution to strengthen constraint signals, leading to an enhanced ProCon method. Experimental results under various datasets, scenarios, and LLMs demonstrate that our method can significantly mitigate safety risks posed by IFT while preserving task performance gains. Even compared with strong baselines, our method consistently delivers superior overall performance. Crucially, our analysis indicates that ProCon can contribute to stabilizing the r-direction during training, while such an interpretability-driven exploration of LLMs' internal mechanisms lays a solid foundation for future safety research. |
| 2025-09-08 | [\texttt{R$^\textbf{2}$AI}: Towards Resistant and Resilient AI in an Evolving World](http://arxiv.org/abs/2509.06786v1) | Youbang Sun, Xiang Wang et al. | In this position paper, we address the persistent gap between rapidly growing AI capabilities and lagging safety progress. Existing paradigms divide into ``Make AI Safe'', which applies post-hoc alignment and guardrails but remains brittle and reactive, and ``Make Safe AI'', which emphasizes intrinsic safety but struggles to address unforeseen risks in open-ended environments. We therefore propose \textit{safe-by-coevolution} as a new formulation of the ``Make Safe AI'' paradigm, inspired by biological immunity, in which safety becomes a dynamic, adversarial, and ongoing learning process. To operationalize this vision, we introduce \texttt{R$^2$AI} -- \textit{Resistant and Resilient AI} -- as a practical framework that unites resistance against known threats with resilience to unforeseen risks. \texttt{R$^2$AI} integrates \textit{fast and slow safe models}, adversarial simulation and verification through a \textit{safety wind tunnel}, and continual feedback loops that guide safety and capability to coevolve. We argue that this framework offers a scalable and proactive path to maintain continual safety in dynamic environments, addressing both near-term vulnerabilities and long-term existential risks as AI advances toward AGI and ASI. |
| 2025-09-08 | [Embodied Hazard Mitigation using Vision-Language Models for Autonomous Mobile Robots](http://arxiv.org/abs/2509.06768v1) | Oluwadamilola Sotomi, Devika Kodi et al. | Autonomous robots operating in dynamic environments should identify and report anomalies. Embodying proactive mitigation improves safety and operational continuity. This paper presents a multimodal anomaly detection and mitigation system that integrates vision-language models and large language models to identify and report hazardous situations and conflicts in real-time. The proposed system enables robots to perceive, interpret, report, and if possible respond to urban and environmental anomalies through proactive detection mechanisms and automated mitigation actions. A key contribution in this paper is the integration of Hazardous and Conflict states into the robot's decision-making framework, where each anomaly type can trigger specific mitigation strategies. User studies (n = 30) demonstrated the effectiveness of the system in anomaly detection with 91.2% prediction accuracy and relatively low latency response times using edge-ai architecture. |
| 2025-09-08 | [Aligning Large Vision-Language Models by Deep Reinforcement Learning and Direct Preference Optimization](http://arxiv.org/abs/2509.06759v1) | Thanh Thi Nguyen, Campbell Wilson et al. | Large Vision-Language Models (LVLMs) or multimodal large language models represent a significant advancement in artificial intelligence, enabling systems to understand and generate content across both visual and textual modalities. While large-scale pretraining has driven substantial progress, fine-tuning these models for aligning with human values or engaging in specific tasks or behaviors remains a critical challenge. Deep Reinforcement Learning (DRL) and Direct Preference Optimization (DPO) offer promising frameworks for this aligning process. While DRL enables models to optimize actions using reward signals instead of relying solely on supervised preference data, DPO directly aligns the policy with preferences, eliminating the need for an explicit reward model. This overview explores paradigms for fine-tuning LVLMs, highlighting how DRL and DPO techniques can be used to align models with human preferences and values, improve task performance, and enable adaptive multimodal interaction. We categorize key approaches, examine sources of preference data, reward signals, and discuss open challenges such as scalability, sample efficiency, continual learning, generalization, and safety. The goal is to provide a clear understanding of how DRL and DPO contribute to the evolution of robust and human-aligned LVLMs. |
| 2025-09-08 | [Reinforcement Learning Foundations for Deep Research Systems: A Survey](http://arxiv.org/abs/2509.06733v1) | Wenjun Li, Zhi Chen et al. | Deep research systems, agentic AI that solve complex, multi-step tasks by coordinating reasoning, search across the open web and user files, and tool use, are moving toward hierarchical deployments with a Planner, Coordinator, and Executors. In practice, training entire stacks end-to-end remains impractical, so most work trains a single planner connected to core tools such as search, browsing, and code. While SFT imparts protocol fidelity, it suffers from imitation and exposure biases and underuses environment feedback. Preference alignment methods such as DPO are schema and proxy-dependent, off-policy, and weak for long-horizon credit assignment and multi-objective trade-offs. A further limitation of SFT and DPO is their reliance on human defined decision points and subskills through schema design and labeled comparisons. Reinforcement learning aligns with closed-loop, tool-interaction research by optimizing trajectory-level policies, enabling exploration, recovery behaviors, and principled credit assignment, and it reduces dependence on such human priors and rater biases.   This survey is, to our knowledge, the first dedicated to the RL foundations of deep research systems. It systematizes work after DeepSeek-R1 along three axes: (i) data synthesis and curation; (ii) RL methods for agentic research covering stability, sample efficiency, long context handling, reward and credit design, multi-objective optimization, and multimodal integration; and (iii) agentic RL training systems and frameworks. We also cover agent architecture and coordination, as well as evaluation and benchmarks, including recent QA, VQA, long-form synthesis, and domain-grounded, tool-interaction tasks. We distill recurring patterns, surface infrastructure bottlenecks, and offer practical guidance for training robust, transparent deep research agents with RL. |
| 2025-09-08 | [Pacing Types: Safe Monitoring of Asynchronous Streams](http://arxiv.org/abs/2509.06724v1) | Florian Kohn, Arthur Correnson et al. | Stream-based monitoring is a real-time safety assurance mechanism for complex cyber-physical systems such as unmanned aerial vehicles. In this context, a monitor aggregates streams of input data from sensors and other sources to give real-time statistics and assessments of the system's health. Since monitors are safety-critical components, it is crucial to ensure that they are free of potential runtime errors. One of the central challenges in designing reliable stream-based monitors is to deal with the asynchronous nature of data streams: in concrete applications, the different sensors being monitored produce values at different speeds, and it is the monitor's responsibility to correctly react to the asynchronous arrival of different streams of values. To ease this process, modern frameworks for stream-based monitoring such as RTLola feature an expressive specification language that allows to finely specify data synchronization policies. While this feature dramatically simplifies the design of monitors, it can also lead to subtle runtime errors. To mitigate this issue, this paper presents pacing types, a novel type system implemented in RTLola to ensure that monitors for asynchronous streams are well-behaved at runtime. We formalize the essence of pacing types for a core fragment of RTLola, and present a soundness proof of the pacing type system using a new logical relation. |
| 2025-09-08 | [Sovereign AI for 6G: Towards the Future of AI-Native Networks](http://arxiv.org/abs/2509.06700v1) | Swarna Bindu Chetty, David Grace et al. | The advent of Generative Artificial Intelligence (GenAI), Large Language Models (LLMs), and Large Telecom Models (LTM) significantly reshapes mobile networks, especially as the telecom industry transitions from 5G's cloud-centric to AI-native 6G architectures. This transition unlocks unprecedented capabilities in real-time automation, semantic networking, and autonomous service orchestration. However, it introduces critical risks related to data sovereignty, security, explainability, and regulatory compliance especially when AI models are trained, deployed, or governed externally. This paper introduces the concept of `Sovereign AI' as a strategic imperative for 6G, proposing architectural, operational, and governance frameworks that enable national or operator-level control over AI development, deployment, and life-cycle management. Focusing on O-RAN architecture, we explore how sovereign AI-based xApps and rApps can be deployed Near-RT and Non-RT RICs to ensure policy-aligned control, secure model updates, and federated learning across trusted infrastructure. We analyse global strategies, technical enablers, and challenges across safety, talent, and model governance. Our findings underscore that Sovereign AI is not just a regulatory necessity but a foundational pillar for secure, resilient, and ethically-aligned 6G networks. |
| 2025-09-08 | [Safe Robust Predictive Control-based Motion Planning of Automated Surface Vessels in Inland Waterways](http://arxiv.org/abs/2509.06687v1) | Sajad Ahmadi, Hossein Nejatbakhsh Esfahani et al. | Deploying self-navigating surface vessels in inland waterways offers a sustainable alternative to reduce road traffic congestion and emissions. However, navigating confined waterways presents unique challenges, including narrow channels, higher traffic density, and hydrodynamic disturbances. Existing methods for autonomous vessel navigation often lack the robustness or precision required for such environments. This paper presents a new motion planning approach for Automated Surface Vessels (ASVs) using Robust Model Predictive Control (RMPC) combined with Control Barrier Functions (CBFs). By incorporating channel borders and obstacles as safety constraints within the control design framework, the proposed method ensures both collision avoidance and robust navigation on complex waterways. Simulation results demonstrate the efficacy of the proposed method in safely guiding ASVs under realistic conditions, highlighting its improved safety and adaptability compared to the state-of-the-art. |
| 2025-09-08 | [Human-Hardware-in-the-Loop simulations for systemic resilience assessment in cyber-socio-technical systems](http://arxiv.org/abs/2509.06657v1) | Francesco Simone, Marco Bortolini et al. | Modern industrial systems require updated approaches to safety management, as the tight interplay between cyber-physical, human, and organizational factors has driven their processes toward increasing complexity. In addition to dealing with known risks, managing system resilience acquires great value to address complex behaviors pragmatically. This manuscript starts from the System-Theoretic Accident Model and Processes (STAMP) as a modelling initiative for such complexity. The STAMP can be natively integrated with simulation-based approaches, which however fail to realistically represent human behaviors and their influence on the system performance. To overcome this limitation, this paper proposes a Human-Hardware-in-the-Loop (HHIL) modeling and simulation framework aimed at supporting a more realistic and comprehensive assessments of systemic resilience. The approach is tested on an experimental oil and gas plant experiencing cyber-attacks, where two personas of operators (experts and novices) work. This research provides a mean to quantitatively assess how variations in operator behavior impact the overall system performance, offering insights into how resilience should be understood and implemented in complex socio-technical systems at large. |
| 2025-09-08 | [LiHRA: A LiDAR-Based HRI Dataset for Automated Risk Monitoring Methods](http://arxiv.org/abs/2509.06597v1) | Frederik Plahl, Georgios Katranis et al. | We present LiHRA, a novel dataset designed to facilitate the development of automated, learning-based, or classical risk monitoring (RM) methods for Human-Robot Interaction (HRI) scenarios. The growing prevalence of collaborative robots in industrial environments has increased the need for reliable safety systems. However, the lack of high-quality datasets that capture realistic human-robot interactions, including potentially dangerous events, slows development. LiHRA addresses this challenge by providing a comprehensive, multi-modal dataset combining 3D LiDAR point clouds, human body keypoints, and robot joint states, capturing the complete spatial and dynamic context of human-robot collaboration. This combination of modalities allows for precise tracking of human movement, robot actions, and environmental conditions, enabling accurate RM during collaborative tasks. The LiHRA dataset covers six representative HRI scenarios involving collaborative and coexistent tasks, object handovers, and surface polishing, with safe and hazardous versions of each scenario. In total, the data set includes 4,431 labeled point clouds recorded at 10 Hz, providing a rich resource for training and benchmarking classical and AI-driven RM algorithms. Finally, to demonstrate LiHRA's utility, we introduce an RM method that quantifies the risk level in each scenario over time. This method leverages contextual information, including robot states and the dynamic model of the robot. With its combination of high-resolution LiDAR data, precise human tracking, robot state data, and realistic collision events, LiHRA offers an essential foundation for future research into real-time RM and adaptive safety strategies in human-robot workspaces. |
| 2025-09-08 | [Hybrid Swin Attention Networks for Simultaneously Low-Dose PET and CT Denoising](http://arxiv.org/abs/2509.06591v1) | Yichao Liu, YueYang Teng | Low-dose computed tomography (LDCT) and positron emission tomography (PET) have emerged as safer alternatives to conventional imaging modalities by significantly reducing radiation exposure. However, this reduction often results in increased noise and artifacts, which can compromise diagnostic accuracy. Consequently, denoising for LDCT/PET has become a vital area of research aimed at enhancing image quality while maintaining radiation safety. In this study, we introduce a novel Hybrid Swin Attention Network (HSANet), which incorporates Efficient Global Attention (EGA) modules and a hybrid upsampling module. The EGA modules enhance both spatial and channel-wise interaction, improving the network's capacity to capture relevant features, while the hybrid upsampling module mitigates the risk of overfitting to noise. We validate the proposed approach using a publicly available LDCT/PET dataset. Experimental results demonstrate that HSANet achieves superior denoising performance compared to existing methods, while maintaining a lightweight model size suitable for deployment on GPUs with standard memory configurations. This makes our approach highly practical for real-world clinical applications. |
| 2025-09-08 | [Relational Algebras for Subset Selection and Optimisation](http://arxiv.org/abs/2509.06439v1) | David Robert Pratten, Luke Mathieson et al. | The database community lacks a unified relational query language for subset selection and optimisation queries, limiting both user expression and query optimiser reasoning about such problems. Decades of research (latterly under the rubric of prescriptive analytics) have produced powerful evaluation algorithms with incompatible, ad-hoc SQL extensions that specify and filter through distinct mechanisms. We present the first unified algebraic foundation for these queries, introducing relational exponentiation to complete the fundamental algebraic operations alongside union (addition) and cross product (multiplication). First, we extend relational algebra to complete domain relations-relations defined by characteristic functions rather than explicit extensions-achieving the expressiveness of NP-complete/hard problems, while simultaneously providing query safety for finite inputs. Second, we introduce solution sets, a higher-order relational algebra over sets of relations that naturally expresses search spaces as functions f: Base to Decision, yielding |Decision|^|Base| candidate relations. Third, we provide structure-preserving translation semantics from solution sets to standard relational algebra, enabling mechanical translation to existing evaluation algorithms. This framework achieves the expressiveness of the most powerful prior approaches while providing the theoretical clarity and compositional properties absent in previous work. We demonstrate the capabilities these algebras open up through a polymorphic SQL where standard clauses seamlessly express data management, subset selection, and optimisation queries within a single paradigm. |
| 2025-09-08 | [Safety Meets Speed: Accelerated Neural MPC with Safety Guarantees and No Retraining](http://arxiv.org/abs/2509.06404v1) | Kaikai Wang, Tianxun Li et al. | While Model Predictive Control (MPC) enforces safety via constraints, its real-time execution can exceed embedded compute budgets. We propose a Barrier-integrated Adaptive Neural Model Predictive Control (BAN-MPC) framework that synergizes neural networks' fast computation with MPC's constraint-handling capability. To ensure strict safety, we replace traditional Euclidean distance with Control Barrier Functions (CBFs) for collision avoidance. We integrate an offline-learned neural value function into the optimization objective of a Short-horizon MPC, substantially reducing online computational complexity. Additionally, we use a second neural network to learn the sensitivity of the value function to system parameters, and adaptively adjust the neural value function based on this neural sensitivity when model parameters change, eliminating the need for retraining and reducing offline computation costs. The hardware in-the-loop (HIL) experiments on Jetson Nano show that BAN-MPC solves 200 times faster than traditional MPC, enabling collision-free navigation with control error below 5\% under model parameter variations within 15\%, making it an effective embedded MPC alternative. |
| 2025-09-08 | [Non-Destructive Rail Monitoring for Defect Identification](http://arxiv.org/abs/2509.06394v1) | Elissa Akiki, Lynda Chehami et al. | Non-destructive evaluation (NDE) of rail tracks is crucial to ensure the safety and reliability of rail transportation systems. In this work, we present a quantitative study using various signal processing methods to identify defects in rail structures. A diffuse field configuration was employed at few dozens of kiloHertz, where the emitter and receiver were remotely located, and wave energy propagated via multiple reflections within the medium. A reference database is first constructed by acquiring measurements at different rail positions and different torque levels (up to 50 N.m). The defect is then identified by comparing its signature to those stacked in the database. First, the destretching technique, based on Coda Wave Interferometry (CWI), is applied to correct for temperature-induced velocity variations. Then, the identification is performed using the Mean Square Error (MSE) metric and Orthogonal Matching Pursuit (OMP) technique. A comparative analysis of the both methods is conducted, focusing on their robustness and performance. |
| 2025-09-08 | [Adaptive Evolution Factor Risk Ellipse Framework for Reliable and Safe Autonomous Driving](http://arxiv.org/abs/2509.06375v1) | Fujiang Yuan, Zhen Tian et al. | In recent years, ensuring safety, efficiency, and comfort in interactive autonomous driving has become a critical challenge. Traditional model-based techniques, such as game-theoretic methods and robust control, are often overly conservative or computationally intensive. Conversely, learning-based approaches typically require extensive training data and frequently exhibit limited interpretability and generalizability. Simpler strategies, such as Risk Potential Fields (RPF), provide lightweight alternatives with minimal data demands but are inherently static and struggle to adapt effectively to dynamic traffic conditions. To overcome these limitations, we propose the Evolutionary Risk Potential Field (ERPF), a novel approach that dynamically updates risk assessments in dynamical scenarios based on historical obstacle proximity data. We introduce a Risk-Ellipse construct that combines longitudinal reach and lateral uncertainty into a unified spatial temporal collision envelope. Additionally, we define an adaptive Evolution Factor metric, computed through sigmoid normalization of Time to Collision (TTC) and Time-Window-of-Hazard (TWH), which dynamically adjusts the dimensions of the ellipse axes in real time. This adaptive risk metric is integrated seamlessly into a Model Predictive Control (MPC) framework, enabling autonomous vehicles to proactively address complex interactive driving scenarios in terms of uncertain driving of surrounding vehicles. Comprehensive comparative experiments demonstrate that our ERPF-MPC approach consistently achieves smoother trajectories, higher average speeds, and collision-free navigation, offering a robust and adaptive solution suitable for complex interactive driving environments. |
| 2025-09-08 | [Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?](http://arxiv.org/abs/2509.06350v1) | Junjie Mu, Zonghao Ying et al. | Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks. |

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



