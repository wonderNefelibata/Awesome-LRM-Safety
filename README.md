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
| 2025-05-07 | [Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization](http://arxiv.org/abs/2505.04578v1) | Wenjun Cao | Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models. |
| 2025-05-07 | [Stow: Robotic Packing of Items into Fabric Pods](http://arxiv.org/abs/2505.04572v1) | Nicolas Hudson, Josh Hooks et al. | This paper presents a compliant manipulation system capable of placing items onto densely packed shelves. The wide diversity of items and strict business requirements for high producing rates and low defect generation have prohibited warehouse robotics from performing this task. Our innovations in hardware, perception, decision-making, motion planning, and control have enabled this system to perform over 500,000 stows in a large e-commerce fulfillment center. The system achieves human levels of packing density and speed while prioritizing work on overhead shelves to enhance the safety of humans working alongside the robots. |
| 2025-05-07 | [Runtime Advocates: A Persona-Driven Framework for Requirements@Runtime Decision Support](http://arxiv.org/abs/2505.04551v1) | Demetrius Hernandez, Jane Cleland-Huang | Complex systems, such as small Uncrewed Aerial Systems (sUAS) swarms dispatched for emergency response, often require dynamic reconfiguration at runtime under the supervision of human operators. This introduces human-on-the-loop requirements, where evolving needs shape ongoing system functionality and behaviors. While traditional personas support upfront, static requirements elicitation, we propose a persona-based advocate framework for runtime requirements engineering to provide ethically informed, safety-driven, and regulatory-aware decision support. Our approach extends standard personas into event-driven personas. When triggered by events such as adverse environmental conditions, evolving mission state, or operational constraints, the framework updates the sUAS operator's view of the personas, ensuring relevance to current conditions. We create three key advocate personas, namely Safety Controller, Ethical Governor, and Regulatory Auditor, to manage trade-offs among risk, ethical considerations, and regulatory compliance. We perform a proof-of-concept validation in an emergency response scenario using sUAS, showing how our advocate personas provide context-aware guidance grounded in safety, regulatory, and ethical constraints. By evolving static, design-time personas into adaptive, event-driven advocates, the framework surfaces mission-critical runtime requirements in response to changing conditions. These requirements shape operator decisions in real time, aligning actions with the operational demands of the moment. |
| 2025-05-07 | [Pangu Ultra MoE: How to Train Your Big MoE on Ascend NPUs](http://arxiv.org/abs/2505.04519v1) | Yehui Tang, Yichun Yin et al. | Sparse large language models (LLMs) with Mixture of Experts (MoE) and close to a trillion parameters are dominating the realm of most capable language models. However, the massive model scale poses significant challenges for the underlying software and hardware systems. In this paper, we aim to uncover a recipe to harness such scale on Ascend NPUs. The key goals are better usage of the computing resources under the dynamic sparse model structures and materializing the expected performance gain on the actual hardware. To select model configurations suitable for Ascend NPUs without repeatedly running the expensive experiments, we leverage simulation to compare the trade-off of various model hyperparameters. This study led to Pangu Ultra MoE, a sparse LLM with 718 billion parameters, and we conducted experiments on the model to verify the simulation results. On the system side, we dig into Expert Parallelism to optimize the communication between NPU devices to reduce the synchronization overhead. We also optimize the memory efficiency within the devices to further reduce the parameter and activation management overhead. In the end, we achieve an MFU of 30.0% when training Pangu Ultra MoE, with performance comparable to that of DeepSeek R1, on 6K Ascend NPUs, and demonstrate that the Ascend system is capable of harnessing all the training stages of the state-of-the-art language models. Extensive experiments indicate that our recipe can lead to efficient training of large-scale sparse language models with MoE. We also study the behaviors of such models for future reference. |
| 2025-05-07 | [Advancements in Solid-State Sodium-Based Batteries: A Comprehensive Review](http://arxiv.org/abs/2505.04391v1) | Arianna Massaro, Lorenzo Squillantini et al. | This manuscript explores recent advancements in solid-state sodium-based battery technology, particularly focusing on electrochemical performance and the challenges associated with developing efficient solid electrolytes. The replacement of conventional liquid electrolytes with solid-state alternatives offers numerous benefits, including enhanced safety and environmental sustainability, as solid-state systems reduce flammability and harsh chemical handling. The work emphasizes the importance of structure and interface characteristics in solid electrolytes, which play a critical role in ionic conductivity and overall battery performance. Various classes of solid electrolytes, such as sodium-based anti-perovskites and sulphide electrolytes, are examined, highlighting their unique ionic transport mechanisms and mechanical properties that facilitate stable cycling. The manuscript also discusses strategies to enhance interfacial stability between the anode and the solid electrolyte to mitigate performance degradation during battery operation. Furthermore, advancements in electrode formulations and the integration of novel materials are considered pivotal in optimizing the charging and discharging processes, thus improving the energy and power densities of sodium batteries. The outlook on the future of sodium-based solid-state batteries underscores their potential to meet emerging energy storage demands while leveraging the abundant availability of sodium compared to lithium. This comprehensive review aims to provide insights into ongoing research and prospective directions for the commercialization of solid-state sodium-based batteries, positioning them as viable alternatives in the renewable energy landscape. |
| 2025-05-07 | [The Aloe Family Recipe for Open and Specialized Healthcare LLMs](http://arxiv.org/abs/2505.04388v1) | Dario Garcia-Gasulla, Jordi Bayarri-Planas et al. | Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.   Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.   Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.   Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare. |
| 2025-05-07 | [Consensus-Aware AV Behavior: Trade-offs Between Safety, Interaction, and Performance in Mixed Urban Traffic](http://arxiv.org/abs/2505.04379v1) | Mohammad Elayan, Wissam Kontar | Transportation systems have long been shaped by complexity and heterogeneity, driven by the interdependency of agent actions and traffic outcomes. The deployment of automated vehicles (AVs) in such systems introduces a new challenge: achieving consensus across safety, interaction quality, and traffic performance. In this work, we position consensus as a fundamental property of the traffic system and aim to quantify it. We use high-resolution trajectory data from the Third Generation Simulation (TGSIM) dataset to empirically analyze AV and human-driven vehicle (HDV) behavior at a signalized urban intersection and around vulnerable road users (VRUs). Key metrics, including Time-to-Collision (TTC), Post-Encroachment Time (PET), deceleration patterns, headways, and string stability, are evaluated across the three performance dimensions. Results show that full consensus across safety, interaction, and performance is rare, with only 1.63% of AV-VRU interaction frames meeting all three conditions. These findings highlight the need for AV models that explicitly balance multi-dimensional performance in mixed-traffic environments. Full reproducibility is supported via our open-source codebase on https://github.com/wissamkontar/Consensus-AV-Analysis. |
| 2025-05-07 | [Resist Platform-Controlled AI Agents and Champion User-Centric Agent Advocates](http://arxiv.org/abs/2505.04345v1) | Sayash Kapoor, Noam Kolt et al. | Language model agents could reshape how users navigate and act in digital environments. If controlled by platform companies -- either those that already dominate online search, communication, and commerce, or those vying to replace them -- platform agents could intensify surveillance, exacerbate user lock-in, and further entrench the incumbent digital giants. This position paper argues that to resist the undesirable effects of platform agents, we should champion agent advocates -- agents that are controlled by users, serve the interests of users, and preserve user autonomy and choice. We identify key interventions to enable agent advocates: ensuring public access to compute, developing interoperability protocols and safety standards, and implementing appropriate market regulations. |
| 2025-05-07 | [Detecting Concept Drift in Neural Networks Using Chi-squared Goodness of Fit Testing](http://arxiv.org/abs/2505.04318v1) | Jacob Glenn Ayers, Buvaneswari A. Ramanan et al. | As the adoption of deep learning models has grown beyond human capacity for verification, meta-algorithms are needed to ensure reliable model inference. Concept drift detection is a field dedicated to identifying statistical shifts that is underutilized in monitoring neural networks that may encounter inference data with distributional characteristics diverging from their training data. Given the wide variety of model architectures, applications, and datasets, it is important that concept drift detection algorithms are adaptable to different inference scenarios. In this paper, we introduce an application of the $\chi^2$ Goodness of Fit Hypothesis Test as a drift detection meta-algorithm applied to a multilayer perceptron, a convolutional neural network, and a transformer trained for machine vision as they are exposed to simulated drift during inference. To that end, we demonstrate how unexpected drops in accuracy due to concept drift can be detected without directly examining the inference outputs. Our approach enhances safety by ensuring models are continually evaluated for reliability across varying conditions. |
| 2025-05-07 | [From Incidents to Insights: Patterns of Responsibility following AI Harms](http://arxiv.org/abs/2505.04291v1) | Isabel Richards, Claire Benn et al. | The AI Incident Database was inspired by aviation safety databases, which enable collective learning from failures to prevent future incidents. The database documents hundreds of AI failures, collected from the news and media. However, criticism highlights that the AIID's reliance on media reporting limits its utility for learning about implementation failures. In this paper, we accept that the AIID falls short in its original mission, but argue that by looking beyond technically-focused learning, the dataset can provide new, highly valuable insights: specifically, opportunities to learn about patterns between developers, deployers, victims, wider society, and law-makers that emerge after AI failures. Through a three-tier mixed-methods analysis of 962 incidents and 4,743 related reports from the AIID, we examine patterns across incidents, focusing on cases with public responses tagged in the database. We identify 'typical' incidents found in the AIID, from Tesla crashes to deepfake scams.   Focusing on this interplay between relevant parties, we uncover patterns in accountability and social expectations of responsibility. We find that the presence of identifiable responsible parties does not necessarily lead to increased accountability. The likelihood of a response and what it amounts to depends highly on context, including who built the technology, who was harmed, and to what extent. Controversy-rich incidents provide valuable data about societal reactions, including insights into social expectations. Equally informative are cases where controversy is notably absent. This work shows that the AIID's value lies not just in preventing technical failures, but in documenting patterns of harms and of institutional response and social learning around AI incidents. These patterns offer crucial insights for understanding how society adapts to and governs emerging AI technologies. |
| 2025-05-07 | [Multi-Agent Reinforcement Learning-based Cooperative Autonomous Driving in Smart Intersections](http://arxiv.org/abs/2505.04231v1) | Taoyuan Yu, Kui Wang et al. | Unsignalized intersections pose significant safety and efficiency challenges due to complex traffic flows. This paper proposes a novel roadside unit (RSU)-centric cooperative driving system leveraging global perception and vehicle-to-infrastructure (V2I) communication. The core of the system is an RSU-based decision-making module using a two-stage hybrid reinforcement learning (RL) framework. At first, policies are pre-trained offline using conservative Q-learning (CQL) combined with behavior cloning (BC) on collected dataset. Subsequently, these policies are fine-tuned in the simulation using multi-agent proximal policy optimization (MAPPO), aligned with a self-attention mechanism to effectively solve inter-agent dependencies. RSUs perform real-time inference based on the trained models to realize vehicle control via V2I communications. Extensive experiments in CARLA environment demonstrate high effectiveness of the proposed system, by: \textit{(i)} achieving failure rates below 0.03\% in coordinating three connected and autonomous vehicles (CAVs) through complex intersection scenarios, significantly outperforming the traditional Autoware control method, and \textit{(ii)} exhibiting strong robustness across varying numbers of controlled agents and shows promising generalization capabilities on other maps. |
| 2025-05-07 | [An Enhanced YOLOv8 Model for Real-Time and Accurate Pothole Detection and Measurement](http://arxiv.org/abs/2505.04207v1) | Mustafa Yurdakul, Şakir Tasdemir | Potholes cause vehicle damage and traffic accidents, creating serious safety and economic problems. Therefore, early and accurate detection of potholes is crucial. Existing detection methods are usually only based on 2D RGB images and cannot accurately analyze the physical characteristics of potholes. In this paper, a publicly available dataset of RGB-D images (PothRGBD) is created and an improved YOLOv8-based model is proposed for both pothole detection and pothole physical features analysis. The Intel RealSense D415 depth camera was used to collect RGB and depth data from the road surfaces, resulting in a PothRGBD dataset of 1000 images. The data was labeled in YOLO format suitable for segmentation. A novel YOLO model is proposed based on the YOLOv8n-seg architecture, which is structurally improved with Dynamic Snake Convolution (DSConv), Simple Attention Module (SimAM) and Gaussian Error Linear Unit (GELU). The proposed model segmented potholes with irregular edge structure more accurately, and performed perimeter and depth measurements on depth maps with high accuracy. The standard YOLOv8n-seg model achieved 91.9% precision, 85.2% recall and 91.9% mAP@50. With the proposed model, the values increased to 93.7%, 90.4% and 93.8% respectively. Thus, an improvement of 1.96% in precision, 6.13% in recall and 2.07% in mAP was achieved. The proposed model performs pothole detection as well as perimeter and depth measurement with high accuracy and is suitable for real-time applications due to its low model complexity. In this way, a lightweight and effective model that can be used in deep learning-based intelligent transportation solutions has been acquired. |
| 2025-05-07 | [Unmasking the Canvas: A Dynamic Benchmark for Image Generation Jailbreaking and LLM Content Safety](http://arxiv.org/abs/2505.04146v1) | Variath Madhupal Gautham Nair, Vishal Varma Dantuluri | Existing large language models (LLMs) are advancing rapidly and produce outstanding results in image generation tasks, yet their content safety checks remain vulnerable to prompt-based jailbreaks. Through preliminary testing on platforms such as ChatGPT, MetaAI, and Grok, we observed that even short, natural prompts could lead to the generation of compromising images ranging from realistic depictions of forged documents to manipulated images of public figures.   We introduce Unmasking the Canvas (UTC Benchmark; UTCB), a dynamic and scalable benchmark dataset to evaluate LLM vulnerability in image generation. Our methodology combines structured prompt engineering, multilingual obfuscation (e.g., Zulu, Gaelic, Base64), and evaluation using Groq-hosted LLaMA-3. The pipeline supports both zero-shot and fallback prompting strategies, risk scoring, and automated tagging. All generations are stored with rich metadata and curated into Bronze (non-verified), Silver (LLM-aided verification), and Gold (manually verified) tiers. UTCB is designed to evolve over time with new data sources, prompt templates, and model behaviors.   Warning: This paper includes visual examples of adversarial inputs designed to test model safety. All outputs have been redacted to ensure responsible disclosure. |
| 2025-05-07 | [In-Situ Hardware Error Detection Using Specification-Derived Petri Net Models and Behavior-Derived State Sequences](http://arxiv.org/abs/2505.04108v1) | Tomonari Tanaka, Takumi Uezono et al. | In hardware accelerators used in data centers and safety-critical applications, soft errors and resultant silent data corruption significantly compromise reliability, particularly when upsets occur in control-flow operations, leading to severe failures. To address this, we introduce two methods for monitoring control flows: using specification-derived Petri nets and using behavior-derived state transitions. We validated our method across four designs: convolutional layer operation, Gaussian blur, AES encryption, and a router in Network-on-Chip. Our fault injection campaign targeting the control registers and primary control inputs demonstrated high error detection rates in both datapath and control logic. Synthesis results show that a maximum detection rate is achieved with a few to around 10% area overhead in most cases. The proposed detectors quickly detect 48% to 100% of failures resulting from upsets in internal control registers and perturbations in primary control inputs. The two proposed methods were compared in terms of area overhead and error detection rate. By selectively applying these two methods, a wide range of area constraints can be accommodated, enabling practical implementation and effectively enhancing error detection capabilities. |
| 2025-05-07 | [Shadow Wireless Intelligence: Large Language Model-Driven Reasoning in Covert Communications](http://arxiv.org/abs/2505.04068v1) | Yuanai Xie, Zhaozhi Liu et al. | Covert Communications (CC) can secure sensitive transmissions in industrial, military, and mission-critical applications within 6G wireless networks. However, traditional optimization methods based on Artificial Noise (AN), power control, and channel manipulation might not adapt to dynamic and adversarial environments due to the high dimensionality, nonlinearity, and stringent real-time covertness requirements. To bridge this gap, we introduce Shadow Wireless Intelligence (SWI), which integrates the reasoning capabilities of Large Language Models (LLMs) with retrieval-augmented generation to enable intelligent decision-making in covert wireless systems. Specifically, we utilize DeepSeek-R1, a mixture-of-experts-based LLM with RL-enhanced reasoning, combined with real-time retrieval of domain-specific knowledge to improve context accuracy and mitigate hallucinations. Our approach develops a structured CC knowledge base, supports context-aware retrieval, and performs semantic optimization, allowing LLMs to generate and adapt CC strategies in real time. In a case study on optimizing AN power in a full-duplex CC scenario, DeepSeek-R1 achieves 85% symbolic derivation accuracy and 94% correctness in the generation of simulation code, outperforming baseline models. These results validate SWI as a robust, interpretable, and adaptive foundation for LLM-driven intelligent covert wireless systems in 6G networks. |
| 2025-05-07 | [Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks](http://arxiv.org/abs/2505.04046v1) | Xuyang Wang, Siyuan Duan et al. | Recently, trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. In practice, however, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view learning models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that our RDML outperforms the state-of-the-art multi-view learning methods by a relatively large margin. |
| 2025-05-06 | [Estimating the Joint Distribution of Two Binary Variables with Marginal Statistics](http://arxiv.org/abs/2505.03995v1) | Longwen Shang, Min Tsao et al. | Clinical trial simulation (CTS) is critical in new drug development, providing insight into safety and efficacy while guiding trial design. Achieving realistic outcomes in CTS requires an accurately estimated joint distribution of the underlying variables. However, privacy concerns and data availability issues often restrict researchers to marginal summary-level data of each variable, making it challenging to estimate the joint distribution due to the lack of access to individual-level data or relational summaries between variables. We propose a novel approach based on the method of maximum likelihood that estimates the joint distribution of two binary variables using only marginal summary data. By leveraging numerical optimization and accommodating varying sample sizes across studies, our method preserves privacy while bypassing the need for granular or relational data. Through an extensive simulation study covering a diverse range of scenarios and an application to a real-world dataset, we demonstrate the accuracy, robustness, and practicality of our method. This method enhances the generation of realistic simulated data, thereby improving decision-making processes in drug development. |
| 2025-05-06 | [An alignment safety case sketch based on debate](http://arxiv.org/abs/2505.03989v1) | Marie Davidsen Buhl, Jacob Pfau et al. | If AI systems match or exceed human capabilities on a wide range of tasks, it may become difficult for humans to efficiently judge their actions -- making it hard to use human feedback to steer them towards desirable traits. One proposed solution is to leverage another superhuman system to point out flaws in the system's outputs via a debate. This paper outlines the value of debate for AI safety, as well as the assumptions and further research required to make debate work. It does so by sketching an ``alignment safety case'' -- an argument that an AI system will not autonomously take actions which could lead to egregious harm, despite being able to do so. The sketch focuses on the risk of an AI R\&D agent inside an AI company sabotaging research, for example by producing false results. To prevent this, the agent is trained via debate, subject to exploration guarantees, to teach the system to be honest. Honesty is maintained throughout deployment via online training. The safety case rests on four key claims: (1) the agent has become good at the debate game, (2) good performance in the debate game implies that the system is mostly honest, (3) the system will not become significantly less honest during deployment, and (4) the deployment context is tolerant of some errors. We identify open research problems that, if solved, could render this a compelling argument that an AI system is safe. |
| 2025-05-06 | [LogiDebrief: A Signal-Temporal Logic based Automated Debriefing Approach with Large Language Models Integration](http://arxiv.org/abs/2505.03985v1) | Zirong Chen, Ziyan An et al. | Emergency response services are critical to public safety, with 9-1-1 call-takers playing a key role in ensuring timely and effective emergency operations. To ensure call-taking performance consistency, quality assurance is implemented to evaluate and refine call-takers' skillsets. However, traditional human-led evaluations struggle with high call volumes, leading to low coverage and delayed assessments. We introduce LogiDebrief, an AI-driven framework that automates traditional 9-1-1 call debriefing by integrating Signal-Temporal Logic (STL) with Large Language Models (LLMs) for fully-covered rigorous performance evaluation. LogiDebrief formalizes call-taking requirements as logical specifications, enabling systematic assessment of 9-1-1 calls against procedural guidelines. It employs a three-step verification process: (1) contextual understanding to identify responder types, incident classifications, and critical conditions; (2) STL-based runtime checking with LLM integration to ensure compliance; and (3) automated aggregation of results into quality assurance reports. Beyond its technical contributions, LogiDebrief has demonstrated real-world impact. Successfully deployed at Metro Nashville Department of Emergency Communications, it has assisted in debriefing 1,701 real-world calls, saving 311.85 hours of active engagement. Empirical evaluation with real-world data confirms its accuracy, while a case study and extensive user study highlight its effectiveness in enhancing call-taking performance. |
| 2025-05-06 | [X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains](http://arxiv.org/abs/2505.03981v1) | Qianchu Liu, Sheng Zhang et al. | Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General-domain text-based post-training can enable such strong generalizable reasoning. Leveraging this finding, we introduce X-Reasoner, a vision-language model post-trained solely on general-domain text for generalizable reasoning, using a two-stage approach: an initial supervised fine-tuning phase with distilled long chain-of-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-Reasoner successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing state-of-the-art models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-Reasoner's performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce X-Reasoner-Med, a medical-specialized variant that achieves new state of the art on numerous text-only and multimodal medical benchmarks. |

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



