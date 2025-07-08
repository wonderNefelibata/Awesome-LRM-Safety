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
| 2025-07-07 | [Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models](http://arxiv.org/abs/2507.05248v1) | Ziqi Miao, Lijun Li et al. | Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. Building on this insight, we propose Response Attack, which uses an auxiliary LLM to generate a mildly harmful response to a paraphrased version of the original malicious query. They are then formatted into the dialogue and followed by a succinct trigger prompt, thereby priming the target model to generate harmful content. Across eight open-source and proprietary LLMs, RA consistently outperforms seven state-of-the-art jailbreak techniques, achieving higher attack success rates. To mitigate this threat, we construct and release a context-aware safety fine-tuning dataset, which significantly reduces the attack success rate while preserving model capabilities. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack. |
| 2025-07-07 | [When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors](http://arxiv.org/abs/2507.05246v1) | Scott Emmons, Erik Jenner et al. | While chain-of-thought (CoT) monitoring is an appealing AI safety defense, recent work on "unfaithfulness" has cast doubt on its reliability. These findings highlight an important failure mode, particularly when CoT acts as a post-hoc rationalization in applications like auditing for bias. However, for the distinct problem of runtime monitoring to prevent severe harm, we argue the key property is not faithfulness but monitorability. To this end, we introduce a conceptual framework distinguishing CoT-as-rationalization from CoT-as-computation. We expect that certain classes of severe harm will require complex, multi-step reasoning that necessitates CoT-as-computation. Replicating the experimental setups of prior work, we increase the difficulty of the bad behavior to enforce this necessity condition; this forces the model to expose its reasoning, making it monitorable. We then present methodology guidelines to stress-test CoT monitoring against deliberate evasion. Applying these guidelines, we find that models can learn to obscure their intentions, but only when given significant help, such as detailed human-written strategies or iterative optimization against the monitor. We conclude that, while not infallible, CoT monitoring offers a substantial layer of defense that requires active protection and continued stress-testing. |
| 2025-07-07 | [NavigScene: Bridging Local Perception and Global Navigation for Beyond-Visual-Range Autonomous Driving](http://arxiv.org/abs/2507.05227v1) | Qucheng Peng, Chen Bai et al. | Autonomous driving systems have made significant advances in Q&A, perception, prediction, and planning based on local visual information, yet they struggle to incorporate broader navigational context that human drivers routinely utilize. We address this critical gap between local sensor data and global navigation information by proposing NavigScene, an auxiliary navigation-guided natural language dataset that simulates a human-like driving environment within autonomous driving systems. Moreover, we develop three complementary paradigms to leverage NavigScene: (1) Navigation-guided Reasoning, which enhances vision-language models by incorporating navigation context into the prompting approach; (2) Navigation-guided Preference Optimization, a reinforcement learning method that extends Direct Preference Optimization to improve vision-language model responses by establishing preferences for navigation-relevant summarized information; and (3) Navigation-guided Vision-Language-Action model, which integrates navigation guidance and vision-language models with conventional driving models through feature fusion. Extensive experiments demonstrate that our approaches significantly improve performance across perception, prediction, planning, and question-answering tasks by enabling reasoning capabilities beyond visual range and improving generalization to diverse driving scenarios. This work represents a significant step toward more comprehensive autonomous driving systems capable of navigating complex, unfamiliar environments with greater reliability and safety. |
| 2025-07-07 | [Real-Time AI-Driven Pipeline for Automated Medical Study Content Generation in Low-Resource Settings: A Kenyan Case Study](http://arxiv.org/abs/2507.05212v1) | Emmanuel Korir, Eugene Wechuli | Juvenotes is a real-time AI-driven pipeline that automates the transformation of academic documents into structured exam-style question banks, optimized for low-resource medical education settings in Kenya. The system combines Azure Document Intelligence for OCR and Azure AI Foundry (OpenAI o3-mini) for question and answer generation in a microservices architecture, with a Vue/TypeScript frontend and AdonisJS backend. Mobile-first design, bandwidth-sensitive interfaces, institutional tagging, and offline features address local challenges. Piloted over seven months at Kenyan medical institutions, Juvenotes reduced content curation time from days to minutes and increased daily active users by 40%. Ninety percent of students reported improved study experiences. Key challenges included intermittent connectivity and AI-generated errors, highlighting the need for offline sync and human validation. Juvenotes shows that AI automation with contextual UX can enhance access to quality study materials in low-resource settings. |
| 2025-07-07 | [A Two-Stage Scheduling Method for Nurse Scheduling and Its Practical Application](http://arxiv.org/abs/2507.05182v1) | Keisuke Nakashima, Kohei Furuike et al. | The creation of nurses' schedules is a critical task that directly impacts the quality and safety of patient care as well as the quality of life for nurses. In most hospitals in Japan, this responsibility falls to the head nurse of each ward. The physical and mental burden of this task is considerable, and recent challenges such as the growing shortage of nurses and increasingly diverse working styles have further complicated the scheduling process. Consequently, there is a growing demand for automated nurse scheduling systems. Technically, modern integer programming solvers can generate feasible schedules within a practical timeframe. However, in many hospitals, schedules are still created manually. This is largely because tacit knowledge, considerations unconsciously applied by head nurses, cannot be fully formalized into explicit constraints, often resulting in automatically generated schedules that are not practically usable. To address this issue, we propose a novel "two-stage scheduling method." This approach divides the scheduling task into night shift and day shift stages, allowing head nurses to make manual adjustments after the first stage. This interactive process makes it possible to produce nurse schedules that are suitable for real-world implementation. Furthermore, to promote the practical adoption of nurse scheduling, we present case studies from acute and chronic care hospitals where systems based on the proposed method were deployed. We also discuss the challenges encountered during implementation and the corresponding solutions. |
| 2025-07-07 | [Perspectives on How Sociology Can Advance Theorizing about Human-Chatbot Interaction and Developing Chatbots for Social Good](http://arxiv.org/abs/2507.05030v1) | Celeste Campos-Castillo, Xuan Kang et al. | Recently, research into chatbots (also known as conversational agents, AI agents, voice assistants), which are computer applications using artificial intelligence to mimic human-like conversation, has grown sharply. Despite this growth, sociology lags other disciplines (including computer science, medicine, psychology, and communication) in publishing about chatbots. We suggest sociology can advance understanding of human-chatbot interaction and offer four sociological theories to enhance extant work in this field. The first two theories (resource substitution theory, power-dependence theory) add new insights to existing models of the drivers of chatbot use, which overlook sociological concerns about how social structure (e.g., systemic discrimination, the uneven distribution of resources within networks) inclines individuals to use chatbots, including problematic levels of emotional dependency on chatbots. The second two theories (affect control theory, fundamental cause of disease theory) help inform the development of chatbot-driven interventions that minimize safety risks and enhance equity by leveraging sociological insights into how chatbot outputs could attend to cultural contexts (e.g., affective norms) to promote wellbeing and enhance communities (e.g., opportunities for civic participation). We discuss the value of applying sociological theories for advancing theorizing about human-chatbot interaction and developing chatbots for social good. |
| 2025-07-07 | [Adaptation of Multi-modal Representation Models for Multi-task Surgical Computer Vision](http://arxiv.org/abs/2507.05020v1) | Soham Walimbe, Britty Baby et al. | Surgical AI often involves multiple tasks within a single procedure, like phase recognition or assessing the Critical View of Safety in laparoscopic cholecystectomy. Traditional models, built for one task at a time, lack flexibility, requiring a separate model for each. To address this, we introduce MML-SurgAdapt, a unified multi-task framework with Vision-Language Models (VLMs), specifically CLIP, to handle diverse surgical tasks through natural language supervision. A key challenge in multi-task learning is the presence of partial annotations when integrating different tasks. To overcome this, we employ Single Positive Multi-Label (SPML) learning, which traditionally reduces annotation burden by training models with only one positive label per instance. Our framework extends this approach to integrate data from multiple surgical tasks within a single procedure, enabling effective learning despite incomplete or noisy annotations. We demonstrate the effectiveness of our model on a combined dataset consisting of Cholec80, Endoscapes2023, and CholecT50, utilizing custom prompts. Extensive evaluation shows that MML-SurgAdapt performs comparably to task-specific benchmarks, with the added advantage of handling noisy annotations. It also outperforms the existing SPML frameworks for the task. By reducing the required labels by 23%, our approach proposes a more scalable and efficient labeling process, significantly easing the annotation burden on clinicians. To our knowledge, this is the first application of SPML to integrate data from multiple surgical tasks, presenting a novel and generalizable solution for multi-task learning in surgical computer vision. Implementation is available at: https://github.com/CAMMA-public/MML-SurgAdapt |
| 2025-07-07 | [Multi-modal Representations for Fine-grained Multi-label Critical View of Safety Recognition](http://arxiv.org/abs/2507.05007v1) | Britty Baby, Vinkle Srivastav et al. | The Critical View of Safety (CVS) is crucial for safe laparoscopic cholecystectomy, yet assessing CVS criteria remains a complex and challenging task, even for experts. Traditional models for CVS recognition depend on vision-only models learning with costly, labor-intensive spatial annotations. This study investigates how text can be harnessed as a powerful tool for both training and inference in multi-modal surgical foundation models to automate CVS recognition. Unlike many existing multi-modal models, which are primarily adapted for multi-class classification, CVS recognition requires a multi-label framework. Zero-shot evaluation of existing multi-modal surgical models shows a significant performance gap for this task. To address this, we propose CVS-AdaptNet, a multi-label adaptation strategy that enhances fine-grained, binary classification across multiple labels by aligning image embeddings with textual descriptions of each CVS criterion using positive and negative prompts. By adapting PeskaVLP, a state-of-the-art surgical foundation model, on the Endoscapes-CVS201 dataset, CVS-AdaptNet achieves 57.6 mAP, improving over the ResNet50 image-only baseline (51.5 mAP) by 6 points. Our results show that CVS-AdaptNet's multi-label, multi-modal framework, enhanced by textual prompts, boosts CVS recognition over image-only methods. We also propose text-specific inference methods, that helps in analysing the image-text alignment. While further work is needed to match state-of-the-art spatial annotation-based methods, this approach highlights the potential of adapting generalist models to specialized surgical tasks. Code: https://github.com/CAMMA-public/CVS-AdaptNet |
| 2025-07-07 | [From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems](http://arxiv.org/abs/2507.04996v1) | Jiangbo Yu | Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks. |
| 2025-07-07 | [MARBLE: A Multi-Agent Rule-Based LLM Reasoning Engine for Accident Severity Prediction](http://arxiv.org/abs/2507.04893v1) | Kaleem Ullah Qasim, Jiashu Zhang | Accident severity prediction plays a critical role in transportation safety systems but is a persistently difficult task due to incomplete data, strong feature dependencies, and severe class imbalance in which rare but high-severity cases are underrepresented and hard to detect. Existing methods often rely on monolithic models or black box prompting, which struggle to scale in noisy, real-world settings and offer limited interpretability. To address these challenges, we propose MARBLE a multiagent rule based LLM engine that decomposes the severity prediction task across a team of specialized reasoning agents, including an interchangeable ML-backed agent. Each agent focuses on a semantic subset of features (e.g., spatial, environmental, temporal), enabling scoped reasoning and modular prompting without the risk of prompt saturation. Predictions are coordinated through either rule-based or LLM-guided consensus mechanisms that account for class rarity and confidence dynamics. The system retains structured traces of agent-level reasoning and coordination outcomes, supporting in-depth interpretability and post-hoc performance diagnostics. Across both UK and US datasets, MARBLE consistently outperforms traditional machine learning classifiers and state-of-the-art (SOTA) prompt-based reasoning methods including Chain-of-Thought (CoT), Least-to-Most (L2M), and Tree-of-Thought (ToT) achieving nearly 90% accuracy where others plateau below 48%. This performance redefines the practical ceiling for accident severity classification under real world noise and extreme class imbalance. Our results position MARBLE as a generalizable and interpretable framework for reasoning under uncertainty in safety-critical applications. |
| 2025-07-07 | [Beyond Training-time Poisoning: Component-level and Post-training Backdoors in Deep Reinforcement Learning](http://arxiv.org/abs/2507.04883v1) | Sanyam Vyas, Alberto Caron et al. | Deep Reinforcement Learning (DRL) systems are increasingly used in safety-critical applications, yet their security remains severely underexplored. This work investigates backdoor attacks, which implant hidden triggers that cause malicious actions only when specific inputs appear in the observation space. Existing DRL backdoor research focuses solely on training-time attacks requiring unrealistic access to the training pipeline. In contrast, we reveal critical vulnerabilities across the DRL supply chain where backdoors can be embedded with significantly reduced adversarial privileges. We introduce two novel attacks: (1) TrojanentRL, which exploits component-level flaws to implant a persistent backdoor that survives full model retraining; and (2) InfrectroRL, a post-training backdoor attack which requires no access to training, validation, nor test data. Empirical and analytical evaluations across six Atari environments show our attacks rival state-of-the-art training-time backdoor attacks while operating under much stricter adversarial constraints. We also demonstrate that InfrectroRL further evades two leading DRL backdoor defenses. These findings challenge the current research focus and highlight the urgent need for robust defenses. |
| 2025-07-07 | [Enabling Security on the Edge: A CHERI Compartmentalized Network Stack](http://arxiv.org/abs/2507.04818v1) | Donato Ferraro, Andrea Bastoni et al. | The widespread deployment of embedded systems in critical infrastructures, interconnected edge devices like autonomous drones, and smart industrial systems requires robust security measures. Compromised systems increase the risks of operational failures, data breaches, and -- in safety-critical environments -- potential physical harm to people. Despite these risks, current security measures are often insufficient to fully address the attack surfaces of embedded devices. CHERI provides strong security from the hardware level by enabling fine-grained compartmentalization and memory protection, which can reduce the attack surface and improve the reliability of such devices. In this work, we explore the potential of CHERI to compartmentalize one of the most critical and targeted components of interconnected systems: their network stack. Our case study examines the trade-offs of isolating applications, TCP/IP libraries, and network drivers on a CheriBSD system deployed on the Arm Morello platform. Our results suggest that CHERI has the potential to enhance security while maintaining performance in embedded-like environments. |
| 2025-07-07 | [Safe Bimanual Teleoperation with Language-Guided Collision Avoidance](http://arxiv.org/abs/2507.04791v1) | Dionis Totsila, Clemente Donoso et al. | Teleoperating precise bimanual manipulations in cluttered environments is challenging for operators, who often struggle with limited spatial perception and difficulty estimating distances between target objects, the robot's body, obstacles, and the surrounding environment. To address these challenges, local robot perception and control should assist the operator during teleoperation. In this work, we introduce a safe teleoperation system that enhances operator control by preventing collisions in cluttered environments through the combination of immersive VR control and voice-activated collision avoidance. Using HTC Vive controllers, operators directly control a bimanual mobile manipulator, while spoken commands such as "avoid the yellow tool" trigger visual grounding and segmentation to build 3D obstacle meshes. These meshes are integrated into a whole-body controller to actively prevent collisions during teleoperation. Experiments in static, cluttered scenes demonstrate that our system significantly improves operational safety without compromising task efficiency. |
| 2025-07-07 | [Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2507.04724v1) | Yizhe Xie, Congcong Zhu et al. | Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at https://anonymous.4open.science/r/AgentXposed-F814. |
| 2025-07-07 | [Optimal Exact Designs of Multiresponse Experiments under Linear and Sparsity Constraints](http://arxiv.org/abs/2507.04713v1) | Lenka Filová, Pál Somogyi et al. | We propose a computational approach to constructing exact designs on finite design spaces that are optimal for multiresponse regression experiments under a combination of the standard linear and specific 'sparsity' constraints. The linear constraints address, for example, limits on multiple resource consumption and the problem of optimal design augmentation, while the sparsity constraints control the set of distinct trial conditions utilized by the design. The key idea is to construct an artificial optimal design problem that can be solved using any existing mathematical programming technique for univariate-response optimal designs under pure linear constraints. The solution to this artificial problem can then be directly converted into an optimal design for the primary multivariate-response setting with combined linear and sparsity constraints. We demonstrate the utility and flexibility of the approach through dose-response experiments with constraints on safety, efficacy, and cost, where cost also depends on the number of distinct doses used. |
| 2025-07-07 | [Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message](http://arxiv.org/abs/2507.04673v1) | Wei Duan, Li Qian | The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity. |
| 2025-07-07 | [VaxPulse: Monitoring of Online Public Concerns to Enhance Post-licensure Vaccine Surveillance](http://arxiv.org/abs/2507.04656v1) | Muhammad Javed, Sedigh Khademi et al. | The recent vaccine-related infodemic has amplified public concerns, highlighting the need for proactive misinformation management. We describe how we enhanced the reporting surveillance system of Victoria's vaccine safety service, SAEFVIC, through the incorporation of new information sources for public sentiment analysis, topics of discussion, and hesitancies about vaccinations online. Using VaxPulse, a multi-step framework, we integrate adverse events following immunisation (AEFI) with sentiment analysis, demonstrating the importance of contextualising public concerns. Additionally, we emphasise the need to address non-English languages to stratify concerns across ethno-lingual communities, providing valuable insights for vaccine uptake strategies and combating mis/disinformation. The framework is applied to real-world examples and a case study on women's vaccine hesitancy, showcasing its benefits and adaptability by identifying public opinion from online media. |
| 2025-07-07 | [Accelerated Online Reinforcement Learning using Auxiliary Start State Distributions](http://arxiv.org/abs/2507.04606v1) | Aman Mehra, Alexandre Capone et al. | A long-standing problem in online reinforcement learning (RL) is of ensuring sample efficiency, which stems from an inability to explore environments efficiently. Most attempts at efficient exploration tackle this problem in a setting where learning begins from scratch, without prior information available to bootstrap learning. However, such approaches fail to leverage expert demonstrations and simulators that can reset to arbitrary states. These affordances are valuable resources that offer enormous potential to guide exploration and speed up learning. In this paper, we explore how a small number of expert demonstrations and a simulator allowing arbitrary resets can accelerate learning during online RL. We find that training with a suitable choice of an auxiliary start state distribution that may differ from the true start state distribution of the underlying Markov Decision Process can significantly improve sample efficiency. We find that using a notion of safety to inform the choice of this auxiliary distribution significantly accelerates learning. By using episode length information as a way to operationalize this notion, we demonstrate state-of-the-art sample efficiency on a sparse-reward hard-exploration environment. |
| 2025-07-06 | [A Data-Driven Novelty Score for Diverse In-Vehicle Data Recording](http://arxiv.org/abs/2507.04529v1) | Philipp Reis, Joshua Ransiek et al. | High-quality datasets are essential for training robust perception systems in autonomous driving. However, real-world data collection is often biased toward common scenes and objects, leaving novel cases underrepresented. This imbalance hinders model generalization and compromises safety. The core issue is the curse of rarity. Over time, novel events occur infrequently, and standard logging methods fail to capture them effectively. As a result, large volumes of redundant data are stored, while critical novel cases are diluted, leading to biased datasets. This work presents a real-time data selection method focused on object-level novelty detection to build more balanced and diverse datasets. The method assigns a data-driven novelty score to image frames using a novel dynamic Mean Shift algorithm. It models normal content based on mean and covariance statistics to identify frames with novel objects, discarding those with redundant elements. The main findings show that reducing the training dataset size with this method can improve model performance, whereas higher redundancy tends to degrade it. Moreover, as data redundancy increases, more aggressive filtering becomes both possible and beneficial. While random sampling can offer some gains, it often leads to overfitting and unpredictability in outcomes. The proposed method supports real-time deployment with 32 frames per second and is constant over time. By continuously updating the definition of normal content, it enables efficient detection of novelties in a continuous data stream. |
| 2025-07-06 | [Verification of Visual Controllers via Compositional Geometric Transformations](http://arxiv.org/abs/2507.04523v1) | Alexander Estornell, Leonard Jung et al. | Perception-based neural network controllers are increasingly used in autonomous systems that rely on visual inputs to operate in the real world. Ensuring the safety of such systems under uncertainty is challenging. Existing verification techniques typically focus on Lp-bounded perturbations in the pixel space, which fails to capture the low-dimensional structure of many real-world effects. In this work, we introduce a novel verification framework for perception-based controllers that can generate outer-approximations of reachable sets through explicitly modeling uncertain observations with geometric perturbations. Our approach constructs a boundable mapping from states to images, enabling the use of state-based verification tools while accounting for uncertainty in perception. We provide theoretical guarantees on the soundness of our method and demonstrate its effectiveness across benchmark control environments. This work provides a principled framework for certifying the safety of perception-driven control systems under realistic visual perturbations. |

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



