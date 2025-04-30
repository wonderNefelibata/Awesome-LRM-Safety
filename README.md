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
| 2025-04-29 | [AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security](http://arxiv.org/abs/2504.20965v1) | Zikui Cai, Shayan Shabihi et al. | We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm |
| 2025-04-29 | [A Domain-Agnostic Scalable AI Safety Ensuring Framework](http://arxiv.org/abs/2504.20924v1) | Beomjun Kim, Kangyeon Kim et al. | Ensuring the safety of AI systems has recently emerged as a critical priority for real-world deployment, particularly in physical AI applications. Current approaches to AI safety typically address predefined domain-specific safety conditions, limiting their ability to generalize across contexts.   We propose a novel AI safety framework that ensures AI systems comply with \textbf{any user-defined constraint}, with \textbf{any desired probability}, and across \textbf{various domains}.   In this framework, we combine an AI component (e.g., neural network) with an optimization problem to produce responses that minimize objectives while satisfying user-defined constraints with probabilities exceeding user-defined thresholds. For credibility assessment of the AI component, we propose \textit{internal test data}, a supplementary set of safety-labeled data, and a \textit{conservative testing} methodology that provides statistical validity of using internal test data. We also present an approximation method of a loss function and how to compute its gradient for training.   We mathematically prove that probabilistic constraint satisfaction is guaranteed under specific, mild conditions and prove a scaling law between safety and the number of internal test data. We demonstrate our framework's effectiveness through experiments in diverse domains: demand prediction for production decision, safe reinforcement learning within the SafetyGym simulator, and guarding AI chatbot outputs. Through these experiments, we demonstrate that our method guarantees safety for user-specified constraints, outperforms {for \textbf{up to several order of magnitudes}} existing methods in low safety threshold regions, and scales effectively with respect to the size of internal test data. |
| 2025-04-29 | [When Testing AI Tests Us: Safeguarding Mental Health on the Digital Frontlines](http://arxiv.org/abs/2504.20910v1) | Sachin R. Pendse, Darren Gergle et al. | Red-teaming is a core part of the infrastructure that ensures that AI models do not produce harmful content. Unlike past technologies, the black box nature of generative AI systems necessitates a uniquely interactional mode of testing, one in which individuals on red teams actively interact with the system, leveraging natural language to simulate malicious actors and solicit harmful outputs. This interactional labor done by red teams can result in mental health harms that are uniquely tied to the adversarial engagement strategies necessary to effectively red team. The importance of ensuring that generative AI models do not propagate societal or individual harm is widely recognized -- one less visible foundation of end-to-end AI safety is also the protection of the mental health and wellbeing of those who work to keep model outputs safe. In this paper, we argue that the unmet mental health needs of AI red-teamers is a critical workplace safety concern. Through analyzing the unique mental health impacts associated with the labor done by red teams, we propose potential individual and organizational strategies that could be used to meet these needs, and safeguard the mental health of red-teamers. We develop our proposed strategies through drawing parallels between common red-teaming practices and interactional labor common to other professions (including actors, mental health professionals, conflict photographers, and content moderators), describing how individuals and organizations within these professional spaces safeguard their mental health given similar psychological demands. Drawing on these protective practices, we describe how safeguards could be adapted for the distinct mental health challenges experienced by red teaming organizations as they mitigate emerging technological risks on the new digital frontlines. |
| 2025-04-29 | [GiBy: A Giant-Step Baby-Step Classifier For Anomaly Detection In Industrial Control Systems](http://arxiv.org/abs/2504.20906v1) | Sarad Venugopalan, Sridhar Adepu | The continuous monitoring of the interactions between cyber-physical components of any industrial control system (ICS) is required to secure automation of the system controls, and to guarantee plant processes are fail-safe and remain in an acceptably safe state. Safety is achieved by managing actuation (where electric signals are used to trigger physical movement), dependent on corresponding sensor readings; used as ground truth in decision making. Timely detection of anomalies (attacks, faults and unascertained states) in ICSs is crucial for the safe running of a plant, the safety of its personnel, and for the safe provision of any services provided. We propose an anomaly detection method that involves accurate linearization of the non-linear forms arising from sensor-actuator(s) relationships, primarily because solving linear models is easier and well understood. Further, the time complexity of the anomaly detection scenario/problem at hand is lowered using dimensionality reduction of the actuator(s) in relationship with a sensor. We accomplish this by using a well-known water treatment testbed as a use case. Our experiments show millisecond time response to detect anomalies and provide explainability; that are not simultaneously achieved by other state of the art AI/ML models with eXplainable AI (XAI) used for the same purpose. Further, we pin-point the sensor(s) and its actuation state for which anomaly was detected. |
| 2025-04-29 | [GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion](http://arxiv.org/abs/2504.20829v1) | Jiaxin Hong, Sixu Chen et al. | As 3D Gaussian Splatting (3DGS) emerges as a breakthrough in scene representation and novel view synthesis, its rapid adoption in safety-critical domains (e.g., autonomous systems, AR/VR) urgently demands scrutiny of potential security vulnerabilities. This paper presents the first systematic study of backdoor threats in 3DGS pipelines. We identify that adversaries may implant backdoor views to induce malicious scene confusion during inference, potentially leading to environmental misperception in autonomous navigation or spatial distortion in immersive environments. To uncover this risk, we propose GuassTrap, a novel poisoning attack method targeting 3DGS models. GuassTrap injects malicious views at specific attack viewpoints while preserving high-quality rendering in non-target views, ensuring minimal detectability and maximizing potential harm. Specifically, the proposed method consists of a three-stage pipeline (attack, stabilization, and normal training) to implant stealthy, viewpoint-consistent poisoned renderings in 3DGS, jointly optimizing attack efficacy and perceptual realism to expose security risks in 3D rendering. Extensive experiments on both synthetic and real-world datasets demonstrate that GuassTrap can effectively embed imperceptible yet harmful backdoor views while maintaining high-quality rendering in normal views, validating its robustness, adaptability, and practical applicability. |
| 2025-04-29 | [DP-SMOTE: Integrating Differential Privacy and Oversampling Technique to Preserve Privacy in Smart Homes](http://arxiv.org/abs/2504.20827v1) | Amr Tarek Elsayed, Almohammady Sobhi Alsharkawy et al. | Smart homes represent intelligent environments where interconnected devices gather information, enhancing users living experiences by ensuring comfort, safety, and efficient energy management. To enhance the quality of life, companies in the smart device industry collect user data, including activities, preferences, and power consumption. However, sharing such data necessitates privacy-preserving practices. This paper introduces a robust method for secure sharing of data to service providers, grounded in differential privacy (DP). This empowers smart home residents to contribute usage statistics while safeguarding their privacy. The approach incorporates the Synthetic Minority Oversampling technique (SMOTe) and seamlessly integrates Gaussian noise to generate synthetic data, enabling data and statistics sharing while preserving individual privacy. The proposed method employs the SMOTe algorithm and applies Gaussian noise to generate data. Subsequently, it employs a k-anonymity function to assess reidentification risk before sharing the data. The simulation outcomes demonstrate that our method delivers strong performance in safeguarding privacy and in accuracy, recall, and f-measure metrics. This approach is particularly effective in smart homes, offering substantial utility in privacy at a reidentification risk of 30%, with Gaussian noise set to 0.3, SMOTe at 500%, and the application of a k-anonymity function with k = 2. Additionally, it shows a high classification accuracy, ranging from 90% to 98%, across various classification techniques. |
| 2025-04-29 | [In defence of post-hoc explanations in medical AI](http://arxiv.org/abs/2504.20741v1) | Joshua Hatherley, Lauritz Munch et al. | Since the early days of the Explainable AI movement, post-hoc explanations have been praised for their potential to improve user understanding, promote trust, and reduce patient safety risks in black box medical AI systems. Recently, however, critics have argued that the benefits of post-hoc explanations are greatly exaggerated since they merely approximate, rather than replicate, the actual reasoning processes that black box systems take to arrive at their outputs. In this article, we aim to defend the value of post-hoc explanations against this recent critique. We argue that even if post-hoc explanations do not replicate the exact reasoning processes of black box systems, they can still improve users' functional understanding of black box systems, increase the accuracy of clinician-AI teams, and assist clinicians in justifying their AI-informed decisions. While post-hoc explanations are not a "silver bullet" solution to the black box problem in medical AI, we conclude that they remain a useful strategy for addressing the black box problem in medical AI. |
| 2025-04-29 | [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](http://arxiv.org/abs/2504.20571v1) | Yiping Wang, Qing Yang et al. | We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6%, and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7%. This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which includes the aforementioned example. Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples (many of which yield approximately 30% or greater improvement on MATH500 when employed as a single training example). In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-domain generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by adding entropy loss with an appropriate coefficient) in 1-shot RLVR training. As a bonus, we observe that applying entropy loss alone, without any outcome reward, significantly enhances Qwen2.5-Math-1.5B's performance on MATH500 by 27.4%. These findings can inspire future work on RLVR data efficiency and encourage a re-examination of both recent progress and the underlying mechanisms in RLVR. Our code, model, and data are open source at https://github.com/ypwang61/One-Shot-RLVR |
| 2025-04-29 | [Safe Bottom-Up Flexibility Provision from Distributed Energy Resources](http://arxiv.org/abs/2504.20529v1) | Costas Mylonas, Emmanouel Varvarigos et al. | Modern renewables-based power systems need to tap on the flexibility of Distributed Energy Resources (DERs) connected to distribution networks. It is important, however, that DER owners/users remain in control of their assets, decisions, and objectives. At the same time, the dynamic landscape of DER-penetrated distribution networks calls for agile, data-driven flexibility management frameworks. In the face of these developments, the Multi-Agent Reinforcement Learning (MARL) paradigm is gaining significant attention, as a distributed and data-driven decision-making policy. This paper addresses the need for bottom-up DER management decisions to account for the distribution network's safety-related constraints. While the related literature on safe MARL typically assumes that network characteristics are available and incorporated into the policy's safety layer, which implies active DSO engagement, this paper ensures that self-organized DER communities are enabled to provide distribution-network-safe flexibility services without relying on the aspirational and problematic requirement of bringing the DSO in the decision-making loop. |
| 2025-04-29 | [Token-Efficient Prompt Injection Attack: Provoking Cessation in LLM Reasoning via Adaptive Token Compression](http://arxiv.org/abs/2504.20493v1) | Yu Cui, Yujun Cai et al. | While reasoning large language models (LLMs) demonstrate remarkable performance across various tasks, they also contain notable security vulnerabilities. Recent research has uncovered a "thinking-stopped" vulnerability in DeepSeek-R1, where model-generated reasoning tokens can forcibly interrupt the inference process, resulting in empty responses that compromise LLM-integrated applications. However, existing methods triggering this vulnerability require complex mathematical word problems with long prompts--even exceeding 5,000 tokens. To reduce the token cost and formally define this vulnerability, we propose a novel prompt injection attack named "Reasoning Interruption Attack", based on adaptive token compression. We demonstrate that simple standalone arithmetic tasks can effectively trigger this vulnerability, and the prompts based on such tasks exhibit simpler logical structures than mathematical word problems. We develop a systematic approach to efficiently collect attack prompts and an adaptive token compression framework that utilizes LLMs to automatically compress these prompts. Experiments show our compression framework significantly reduces prompt length while maintaining effective attack capabilities. We further investigate the attack's performance via output prefix and analyze the underlying causes of the vulnerability, providing valuable insights for improving security in reasoning LLMs. |
| 2025-04-29 | [An Algebraic Approach to Asymmetric Delegation and Polymorphic Label Inference (Technical Report)](http://arxiv.org/abs/2504.20432v1) | Silei Ren, Coşku Acay et al. | Language-based information flow control (IFC) enables reasoning about and enforcing security policies in decentralized applications. While information flow properties are relatively extensional and compositional, designing expressive systems that enforce such properties remains challenging. In particular, it can be difficult to use IFC labels to model certain security assumptions, such as semi-honest agents.   Motivated by these modeling limitations, we study the algebraic semantics of lattice-based IFC label models, and propose a semantic framework that allows formalizing asymmetric delegation, which is partial delegation of confidentiality or integrity. Our framework supports downgrading of information and ensures their safety through nonmalleable information flow (NMIF).   To demonstrate the practicality of our framework, we design and implement a novel algorithm that statically checks NMIF and a label inference procedure that efficiently supports bounded label polymorphism, allowing users to write code generic with respect to labels. |
| 2025-04-29 | [Safe and Optimal N-Spacecraft Swarm Reconfiguration in Non-Keplerian Cislunar Orbits](http://arxiv.org/abs/2504.20386v1) | Yuji Takubo, Walter Manuel et al. | This paper presents a novel fuel-optimal guidance and control methodology for spacecraft swarm reconfiguration in Restricted Multi-Body Problems (RMBPs) with a guarantee of passive safety, maintaining miss distance even under abrupt loss of control authority. A new set of constraints exploits a quasi-periodic structure of RMBPs to guarantee passive safety. Particularly, this can be expressed as simple geometric constraints by solving optimal control in Local Toroidal Coordinates, which is based on a local eigenspace of a quasi-periodic motion around the corresponding periodic orbit. The proposed formulation enables a significant simplification of problem structure, which is highly applicable to large-scale swarm reconfiguration in cislunar orbits. The method is demonstrated in various models of RMBPs (Elliptical Restricted Three-Body Problem and Bi-Circular Restricted Four-Body Problem) and also validated in the full-ephemeris dynamics. By extending and generalizing well-known concepts from the two- to the three- and four-body problems, this paper lays the foundation for the practical control schemes of relative motion in cislunar space. |
| 2025-04-29 | [Inception: Jailbreak the Memory Mechanism of Text-to-Image Generation Systems](http://arxiv.org/abs/2504.20376v1) | Shiqian Zhao, Jiayang Liu et al. | Currently, the memory mechanism has been widely and successfully exploited in online text-to-image (T2I) generation systems ($e.g.$, DALL$\cdot$E 3) for alleviating the growing tokenization burden and capturing key information in multi-turn interactions. Despite its practicality, its security analyses have fallen far behind. In this paper, we reveal that this mechanism exacerbates the risk of jailbreak attacks. Different from previous attacks that fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or may generate non-unsafe images due to under- or over-optimization, we propose Inception, the first multi-turn jailbreak attack against the memory mechanism in real-world text-to-image generation systems. Inception embeds the malice at the inception of the chat session turn by turn, leveraging the mechanism that T2I generation systems retrieve key information in their memory. Specifically, Inception mainly consists of two modules. It first segments the unsafe prompt into chunks, which are subsequently fed to the system in multiple turns, serving as pseudo-gradients for directive optimization. Specifically, we develop a series of segmentation policies that ensure the images generated are semantically consistent with the target prompt. Secondly, after segmentation, to overcome the challenge of the inseparability of minimum unsafe words, we propose recursion, a strategy that makes minimum unsafe words subdivisible. Collectively, segmentation and recursion ensure that all the request prompts are benign but can lead to malicious outcomes. We conduct experiments on the real-world text-to-image generation system ($i.e.$, DALL$\cdot$E 3) to validate the effectiveness of Inception. The results indicate that Inception surpasses the state-of-the-art by a 14\% margin in attack success rate. |
| 2025-04-28 | [Online Safety for All: Sociocultural Insights from a Systematic Review of Youth Online Safety in the Global South](http://arxiv.org/abs/2504.20308v1) | Ozioma C. Oguine, Oghenemaro Anuyah et al. | Youth online safety research in HCI has historically centered on perspectives from the Global North, often overlooking the unique particularities and cultural contexts of regions in the Global South. This paper presents a systematic review of 66 youth online safety studies published between 2014 and 2024, specifically focusing on regions in the Global South. Our findings reveal a concentrated research focus in Asian countries and predominance of quantitative methods. We also found limited research on marginalized youth populations and a primary focus on risks related to cyberbullying. Our analysis underscores the critical role of cultural factors in shaping online safety, highlighting the need for educational approaches that integrate social dynamics and awareness. We propose methodological recommendations and a future research agenda that encourages the adoption of situated, culturally sensitive methodologies and youth-centered approaches to researching youth online safety regions in the Global South. This paper advocates for greater inclusivity in youth online safety research, emphasizing the importance of addressing varied sociocultural contexts to better understand and meet the online safety needs of youth in the Global South. |
| 2025-04-28 | [Improving trajectory continuity in drone-based crowd monitoring using a set of minimal-cost techniques and deep discriminative correlation filters](http://arxiv.org/abs/2504.20234v1) | Bartosz Ptak, Marek Kraft | Drone-based crowd monitoring is the key technology for applications in surveillance, public safety, and event management. However, maintaining tracking continuity and consistency remains a significant challenge. Traditional detection-assignment tracking methods struggle with false positives, false negatives, and frequent identity switches, leading to degraded counting accuracy and making in-depth analysis impossible. This paper introduces a point-oriented online tracking algorithm that improves trajectory continuity and counting reliability in drone-based crowd monitoring. Our method builds on the Simple Online and Real-time Tracking (SORT) framework, replacing the original bounding-box assignment with a point-distance metric. The algorithm is enhanced with three cost-effective techniques: camera motion compensation, altitude-aware assignment, and classification-based trajectory validation. Further, Deep Discriminative Correlation Filters (DDCF) that re-use spatial feature maps from localisation algorithms for increased computational efficiency through neural network resource sharing are integrated to refine object tracking by reducing noise and handling missed detections. The proposed method is evaluated on the DroneCrowd and newly shared UP-COUNT-TRACK datasets, demonstrating substantial improvements in tracking metrics, reducing counting errors to 23% and 15%, respectively. The results also indicate a significant reduction of identity switches while maintaining high tracking accuracy, outperforming baseline online trackers and even an offline greedy optimisation method. |
| 2025-04-28 | [Representation Learning on a Random Lattice](http://arxiv.org/abs/2504.20197v1) | Aryeh Brill | Decomposing a deep neural network's learned representations into interpretable features could greatly enhance its safety and reliability. To better understand features, we adopt a geometric perspective, viewing them as a learned coordinate system for mapping an embedded data distribution. We motivate a model of a generic data distribution as a random lattice and analyze its properties using percolation theory. Learned features are categorized into context, component, and surface features. The model is qualitatively consistent with recent findings in mechanistic interpretability and suggests directions for future research. |
| 2025-04-28 | [Cybersecurity for Autonomous Vehicles](http://arxiv.org/abs/2504.20180v1) | Sai varun reddy Bhemavarapu | The increasing adoption of autonomous vehicles is bringing a major shift in the automotive industry. However, as these vehicles become more connected, cybersecurity threats have emerged as a serious concern. Protecting the security and integrity of autonomous systems is essential to prevent malicious activities that can harm passengers, other road users, and the overall transportation network. This paper focuses on addressing the cybersecurity issues in autonomous vehicles by examining the challenges and risks involved, which are important for building a secure future. Since autonomous vehicles depend on the communication between sensors, artificial intelligence, external infrastructure, and other systems, they are exposed to different types of cyber threats. A cybersecurity breach in an autonomous vehicle can cause serious problems, including a loss of public trust and safety. Therefore, it is very important to develop and apply strong cybersecurity measures to support the growth and acceptance of self-driving cars. This paper discusses major cybersecurity challenges like vulnerabilities in software and hardware, risks from wireless communication, and threats through external interfaces. It also reviews existing solutions such as secure software development, intrusion detection systems, cryptographic protocols, and anomaly detection methods. Additionally, the paper highlights the role of regulatory bodies, industry collaborations, and cybersecurity standards in creating a secure environment for autonomous vehicles. Setting clear rules and best practices is necessary for consistent protection across manufacturers and regions. By analyzing the current cybersecurity landscape and suggesting practical countermeasures, this paper aims to contribute to the safe development and public trust of autonomous vehicle technology. |
| 2025-04-28 | [Modular Machine Learning: An Indispensable Path towards New-Generation Large Language Models](http://arxiv.org/abs/2504.20020v1) | Xin Wang, Haoyang Li et al. | Large language models (LLMs) have dramatically advanced machine learning research including natural language processing, computer vision, data mining, etc., yet they still exhibit critical limitations in reasoning, factual consistency, and interpretability. In this paper, we introduce a novel learning paradigm -- Modular Machine Learning (MML) -- as an essential approach toward new-generation LLMs. MML decomposes the complex structure of LLMs into three interdependent components: modular representation, modular model, and modular reasoning, aiming to enhance LLMs' capability of counterfactual reasoning, mitigating hallucinations, as well as promoting fairness, safety, and transparency. Specifically, the proposed MML paradigm can: i) clarify the internal working mechanism of LLMs through the disentanglement of semantic components; ii) allow for flexible and task-adaptive model design; iii) enable interpretable and logic-driven decision-making process. We present a feasible implementation of MML-based LLMs via leveraging advanced techniques such as disentangled representation learning, neural architecture search and neuro-symbolic learning. We critically identify key challenges, such as the integration of continuous neural and discrete symbolic processes, joint optimization, and computational scalability, present promising future research directions that deserve further exploration. Ultimately, the integration of the MML paradigm with LLMs has the potential to bridge the gap between statistical (deep) learning and formal (logical) reasoning, thereby paving the way for robust, adaptable, and trustworthy AI systems across a wide range of real-world applications. |
| 2025-04-28 | [Socially-Aware Autonomous Driving: Inferring Yielding Intentions for Safer Interactions](http://arxiv.org/abs/2504.20004v1) | Jing Wang, Yan Jin et al. | Since the emergence of autonomous driving technology, it has advanced rapidly over the past decade. It is becoming increasingly likely that autonomous vehicles (AVs) would soon coexist with human-driven vehicles (HVs) on the roads. Currently, safety and reliable decision-making remain significant challenges, particularly when AVs are navigating lane changes and interacting with surrounding HVs. Therefore, precise estimation of the intentions of surrounding HVs can assist AVs in making more reliable and safe lane change decision-making. This involves not only understanding their current behaviors but also predicting their future motions without any direct communication. However, distinguishing between the passing and yielding intentions of surrounding HVs still remains ambiguous. To address the challenge, we propose a social intention estimation algorithm rooted in Directed Acyclic Graph (DAG), coupled with a decision-making framework employing Deep Reinforcement Learning (DRL) algorithms. To evaluate the method's performance, the proposed framework can be tested and applied in a lane-changing scenario within a simulated environment. Furthermore, the experiment results demonstrate how our approach enhances the ability of AVs to navigate lane changes safely and efficiently on roads. |
| 2025-04-28 | [Mitigating Societal Cognitive Overload in the Age of AI: Challenges and Directions](http://arxiv.org/abs/2504.19990v1) | Salem Lahlou | Societal cognitive overload, driven by the deluge of information and complexity in the AI age, poses a critical challenge to human well-being and societal resilience. This paper argues that mitigating cognitive overload is not only essential for improving present-day life but also a crucial prerequisite for navigating the potential risks of advanced AI, including existential threats. We examine how AI exacerbates cognitive overload through various mechanisms, including information proliferation, algorithmic manipulation, automation anxieties, deregulation, and the erosion of meaning. The paper reframes the AI safety debate to center on cognitive overload, highlighting its role as a bridge between near-term harms and long-term risks. It concludes by discussing potential institutional adaptations, research directions, and policy considerations that arise from adopting an overload-resilient perspective on human-AI alignment, suggesting pathways for future exploration rather than prescribing definitive solutions. |

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



