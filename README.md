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
| 2025-11-07 | [ETHOS: A Robotic Encountered-Type Haptic Display for Social Interaction in Virtual Reality](http://arxiv.org/abs/2511.05379v1) | Eric Godden, Jacquie Groenewegen et al. | We present ETHOS (Encountered-Type Haptics for On-demand Social Interaction), a dynamic encountered-type haptic display (ETHD) that enables natural physical contact in virtual reality (VR) during social interactions such as handovers, fist bumps, and high-fives. The system integrates a torque-controlled robotic manipulator with interchangeable passive props (silicone hand replicas and a baton), marker-based physical-virtual registration via a ChArUco board, and a safety monitor that gates motion based on the user's head and hand pose. We introduce two control strategies: (i) a static mode that presents a stationary prop aligned with its virtual counterpart, consistent with prior ETHD baselines, and (ii) a dynamic mode that continuously updates prop position by exponentially blending an initial mid-point trajectory with real-time hand tracking, generating a unique contact point for each interaction. Bench tests show static colocation accuracy of 5.09 +/- 0.94 mm, while user interactions achieved temporal alignment with an average contact latency of 28.53 +/- 31.21 ms across all interaction and control conditions. These results demonstrate the feasibility of recreating socially meaningful haptics in VR. By incorporating essential safety and control mechanisms, ETHOS establishes a practical foundation for high-fidelity, dynamic interpersonal interactions in virtual environments. |
| 2025-11-07 | [ConVerse: Benchmarking Contextual Safety in Agent-to-Agent Conversations](http://arxiv.org/abs/2511.05359v1) | Amr Gomaa, Ahmed Salem et al. | As language models evolve into autonomous agents that act and communicate on behalf of users, ensuring safety in multi-agent ecosystems becomes a central challenge. Interactions between personal assistants and external service providers expose a core tension between utility and protection: effective collaboration requires information sharing, yet every exchange creates new attack surfaces. We introduce ConVerse, a dynamic benchmark for evaluating privacy and security risks in agent-agent interactions. ConVerse spans three practical domains (travel, real estate, insurance) with 12 user personas and over 864 contextually grounded attacks (611 privacy, 253 security). Unlike prior single-agent settings, it models autonomous, multi-turn agent-to-agent conversations where malicious requests are embedded within plausible discourse. Privacy is tested through a three-tier taxonomy assessing abstraction quality, while security attacks target tool use and preference manipulation. Evaluating seven state-of-the-art models reveals persistent vulnerabilities; privacy attacks succeed in up to 88% of cases and security breaches in up to 60%, with stronger models leaking more. By unifying privacy and security within interactive multi-agent contexts, ConVerse reframes safety as an emergent property of communication. |
| 2025-11-07 | [SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning](http://arxiv.org/abs/2511.05355v1) | Tzu-Yuan Huang, Armin Lederer et al. | Flow matching (FM) has shown promising results in data-driven planning. However, it inherently lacks formal guarantees for ensuring state and action constraints, whose satisfaction is a fundamental and crucial requirement for the safety and admissibility of planned trajectories on various systems. Moreover, existing FM planners do not ensure the dynamical consistency, which potentially renders trajectories inexecutable. We address these shortcomings by proposing SAD-Flower, a novel framework for generating Safe, Admissible, and Dynamically consistent trajectories. Our approach relies on an augmentation of the flow with a virtual control input. Thereby, principled guidance can be derived using techniques from nonlinear control theory, providing formal guarantees for state constraints, action constraints, and dynamic consistency. Crucially, SAD-Flower operates without retraining, enabling test-time satisfaction of unseen constraints. Through extensive experiments across several tasks, we demonstrate that SAD-Flower outperforms various generative-model-based baselines in ensuring constraint satisfaction. |
| 2025-11-07 | [Force-Safe Environment Maps and Real-Time Detection for Soft Robot Manipulators](http://arxiv.org/abs/2511.05307v1) | Akua K. Dickson, Juan C. Pacheco Garcia et al. | Soft robot manipulators have the potential for deployment in delicate environments to perform complex manipulation tasks. However, existing obstacle detection and avoidance methods do not consider limits on the forces that manipulators may exert upon contact with delicate obstacles. This work introduces a framework that maps force safety criteria from task space (i.e. positions along the robot's body) to configuration space (i.e. the robot's joint angles) and enables real-time force safety detection. We incorporate limits on allowable environmental contact forces for given task-space obstacles, and map them into configuration space (C-space) through the manipulator's forward kinematics. This formulation ensures that configurations classified as safe are provably below the maximum force thresholds, thereby allowing us to determine force-safe configurations of the soft robot manipulator in real-time. We validate our approach in simulation and hardware experiments on a two-segment pneumatic soft robot manipulator. Results demonstrate that the proposed method accurately detects force safety during interactions with deformable obstacles, thereby laying the foundation for real-time safe planning of soft manipulators in delicate, cluttered environments. |
| 2025-11-07 | [TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems](http://arxiv.org/abs/2511.05269v1) | Ishan Kavathekar, Hemang Jain et al. | Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems. |
| 2025-11-07 | [Poissonian Analysis of Glitches Observed in the LIGO Gravitational Wave Interferometers](http://arxiv.org/abs/2511.05244v1) | Giovanna Souza Rodrigues Costa, Julio Cesar Martins et al. | This work investigates the temporal distribution of glitches detected by LIGO, focusing on the morphological classification provided by the Gravity Spy project. Starting from the hypothesis that these events follow a Poisson process, we developed a statistical methodology to evaluate the agreement between the empirical distribution of glitches and an ideal Poisson model, using the coefficient of determination ($R^2$) as the main metric. The analysis was applied to real data from the LIGO detectors in Livingston and Hanford throughout the O3 run, as well as to synthetic datasets generated from pure Poisson distributions. The results show that while several morphologies exhibit good agreement with the proposed model, classes such as 1400Ripples, Fast Scattering, and Power Line display significant deviations ($R^2 \leq 0.6$), suggesting that their origins do not strictly follow Poissonian statistics. In some cases, a dependence on the detector or the observing run was also observed. This analysis provides a quantitative basis for distinguishing glitch classes based on their degree of "Poissonness", potentially supporting the development of more effective glitch mitigation strategies in gravitational wave detector data. |
| 2025-11-07 | [TRICK: Time and Range Integrity ChecK using Low Earth Orbiting Satellite for Securing GNSS](http://arxiv.org/abs/2511.05100v1) | Arslan Mumtaz, Mridula Singh | Global Navigation Satellite Systems (GNSS) provide Positioning, Navigation, and Timing (PNT) information to over 4 billion devices worldwide. Despite its pervasive use in safety critical and high precision applications, GNSS remains vulnerable to spoofing attacks. Cryptographic enhancements, such as the use of TESLA protocol in Galileo, to provide navigation message authentication do not mitigate time of arrival manipulations. In this paper, we propose TRICK, a primitive for secure positioning that closes this gap by introducing a fundamentally new approach that only requires two way communications with a single reference node along with multiple broadcast signals. Unlike classical Verifiable Multilateration (VM), which requires establishing two way communication with each reference nodes, our solution relies on only two measurements with a trusted Low Earth Orbiting (LEO) satellite and combines broadcast navigation signals. We rigorously prove that combining the LEO satellite based two way range measurements and multiple one way ranges such as from broadcast signals of GNSS into ellipsoidal constraint restores the same guarantees as offered by VM whilst using minimal infrastructure and message exchanges. Through detailed analysis, we show that our approach reliably detects spoofing attempts while adding negligible computation overhead. |
| 2025-11-07 | [On Text Simplification Metrics and General-Purpose LLMs for Accessible Health Information, and A Potential Architectural Advantage of The Instruction-Tuned LLM class](http://arxiv.org/abs/2511.05080v1) | P. Bilha Githinji, Aikaterini Meilliou et al. | The increasing health-seeking behavior and digital consumption of biomedical information by the general public necessitate scalable solutions for automatically adapting complex scientific and technical documents into plain language. Automatic text simplification solutions, including advanced large language models, however, continue to face challenges in reliably arbitrating the tension between optimizing readability performance and ensuring preservation of discourse fidelity. This report empirically assesses the performance of two major classes of general-purpose LLMs, demonstrating their linguistic capabilities and foundational readiness for the task compared to a human benchmark. Using a comparative analysis of the instruction-tuned Mistral 24B and the reasoning-augmented QWen2.5 32B, we identify a potential architectural advantage in the instruction-tuned LLM. Mistral exhibits a tempered lexical simplification strategy that enhances readability across a suite of metrics and the simplification-specific formula SARI (mean 42.46), while preserving human-level discourse with a BERTScore of 0.91. QWen also attains enhanced readability performance, but its operational strategy shows a disconnect in balancing between readability and accuracy, reaching a statistically significantly lower BERTScore of 0.89. Additionally, a comprehensive correlation analysis of 21 metrics spanning readability, discourse fidelity, content safety, and underlying distributional measures for mechanistic insights, confirms strong functional redundancies among five readability indices. This empirical evidence tracks baseline performance of the evolving LLMs for the task of text simplification, identifies the instruction-tuned Mistral 24B for simplification, provides necessary heuristics for metric selection, and points to lexical support as a primary domain-adaptation issue for simplification. |
| 2025-11-07 | [SurgiATM: A Physics-Guided Plug-and-Play Model for Deep Learning-Based Smoke Removal in Laparoscopic Surgery](http://arxiv.org/abs/2511.05059v1) | Mingyu Sheng, Jianan Fan et al. | During laparoscopic surgery, smoke generated by tissue cauterization can significantly degrade the visual quality of endoscopic frames, increasing the risk of surgical errors and hindering both clinical decision-making and computer-assisted visual analysis. Consequently, removing surgical smoke is critical to ensuring patient safety and maintaining operative efficiency. In this study, we propose the Surgical Atmospheric Model (SurgiATM) for surgical smoke removal. SurgiATM statistically bridges a physics-based atmospheric model and data-driven deep learning models, combining the superior generalizability of the former with the high accuracy of the latter. Furthermore, SurgiATM is designed as a lightweight, plug-and-play module that can be seamlessly integrated into diverse surgical desmoking architectures to enhance their accuracy and stability, better meeting clinical requirements. It introduces only two hyperparameters and no additional trainable weights, preserving the original network architecture with minimal computational and modification overhead. We conduct extensive experiments on three public surgical datasets with ten desmoking methods, involving multiple network architectures and covering diverse procedures, including cholecystectomy, partial nephrectomy, and diaphragm dissection. The results demonstrate that incorporating SurgiATM commonly reduces the restoration errors of existing models and relatively enhances their generalizability, without adding any trainable layers or weights. This highlights the convenience, low cost, effectiveness, and generalizability of the proposed method. The code for SurgiATM is released at https://github.com/MingyuShengSMY/SurgiATM. |
| 2025-11-07 | [UA-Code-Bench: A Competitive Programming Benchmark for Evaluating LLM Code Generation in Ukrainian](http://arxiv.org/abs/2511.05040v1) | Mykyta Syromiatnikov, Victoria Ruvinskaya | Evaluating the real capabilities of large language models in low-resource languages still represents a challenge, as many existing benchmarks focus on widespread tasks translated from English or evaluate only simple language understanding. This paper introduces UA-Code-Bench, a new open-source benchmark established for a thorough evaluation of language models' code generation and competitive programming problem-solving abilities in Ukrainian. The benchmark comprises 500 problems from the Eolymp platform, evenly distributed across five complexity levels from very easy to very hard. A diverse set of 13 leading proprietary and open-source models, generating Python solutions based on a one-shot prompt, was evaluated via the dedicated Eolymp environment against hidden tests, ensuring code correctness. The obtained results reveal that even top-performing models, such as OpenAI o3 and GPT-5, solve only half of the problems, highlighting the challenge of code generation in low-resource natural language. Furthermore, this research presents a comprehensive analysis of performance across various difficulty levels, as well as an assessment of solution uniqueness and computational efficiency, measured by both elapsed time and memory consumption of the generated solutions. In conclusion, this work demonstrates the value of competitive programming benchmarks in evaluating large language models, especially in underrepresented languages. It also paves the way for future research on multilingual code generation and reasoning-enhanced models. The benchmark, data parsing, preparation, code generation, and evaluation scripts are available at https://huggingface.co/datasets/NLPForUA/ua-code-bench. |
| 2025-11-07 | [Pluralistic Behavior Suite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies](http://arxiv.org/abs/2511.05018v1) | Prasoon Varshney, Makesh Narsimhan Sreedhar et al. | Large language models (LLMs) are typically aligned to a universal set of safety and usage principles intended for broad public acceptability. Yet, real-world applications of LLMs often take place within organizational ecosystems shaped by distinctive corporate policies, regulatory requirements, use cases, brand guidelines, and ethical commitments. This reality highlights the need for rigorous and comprehensive evaluation of LLMs with pluralistic alignment goals, an alignment paradigm that emphasizes adaptability to diverse user values and needs. In this work, we present PLURALISTIC BEHAVIOR SUITE (PBSUITE), a dynamic evaluation suite designed to systematically assess LLMs' capacity to adhere to pluralistic alignment specifications in multi-turn, interactive conversations. PBSUITE consists of (1) a diverse dataset of 300 realistic LLM behavioral policies, grounded in 30 industries; and (2) a dynamic evaluation framework for stress-testing model compliance with custom behavioral specifications under adversarial conditions. Using PBSUITE, We find that leading open- and closed-source LLMs maintain robust adherence to behavioral policies in single-turn settings (less than 4% failure rates), but their compliance weakens substantially in multi-turn adversarial interactions (up to 84% failure rates). These findings highlight that existing model alignment and safety moderation methods fall short in coherently enforcing pluralistic behavioral policies in real-world LLM interactions. Our work contributes both the dataset and analytical framework to support future research toward robust and context-aware pluralistic alignment techniques. |
| 2025-11-07 | [Encoding Biomechanical Energy Margin into Passivity-based Synchronization for Networked Telerobotic Systems](http://arxiv.org/abs/2511.04994v1) | Xingyuan Zhou, Peter Paik et al. | Maintaining system stability and accurate position tracking is imperative in networked robotic systems, particularly for haptics-enabled human-robot interaction. Recent literature has integrated human biomechanics into the stabilizers implemented for teleoperation, enhancing force preservation while guaranteeing convergence and safety. However, position desynchronization due to imperfect communication and non-passive behaviors remains a challenge. This paper proposes a two-port biomechanics-aware passivity-based synchronizer and stabilizer, referred to as TBPS2. This stabilizer optimizes position synchronization by leveraging human biomechanics while reducing the stabilizer's conservatism in its activation. We provide the mathematical design synthesis of the stabilizer and the proof of stability. We also conducted a series of grid simulations and systematic experiments, comparing their performance with that of state-of-the-art solutions under varying time delays and environmental conditions. |
| 2025-11-07 | [Too Good to be Bad: On the Failure of LLMs to Role-Play Villains](http://arxiv.org/abs/2511.04962v1) | Zihao Yi, Qingxuan Jiang et al. | Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods. |
| 2025-11-07 | [Beta Distribution Learning for Reliable Roadway Crash Risk Assessment](http://arxiv.org/abs/2511.04886v1) | Ahmad Elallaf, Nathan Jacobs et al. | Roadway traffic accidents represent a global health crisis, responsible for over a million deaths annually and costing many countries up to 3% of their GDP. Traditional traffic safety studies often examine risk factors in isolation, overlooking the spatial complexity and contextual interactions inherent in the built environment. Furthermore, conventional Neural Network-based risk estimators typically generate point estimates without conveying model uncertainty, limiting their utility in critical decision-making. To address these shortcomings, we introduce a novel geospatial deep learning framework that leverages satellite imagery as a comprehensive spatial input. This approach enables the model to capture the nuanced spatial patterns and embedded environmental risk factors that contribute to fatal crash risks. Rather than producing a single deterministic output, our model estimates a full Beta probability distribution over fatal crash risk, yielding accurate and uncertainty-aware predictions--a critical feature for trustworthy AI in safety-critical applications. Our model outperforms baselines by achieving a 17-23% improvement in recall, a key metric for flagging potential dangers, while delivering superior calibration. By providing reliable and interpretable risk assessments from satellite imagery alone, our method enables safer autonomous navigation and offers a highly scalable tool for urban planners and policymakers to enhance roadway safety equitably and cost-effectively. |
| 2025-11-06 | [Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs](http://arxiv.org/abs/2511.04875v1) | Matthew Bozoukov, Matthew Nguyen et al. | Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated. |
| 2025-11-06 | [Investigating U.S. Consumer Demand for Food Products with Innovative Transportation Certificates Based on Stated Preferences and Machine Learning Approaches](http://arxiv.org/abs/2511.04845v1) | Jingchen Bi, Rodrigo Mesa-Arango | This paper utilizes a machine learning model to estimate the consumer's behavior for food products with innovative transportation certificates in the U.S. Building on previous research that examined demand for food products with supply chain traceability using stated preference analysis, transportation factors were identified as significant in consumer food purchasing choices. Consequently, a second experiment was conducted to pinpoint the specific transportation attributes valued by consumers. A machine learning model was applied, and five innovative certificates related to transportation were proposed: Transportation Mode, Internet of Things (IoT), Safety measures, Energy Source, and Must Arrive By Dates (MABDs). The preference experiment also incorporated product-specific and decision-maker factors for control purposes. The findings reveal a notable inclination toward safety and energy certificates within the transportation domain of the U.S. food supply chain. Additionally, the study examined the influence of price, product type, certificates, and decision-maker factors on purchasing choices. Ultimately, the study offers data-driven recommendations for improving food supply chain systems. |
| 2025-11-06 | [Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](http://arxiv.org/abs/2511.04834v1) | Jiwoo Shin, Byeonghu Na et al. | Recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts. |
| 2025-11-06 | [GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction](http://arxiv.org/abs/2511.04679v1) | Qingzhou Lu, Yao Feng et al. | Humanoid robots are expected to operate in human-centered environments where safe and natural physical interaction is essential. However, most recent reinforcement learning (RL) policies emphasize rigid tracking and suppress external forces. Existing impedance-augmented approaches are typically restricted to base or end-effector control and focus on resisting extreme forces rather than enabling compliance. We introduce GentleHumanoid, a framework that integrates impedance control into a whole-body motion tracking policy to achieve upper-body compliance. At its core is a unified spring-based formulation that models both resistive contacts (restoring forces when pressing against surfaces) and guiding contacts (pushes or pulls sampled from human motion data). This formulation ensures kinematically consistent forces across the shoulder, elbow, and wrist, while exposing the policy to diverse interaction scenarios. Safety is further supported through task-adjustable force thresholds. We evaluate our approach in both simulation and on the Unitree G1 humanoid across tasks requiring different levels of compliance, including gentle hugging, sit-to-stand assistance, and safe object manipulation. Compared to baselines, our policy consistently reduces peak contact forces while maintaining task success, resulting in smoother and more natural interactions. These results highlight a step toward humanoid robots that can safely and effectively collaborate with humans and handle objects in real-world environments. |
| 2025-11-06 | [Nonparametric Safety Stock Dimensioning: A Data-Driven Approach for Supply Chains of Hardware OEMs](http://arxiv.org/abs/2511.04616v1) | Elvis Agbenyega, Cody Quick | Resilient supply chains are critical, especially for Original Equipment Manufacturers (OEMs) that power today's digital economy. Safety Stock dimensioning-the computation of the appropriate safety stock quantity-is one of several mechanisms to ensure supply chain resiliency, as it protects the supply chain against demand and supply uncertainties. Unfortunately, the major approaches to dimensioning safety stock heavily assume that demand is normally distributed and ignore future demand variability, limiting their applicability in manufacturing contexts where demand is non-normal, intermittent, and highly skewed. In this paper, we propose a data-driven approach that relaxes the assumption of normality, enabling the demand distribution of each inventory item to be analytically determined using Kernel Density Estimation. Also, we extended the analysis from historical demand variability to forecasted demand variability. We evaluated the proposed approach against a normal distribution model in a near-world inventory replenishment simulation. Afterwards, we used a linear optimization model to determine the optimal safety stock configuration. The results from the simulation and linear optimization models showed that the data-driven approach outperformed traditional approaches. In particular, the data-driven approach achieved the desired service levels at lower safety stock levels than the conventional approaches. |
| 2025-11-06 | [From Model to Breach: Towards Actionable LLM-Generated Vulnerabilities Reporting](http://arxiv.org/abs/2511.04538v1) | Cyril Vallez, Alexander Sternfeld et al. | As the role of Large Language Models (LLM)-based coding assistants in software development becomes more critical, so does the role of the bugs they generate in the overall cybersecurity landscape. While a number of LLM code security benchmarks have been proposed alongside approaches to improve the security of generated code, it remains unclear to what extent they have impacted widely used coding LLMs. Here, we show that even the latest open-weight models are vulnerable in the earliest reported vulnerability scenarios in a realistic use setting, suggesting that the safety-functionality trade-off has until now prevented effective patching of vulnerabilities. To help address this issue, we introduce a new severity metric that reflects the risk posed by an LLM-generated vulnerability, accounting for vulnerability severity, generation chance, and the formulation of the prompt that induces vulnerable code generation - Prompt Exposure (PE). To encourage the mitigation of the most serious and prevalent vulnerabilities, we use PE to define the Model Exposure (ME) score, which indicates the severity and prevalence of vulnerabilities a model generates. |

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



