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
| 2025-07-21 | [The Impact of Language Mixing on Bilingual LLM Reasoning](http://arxiv.org/abs/2507.15849v1) | Yihao Li, Jiayi Xin et al. | Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior. |
| 2025-07-21 | [Towards physician-centered oversight of conversational diagnostic AI](http://arxiv.org/abs/2507.15743v1) | Elahe Vedadi, David Barrett et al. | Recent work has demonstrated the promise of conversational AI systems for diagnostic dialogue. However, real-world assurance of patient safety means that providing individual diagnoses and treatment plans is considered a regulated activity by licensed professionals. Furthermore, physicians commonly oversee other team members in such activities, including nurse practitioners (NPs) or physician assistants/associates (PAs). Inspired by this, we propose a framework for effective, asynchronous oversight of the Articulate Medical Intelligence Explorer (AMIE) AI system. We propose guardrailed-AMIE (g-AMIE), a multi-agent system that performs history taking within guardrails, abstaining from individualized medical advice. Afterwards, g-AMIE conveys assessments to an overseeing primary care physician (PCP) in a clinician cockpit interface. The PCP provides oversight and retains accountability of the clinical decision. This effectively decouples oversight from intake and can thus happen asynchronously. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) of text consultations with asynchronous oversight, we compared g-AMIE to NPs/PAs or a group of PCPs under the same guardrails. Across 60 scenarios, g-AMIE outperformed both groups in performing high-quality intake, summarizing cases, and proposing diagnoses and management plans for the overseeing PCP to review. This resulted in higher quality composite decisions. PCP oversight of g-AMIE was also more time-efficient than standalone PCP consultations in prior work. While our study does not replicate existing clinical practices and likely underestimates clinicians' capabilities, our results demonstrate the promise of asynchronous oversight as a feasible paradigm for diagnostic AI systems to operate under expert human oversight for enhancing real-world care. |
| 2025-07-21 | [BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning](http://arxiv.org/abs/2507.15717v1) | Sahana Srinivasan, Xuguang Ai et al. | Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models. |
| 2025-07-21 | [Surfacing Variations to Calibrate Perceived Reliability of MLLM-generated Image Descriptions](http://arxiv.org/abs/2507.15692v1) | Meng Chen, Akhil Iyer et al. | Multimodal large language models (MLLMs) provide new opportunities for blind and low vision (BLV) people to access visual information in their daily lives. However, these models often produce errors that are difficult to detect without sight, posing safety and social risks in scenarios from medication identification to outfit selection. While BLV MLLM users use creative workarounds such as cross-checking between tools and consulting sighted individuals, these approaches are often time-consuming and impractical. We explore how systematically surfacing variations across multiple MLLM responses can support BLV users to detect unreliable information without visually inspecting the image. We contribute a design space for eliciting and presenting variations in MLLM descriptions, a prototype system implementing three variation presentation styles, and findings from a user study with 15 BLV participants. Our results demonstrate that presenting variations significantly increases users' ability to identify unreliable claims (by 4.9x using our approach compared to single descriptions) and significantly decreases perceived reliability of MLLM responses. 14 of 15 participants preferred seeing variations of MLLM responses over a single description, and all expressed interest in using our system for tasks from understanding a tornado's path to posting an image on social media. |
| 2025-07-21 | [EMP: Executable Motion Prior for Humanoid Robot Standing Upper-body Motion Imitation](http://arxiv.org/abs/2507.15649v1) | Haocheng Xu, Haodong Zhang et al. | To support humanoid robots in performing manipulation tasks, it is essential to study stable standing while accommodating upper-body motions. However, the limited controllable range of humanoid robots in a standing position affects the stability of the entire body. Thus we introduce a reinforcement learning based framework for humanoid robots to imitate human upper-body motions while maintaining overall stability. Our approach begins with designing a retargeting network that generates a large-scale upper-body motion dataset for training the reinforcement learning (RL) policy, which enables the humanoid robot to track upper-body motion targets, employing domain randomization for enhanced robustness. To avoid exceeding the robot's execution capability and ensure safety and stability, we propose an Executable Motion Prior (EMP) module, which adjusts the input target movements based on the robot's current state. This adjustment improves standing stability while minimizing changes to motion amplitude. We evaluate our framework through simulation and real-world tests, demonstrating its practical applicability. |
| 2025-07-21 | [Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems](http://arxiv.org/abs/2507.15613v1) | Andrii Balashov, Olena Ponomarova et al. | Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses. |
| 2025-07-21 | [Improving Functional Reliability of Near-Field Monitoring for Emergency Braking in Autonomous Vehicles](http://arxiv.org/abs/2507.15594v1) | Junnan Pan, Prodromos Sotiriadis et al. | Autonomous vehicles require reliable hazard detection. However, primary sensor systems may miss near-field obstacles, resulting in safety risks. Although a dedicated fast-reacting near-field monitoring system can mitigate this, it typically suffers from false positives. To mitigate these, in this paper, we introduce three monitoring strategies based on dynamic spatial properties, relevant object sizes, and motion-aware prediction. In experiments in a validated simulation, we compare the initial monitoring strategy against the proposed improvements. The results demonstrate that the proposed strategies can significantly improve the reliability of near-field monitoring systems. |
| 2025-07-21 | [Red-Team Multi-Agent Reinforcement Learning for Emergency Braking Scenario](http://arxiv.org/abs/2507.15587v1) | Yinsong Chen, Kaifeng Wang et al. | Current research on decision-making in safety-critical scenarios often relies on inefficient data-driven scenario generation or specific modeling approaches, which fail to capture corner cases in real-world contexts. To address this issue, we propose a Red-Team Multi-Agent Reinforcement Learning framework, where background vehicles with interference capabilities are treated as red-team agents. Through active interference and exploration, red-team vehicles can uncover corner cases outside the data distribution. The framework uses a Constraint Graph Representation Markov Decision Process, ensuring that red-team vehicles comply with safety rules while continuously disrupting the autonomous vehicles (AVs). A policy threat zone model is constructed to quantify the threat posed by red-team vehicles to AVs, inducing more extreme actions to increase the danger level of the scenario. Experimental results show that the proposed framework significantly impacts AVs decision-making safety and generates various corner cases. This method also offers a novel direction for research in safety-critical scenarios. |
| 2025-07-21 | [Towards Holistic Surgical Scene Graph](http://arxiv.org/abs/2507.15541v1) | Jongmin Shin, Enki Cho et al. | Surgical scene understanding is crucial for computer-assisted intervention systems, requiring visual comprehension of surgical scenes that involves diverse elements such as surgical tools, anatomical structures, and their interactions. To effectively represent the complex information in surgical scenes, graph-based approaches have been explored to structurally model surgical entities and their relationships. Previous surgical scene graph studies have demonstrated the feasibility of representing surgical scenes using graphs. However, certain aspects of surgical scenes-such as diverse combinations of tool-action-target and the identity of the hand operating the tool-remain underexplored in graph-based representations, despite their importance. To incorporate these aspects into graph representations, we propose Endoscapes-SG201 dataset, which includes annotations for tool-action-target combinations and hand identity. We also introduce SSG-Com, a graph-based method designed to learn and represent these critical elements. Through experiments on downstream tasks such as critical view of safety assessment and action triplet recognition, we demonstrated the importance of integrating these essential scene graph components, highlighting their significant contribution to surgical scene understanding. The code and dataset are available at https://github.com/ailab-kyunghee/SSG-Com |
| 2025-07-21 | [Strategies to Manage Human Factors in Mixed Reality Pilot Training: A Survey](http://arxiv.org/abs/2507.15526v1) | Antonio Perez, Avinash Singh et al. | Mixed Reality (MR) head mounted displays (HMDs) offer a promising alternative to traditional Flight Simulator Training Device (FSTD) displays, providing immersion, realism and cost efficiency. However, these technologies require management of human factors; cybersickness, visual fatigue and ergonomic strain. If left unmitigated, these effects can hinder pilot performance and training outcomes. For safety critical fields like aviation, addressing human factors challenges is crucial for MR's training potential. This survey systematically reviews the current literature identifying key human factors challenges in MR HMD use in pilot training and examines strategies to mitigate these barriers. Drawing on existing industry standards set by a leading aviation authority, the review adopts a regulatory perspective to explore hardware, software, ergonomic, physiological and psychological interventions improving pilot comfort, safety and training effectiveness in an MR FSTD. Additionally, it evaluates which of these interventions are most appropriate and viable for MR pilot training under existing aviation training regulations, ensuring that technical requirements and pilot wellbeing remain balanced. The findings yield significant insights for the human dimensions of aviation simulation training, highlighting how regulatory considerations shape the practicality of mitigation measures. These insights inform emerging MR aviation training guidelines and best practices, supporting MR's readiness to enhance aviation training. |
| 2025-07-21 | [Chart-R1: Chain-of-Thought Supervision and Reinforcement for Advanced Chart Reasoner](http://arxiv.org/abs/2507.15509v1) | Lei Chen, Xuanle Zhao et al. | Recently, inspired by OpenAI-o1/o3 and Deepseek-R1, the R1-Style method based on reinforcement learning fine-tuning has received widespread attention from the community. Previous R1-Style methods mainly focus on mathematical reasoning and code intelligence. It is of great research significance to verify their advantages on more general multimodal data. Chart is an important multimodal data type with rich information, which brings important research challenges in complex reasoning. In this work, we introduce Chart-R1, a chart-domain vision-language model with reinforcement learning fine-tuning to enable complex chart reasoning. To support Chart-R1, we first propose a novel programmatic data synthesis technology to generate high-quality step-by-step chart reasoning data covering single- and multi-subcharts, which makes up for the lack of reasoning data in the chart domain. Then we develop a two-stage training strategy: Chart-COT with step-by-step chain-of-thought supervision, and Chart-RFT with numerically sensitive reinforcement fine-tuning. Chart-COT aims to decompose complex chart reasoning tasks into fine-grained, understandable subtasks through step-by-step supervision, which lays a good foundation for improving the reasoning level of reinforcement learning. Chart-RFT utilize the typical group relative policy optimization strategy, in which a relatively soft reward is adopted for numerical response to emphasize the numerical sensitivity in the chart domain. We conduct extensive experiments on open-source benchmarks and self-built chart reasoning dataset (\emph{i.e., ChartRQA}). Experimental results show that Chart-R1 has significant advantages compared to chart-domain methods, even comparable to open/closed source large-scale models (\emph{e.g., GPT-4o, Claude-3.5}). |
| 2025-07-21 | [Constrained Control Allocation With Continuous-Time Rate Constraints: Three-Dimensional Case](http://arxiv.org/abs/2507.15489v1) | Süleyman Özkurt, Adrian Grimm et al. | This paper presents a novel quadratic programming (QP) approach for constrained control allocation that directly incorporates continuous-time actuator rate constraints without requiring slack variables. Over-actuated aircraft configurations, particularly prevalent in eVTOL and military applications, require control allocation algorithms to distribute commanded control moments among available actuators while respecting position and rate constraints. Existing methods such as direct allocation, pseudo-inverse, cascaded generalized inverse, and exact redistributed pseudo-inverse either cannot handle rate constraints in continuous time or require discretization approaches that compromise performance. Current QP methods that incorporate rate constraints rely on slack variables to ensure feasibility, which prevents full utilization of the attainable moment set and degrades allocation performance. The proposed methodology addresses this limitation by calculating the attainable moment set from both position and rate constraints through convex hull operations, then ensuring feasibility by scaling unattainable commanded moments to the boundary of the attainable moment set while preserving their direction. This approach guarantees the feasibility of the optimization problem without slack variables. The method is validated through simulation on an F-18 fighter aircraft control allocation problem, demonstrating equivalent performance to the established exact redistributed pseudo-inverse method while providing smoother actuator behavior and enhanced constraint satisfaction. Results show that incorporating continuous-time rate constraints leads to improved actuator tracking, reduced overshoot, and more precise adherence to position limits, which is essential for aircraft safety, ride comfort, and actuator longevity. |
| 2025-07-21 | [The Constitutional Controller: Doubt-Calibrated Steering of Compliant Agents](http://arxiv.org/abs/2507.15478v1) | Simon Kohaut, Felix Divo et al. | Ensuring reliable and rule-compliant behavior of autonomous agents in uncertain environments remains a fundamental challenge in modern robotics. Our work shows how neuro-symbolic systems, which integrate probabilistic, symbolic white-box reasoning models with deep learning methods, offer a powerful solution to this challenge. This enables the simultaneous consideration of explicit rules and neural models trained on noisy data, combining the strength of structured reasoning with flexible representations. To this end, we introduce the Constitutional Controller (CoCo), a novel framework designed to enhance the safety and reliability of agents by reasoning over deep probabilistic logic programs representing constraints such as those found in shared traffic spaces. Furthermore, we propose the concept of self-doubt, implemented as a probability density conditioned on doubt features such as travel velocity, employed sensors, or health factors. In a real-world aerial mobility study, we demonstrate CoCo's advantages for intelligent autonomous systems to learn appropriate doubts and navigate complex and uncertain environments safely and compliantly. |
| 2025-07-21 | [FedMultiEmo: Real-Time Emotion Recognition via Multimodal Federated Learning](http://arxiv.org/abs/2507.15470v1) | Baran Can Gül, Suraksha Nadig et al. | In-vehicle emotion recognition underpins adaptive driver-assistance systems and, ultimately, occupant safety. However, practical deployment is hindered by (i) modality fragility - poor lighting and occlusions degrade vision-based methods; (ii) physiological variability - heart-rate and skin-conductance patterns differ across individuals; and (iii) privacy risk - centralized training requires transmission of sensitive data. To address these challenges, we present FedMultiEmo, a privacy-preserving framework that fuses two complementary modalities at the decision level: visual features extracted by a Convolutional Neural Network from facial images, and physiological cues (heart rate, electrodermal activity, and skin temperature) classified by a Random Forest. FedMultiEmo builds on three key elements: (1) a multimodal federated learning pipeline with majority-vote fusion, (2) an end-to-end edge-to-cloud prototype on Raspberry Pi clients and a Flower server, and (3) a personalized Federated Averaging scheme that weights client updates by local data volume. Evaluated on FER2013 and a custom physiological dataset, the federated Convolutional Neural Network attains 77% accuracy, the Random Forest 74%, and their fusion 87%, matching a centralized baseline while keeping all raw data local. The developed system converges in 18 rounds, with an average round time of 120 seconds and a per-client memory footprint below 200 MB. These results indicate that FedMultiEmo offers a practical approach to real-time, privacy-aware emotion recognition in automotive settings. |
| 2025-07-21 | [AlgoSimBench: Identifying Algorithmically Similar Problems for Competitive Programming](http://arxiv.org/abs/2507.15378v1) | Jierui Li, Raymond Mooney | Recent progress in LLMs, such as reasoning models, has demonstrated strong abilities to solve complex competitive programming problems, often rivaling top human competitors. However, it remains underexplored whether these abilities generalize to relevant domains that are less seen during training. To address this, we introduce AlgoSimBench, a new benchmark designed to assess LLMs' ability to identify algorithmically similar problems (ASPs)-problems that can be solved using similar algorithmic approaches. AlgoSimBench consists of 1317 problems, annotated with 231 distinct fine-grained algorithm tags, from which we curate 402 multiple-choice questions (MCQs), where each question presents one algorithmically similar problem alongside three textually similar but algorithmically dissimilar distractors. Our evaluation reveals that LLMs struggle to identify ASPs, with the best-performing model (o3-mini) achieving only 65.9% accuracy on the MCQ task. To address this challenge, we propose attempted solution matching (ASM), a novel method for improving problem similarity detection. On our MCQ task, ASM yields an absolute accuracy improvement of 6.7% to 11.7% across different models. We also evaluated code embedding models and retrieval methods on similar problem identification. While the adversarial selection of problems degrades the performance to be less than random, we found that simply summarizing the problem to remove narrative elements eliminates the effect, and combining ASM with a keyword-prioritized method, BM25, can yield up to 52.2% accuracy. Code and data are available at github.com |
| 2025-07-21 | [LionGuard 2: Building Lightweight, Data-Efficient & Localised Multilingual Content Moderators](http://arxiv.org/abs/2507.15339v1) | Leanne Tan, Gabriel Chua et al. | Modern moderation systems increasingly support multiple languages, but often fail to address localisation and low-resource variants - creating safety gaps in real-world deployments. Small models offer a potential alternative to large LLMs, yet still demand considerable data and compute. We present LionGuard 2, a lightweight, multilingual moderation classifier tailored to the Singapore context, supporting English, Chinese, Malay, and partial Tamil. Built on pre-trained OpenAI embeddings and a multi-head ordinal classifier, LionGuard 2 outperforms several commercial and open-source systems across 17 benchmarks, including both Singapore-specific and public English datasets. The system is actively deployed within the Singapore Government, demonstrating practical efficacy at scale. Our findings show that high-quality local data and robust multilingual embeddings can achieve strong moderation performance, without fine-tuning large models. We release our model weights and part of our training data to support future work on LLM safety. |
| 2025-07-20 | [STL-GO: Spatio-Temporal Logic with Graph Operators for Distributed Systems with Multiple Network Topologies](http://arxiv.org/abs/2507.15147v1) | Yiqi Zhao, Xinyi Yu et al. | Multi-agent systems (MASs) consisting of a number of autonomous agents that communicate, coordinate, and jointly sense the environment to achieve complex missions can be found in a variety of applications such as robotics, smart cities, and internet-of-things applications. Modeling and monitoring MAS requirements to guarantee overall mission objectives, safety, and reliability is an important problem. Such requirements implicitly require reasoning about diverse sensing and communication modalities between agents, analysis of the dependencies between agent tasks, and the spatial or virtual distance between agents. To capture such rich MAS requirements, we model agent interactions via multiple directed graphs, and introduce a new logic -- Spatio-Temporal Logic with Graph Operators (STL-GO). The key innovation in STL-GO are graph operators that enable us to reason about the number of agents along either the incoming or outgoing edges of the underlying interaction graph that satisfy a given property of interest; for example, the requirement that an agent should sense at least two neighboring agents whose task graphs indicate the ability to collaborate. We then propose novel distributed monitoring conditions for individual agents that use only local information to determine whether or not an STL-GO specification is satisfied. We compare the expressivity of STL-GO against existing spatio-temporal logic formalisms, and demonstrate the utility of STL-GO and our distributed monitors in a bike-sharing and a multi-drone case study. |
| 2025-07-20 | [ROBAD: Robust Adversary-aware Local-Global Attended Bad Actor Detection Sequential Model](http://arxiv.org/abs/2507.15067v1) | Bing He, Mustaque Ahamad et al. | Detecting bad actors is critical to ensure the safety and integrity of internet platforms. Several deep learning-based models have been developed to identify such users. These models should not only accurately detect bad actors, but also be robust against adversarial attacks that aim to evade detection. However, past deep learning-based detection models do not meet the robustness requirement because they are sensitive to even minor changes in the input sequence. To address this issue, we focus on (1) improving the model understanding capability and (2) enhancing the model knowledge such that the model can recognize potential input modifications when making predictions. To achieve these goals, we create a novel transformer-based classification model, called ROBAD (RObust adversary-aware local-global attended Bad Actor Detection model), which uses the sequence of user posts to generate user embedding to detect bad actors. Particularly, ROBAD first leverages the transformer encoder block to encode each post bidirectionally, thus building a post embedding to capture the local information at the post level. Next, it adopts the transformer decoder block to model the sequential pattern in the post embeddings by using the attention mechanism, which generates the sequence embedding to obtain the global information at the sequence level. Finally, to enrich the knowledge of the model, embeddings of modified sequences by mimicked attackers are fed into a contrastive-learning-enhanced classification layer for sequence prediction. In essence, by capturing the local and global information (i.e., the post and sequence information) and leveraging the mimicked behaviors of bad actors in training, ROBAD can be robust to adversarial attacks. Extensive experiments on Yelp and Wikipedia datasets show that ROBAD can effectively detect bad actors when under state-of-the-art adversarial attacks. |
| 2025-07-20 | [On an Abstraction of Lyapunov and Lagrange Stability](http://arxiv.org/abs/2507.15047v1) | Michelangelo Bin, David Angeli | This paper studies a set-theoretic generalization of Lyapunov and Lagrange stability for abstract systems described by set-valued maps. Lyapunov stability is characterized as the property of inversely mapping filters to filters, Lagrange stability as that of mapping ideals to ideals. These abstract definitions unveil a deep duality between the two stability notions, enable a definition of global stability for abstract systems, and yield an agile generalization of the stability theorems for basic series, parallel, and feedback interconnections, including a small-gain theorem. Moreover, it is shown that Lagrange stability is abstractly identical to other properties of interest in control theory, such as safety and positivity, whose preservation under interconnections can be thus studied owing to the developed stability results. |
| 2025-07-20 | [Safety Controller Synthesis for Stochastic Networked Systems under Communication Constraints](http://arxiv.org/abs/2507.15031v1) | Omid Akbarzadeh, Mohammad H. Mamduhi et al. | This paper develops a framework for synthesizing safety controllers for discrete-time stochastic linear control systems (dt-SLS) operating under communication imperfections. The control unit is remote and communicates with the sensor and actuator through an imperfect wireless network. We consider a constant delay in the sensor-to-controller channel (uplink), and data loss in both sensor-to-controller and controller-to-actuator (downlink) channels. In our proposed scheme, data loss in each channel is modeled as an independent Bernoulli-distributed random process. To systematically handle the uplink delay, we first introduce an augmented discrete-time stochastic linear system (dt-ASLS) by concatenating all states and control inputs that sufficiently represent the state-input evolution of the original dt-SLS under the delay and packet loss constraints. We then leverage control barrier certificates (CBCs) for dt-ASLS to synthesize a controller that guarantees dt-SLS safety in a stochastic sense, ensuring that all trajectories of dt-SLS remain within safe regions with a quantified probabilistic bound. Our approach translates safety constraints into matrix inequalities, leading to an optimization problem that eventually quantifies the probability of satisfying the safety specification in the presence of communication imperfections. We validate our results on an RLC circuit subject to both constant delay and probabilistic data loss. |

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



