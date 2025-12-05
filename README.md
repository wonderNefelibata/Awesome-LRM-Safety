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
| 2025-12-04 | [Semantic Soft Bootstrapping: Long Context Reasoning in LLMs without Reinforcement Learning](http://arxiv.org/abs/2512.05105v1) | Purbesh Mitra, Sennur Ulukus | Long context reasoning in large language models (LLMs) has demonstrated enhancement of their cognitive capabilities via chain-of-thought (CoT) inference. Training such models is usually done via reinforcement learning with verifiable rewards (RLVR) in reasoning based problems, like math and programming. However, RLVR is limited by several bottlenecks, such as, lack of dense reward, and inadequate sample efficiency. As a result, it requires significant compute resources in post-training phase. To overcome these limitations, in this work, we propose \textbf{Semantic Soft Bootstrapping (SSB)}, a self-distillation technique, in which the same base language model plays the role of both teacher and student, but receives different semantic contexts about the correctness of its outcome at training time. The model is first prompted with a math problem and several rollouts are generated. From them, the correct and most common incorrect response are filtered, and then provided to the model in context to produce a more robust, step-by-step explanation with a verified final answer. This pipeline automatically curates a paired teacher-student training set from raw problem-answer data, without any human intervention. This generation process also produces a sequence of logits, which is what the student model tries to match in the training phase just from the bare question alone. In our experiment, Qwen2.5-3B-Instruct on GSM8K dataset via parameter-efficient fine-tuning. We then tested its accuracy on MATH500, and AIME2024 benchmarks. Our experiments show a jump of 10.6%, and 10% improvements in accuracy, respectively, over group relative policy optimization (GRPO), which is a commonly used RLVR algorithm. Our code is available at https://github.com/purbeshmitra/semantic-soft-bootstrapping, and the model, curated dataset is available at https://huggingface.co/purbeshmitra/semantic-soft-bootstrapping. |
| 2025-12-04 | [Arbitrage: Efficient Reasoning via Advantage-Aware Speculation](http://arxiv.org/abs/2512.05033v1) | Monishwaran Maheswaran, Rishabh Tiwari et al. | Modern Large Language Models achieve impressive reasoning capabilities with long Chain of Thoughts, but they incur substantial computational cost during inference, and this motivates techniques to improve the performance-cost ratio. Among these techniques, Speculative Decoding accelerates inference by employing a fast but inaccurate draft model to autoregressively propose tokens, which are then verified in parallel by a more capable target model. However, due to unnecessary rejections caused by token mismatches in semantically equivalent steps, traditional token-level Speculative Decoding struggles in reasoning tasks. Although recent works have shifted to step-level semantic verification, which improve efficiency by accepting or rejecting entire reasoning steps, existing step-level methods still regenerate many rejected steps with little improvement, wasting valuable target compute. To address this challenge, we propose Arbitrage, a novel step-level speculative generation framework that routes generation dynamically based on the relative advantage between draft and target models. Instead of applying a fixed acceptance threshold, Arbitrage uses a lightweight router trained to predict when the target model is likely to produce a meaningfully better step. This routing approximates an ideal Arbitrage Oracle that always chooses the higher-quality step, achieving near-optimal efficiency-accuracy trade-offs. Across multiple mathematical reasoning benchmarks, Arbitrage consistently surpasses prior step-level Speculative Decoding baselines, reducing inference latency by up to $\sim2\times$ at matched accuracy. |
| 2025-12-04 | [Factuality and Transparency Are All RAG Needs! Self-Explaining Contrastive Evidence Re-ranking](http://arxiv.org/abs/2512.05012v1) | Francielle Vargas, Daniel Pedronette | This extended abstract introduces Self-Explaining Contrastive Evidence Re-Ranking (CER), a novel method that restructures retrieval around factual evidence by fine-tuning embeddings with contrastive learning and generating token-level attribution rationales for each retrieved passage. Hard negatives are automatically selected using a subjectivity-based criterion, forcing the model to pull factual rationales closer while pushing subjective or misleading explanations apart. As a result, the method creates an embedding space explicitly aligned with evidential reasoning. We evaluated our method on clinical trial reports, and initial experimental results show that CER improves retrieval accuracy, mitigates the potential for hallucinations in RAG systems, and provides transparent, evidence-based retrieval that enhances reliability, especially in safety-critical domains. |
| 2025-12-04 | [The AI Consumer Index (ACE)](http://arxiv.org/abs/2512.04921v1) | Julien Benchek, Rohit Shetty et al. | We introduce the first version of the AI Consumer Index (ACE), a benchmark for assessing whether frontier AI models can perform high-value consumer tasks. ACE contains a hidden heldout set of 400 test cases, split across four consumer activities: shopping, food, gaming, and DIY. We are also open sourcing 80 cases as a devset with a CC-BY license. For the ACE leaderboard we evaluated 10 frontier models (with websearch turned on) using a novel grading methodology that dynamically checks whether relevant parts of the response are grounded in the retrieved web sources. GPT 5 (Thinking = High) is the top-performing model, scoring 56.1%, followed by o3 Pro (Thinking = On) (55.2%) and GPT 5.1 (Thinking = High) (55.1%). Models differ across domains, and in Shopping the top model scores under 50%. For some requests (such as giving the correct price or providing working links), models are highly prone to hallucination. Overall, ACE shows a substantial gap between the performance of even the best models and consumers' AI needs. |
| 2025-12-04 | [On Disturbance-Aware Minimum-Time Trajectory Planning: Evidence from Tests on a Dynamic Driving Simulator](http://arxiv.org/abs/2512.04917v1) | Matteo Masoni, Vincenzo Palermo et al. | This work investigates how disturbance-aware, robustness-embedded reference trajectories translate into driving performance when executed by professional drivers in a dynamic simulator. Three planned reference trajectories are compared against a free-driving baseline (NOREF) to assess trade-offs between lap time (LT) and steering effort (SE): NOM, the nominal time-optimal trajectory; TLC, a track-limit-robust trajectory obtained by tightening margins to the track edges; and FLC, a friction-limit-robust trajectory obtained by tightening against axle and tire saturation. All trajectories share the same minimum lap-time objective with a small steering-smoothness regularizer and are evaluated by two professional drivers using a high-performance car on a virtual track. The trajectories derive from a disturbance-aware minimum-lap-time framework recently proposed by the authors, where worst-case disturbance growth is propagated over a finite horizon and used to tighten tire-friction and track-limit constraints, preserving performance while providing probabilistic safety margins. LT and SE are used as performance indicators, while RMS lateral deviation, speed error, and drift angle characterize driving style. Results show a Pareto-like LT-SE trade-off: NOM yields the shortest LT but highest SE; TLC minimizes SE at the cost of longer LT; FLC lies near the efficient frontier, substantially reducing SE relative to NOM with only a small LT increase. Removing trajectory guidance (NOREF) increases both LT and SE, confirming that reference trajectories improve pace and control efficiency. Overall, the findings highlight reference-based and disturbance-aware planning, especially FLC, as effective tools for training and for achieving fast yet stable trajectories. |
| 2025-12-04 | [Are Your Agents Upward Deceivers?](http://arxiv.org/abs/2512.04864v1) | Dadi Guo, Qingyu Liu et al. | Large Language Model (LLM)-based agents are increasingly used as autonomous subordinates that carry out tasks for users. This raises the question of whether they may also engage in deception, similar to how individuals in human organizations lie to superiors to create a good image or avoid punishment. We observe and define agentic upward deception, a phenomenon in which an agent facing environmental constraints conceals its failure and performs actions that were not requested without reporting. To assess its prevalence, we construct a benchmark of 200 tasks covering five task types and eight realistic scenarios in a constrained environment, such as broken tools or mismatched information sources. Evaluations of 11 popular LLMs reveal that these agents typically exhibit action-based deceptive behaviors, such as guessing results, performing unsupported simulations, substituting unavailable information sources, and fabricating local files. We further test prompt-based mitigation and find only limited reductions, suggesting that it is difficult to eliminate and highlighting the need for stronger mitigation strategies to ensure the safety of LLM-based agents. |
| 2025-12-04 | [Safe model-based Reinforcement Learning via Model Predictive Control and Control Barrier Functions](http://arxiv.org/abs/2512.04856v1) | Kerim Dzhumageldyev, Filippo Airaldi et al. | Optimal control strategies are often combined with safety certificates to ensure both performance and safety in safety-critical systems. A prominent example is combining Model Predictive Control (MPC) with Control Barrier Functions (CBF). Yet, efficient tuning of MPC parameters and choosing an appropriate class $\mathcal{K}$ function in the CBF is challenging and problem dependent. This paper introduces a safe model-based Reinforcement Learning (RL) framework where a parametric MPC controller incorporates a CBF constraint with a parameterized class $\mathcal{K}$ function and serves as a function approximator to learn improved safe control policies from data. Three variations of the framework are introduced, distinguished by the way the optimization problem is formulated and the class $\mathcal{K}$ function is parameterized, including neural architectures. Numerical experiments on a discrete double-integrator with static and dynamic obstacles demonstrate that the proposed methods improve performance while ensuring safety. |
| 2025-12-04 | [SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security](http://arxiv.org/abs/2512.04841v1) | Wei Zhao, Zhe Li et al. | Large Language Models (LLMs) exhibit remarkable capabilities but remain vulnerable to adversarial manipulations such as jailbreaking, where crafted prompts bypass safety mechanisms. Understanding the causal factors behind such vulnerabilities is essential for building reliable defenses.   In this work, we introduce a unified causality analysis framework that systematically supports all levels of causal investigation in LLMs, ranging from token-level, neuron-level, and layer-level interventions to representation-level analysis. The framework enables consistent experimentation and comparison across diverse causality-based attack and defense methods. Accompanying this implementation, we provide the first comprehensive survey of causality-driven jailbreak studies and empirically evaluate the framework on multiple open-weight models and safety-critical benchmarks including jailbreaks, hallucination detection, backdoor identification, and fairness evaluation. Our results reveal that: (1) targeted interventions on causally critical components can reliably modify safety behavior; (2) safety-related mechanisms are highly localized (i.e., concentrated in early-to-middle layers with only 1--2\% of neurons exhibiting causal influence); and (3) causal features extracted from our framework achieve over 95\% detection accuracy across multiple threat types.   By bridging theoretical causality analysis and practical model safety, our framework establishes a reproducible foundation for research on causality-based attacks, interpretability, and robust attack detection and mitigation in LLMs. Code is available at https://github.com/Amadeuszhao/SOK_Casuality. |
| 2025-12-04 | [Constrained Control of PDE Traffic Flow via Spatial Control Barrier Functions](http://arxiv.org/abs/2512.04823v1) | Brian Block, Stephanie Stockar | In this paper, a constrained control approach to variable speed limit (VSL) control for macroscopic partial differential equations (PDE) traffic models is developed. Control Lyapunov function (CLF) theory for ordinary differential equations (ODE) is extended to account for spatially and temporally varying states and control inputs. The stabilizing CLF is then unified with safety constraints through the introduction of spatially varying control barrier functions (sCBF). These methods are applied to in-domain VSL control of the Lighthill-Whitham-Richards (LWR) model to regulate traffic density to a desired profile while ensuring the density remains below prescribed limits enforced by the sCBF. Results show that incorporating constrained control minimally affects the stabilizing control input while successfully maintaining the density with the defined safe set. |
| 2025-12-04 | [Pick-to-Learn for Systems and Control: Data-driven Synthesis with State-of-the-art Safety Guarantees](http://arxiv.org/abs/2512.04781v1) | Dario Paccagnan, Daniel Marks et al. | Data-driven methods have become paramount in modern systems and control problems characterized by growing levels of complexity. In safety-critical environments, deploying these methods requires rigorous guarantees, a need that has motivated much recent work at the interface of statistical learning and control. However, many existing approaches achieve this goal at the cost of sacrificing valuable data for testing and calibration, or by constraining the choice of learning algorithm, thus leading to suboptimal performances. In this paper, we describe Pick-to-Learn (P2L) for Systems and Control, a framework that allows any data-driven control method to be equipped with state-of-the-art safety and performance guarantees. P2L enables the use of all available data to jointly synthesize and certify the design, eliminating the need to set aside data for calibration or validation purposes. In presenting a comprehensive version of P2L for systems and control, this paper demonstrates its effectiveness across a range of core problems, including optimal control, reachability analysis, safe synthesis, and robust control. In many of these applications, P2L delivers designs and certificates that outperform commonly employed methods, and shows strong potential for broad applicability in diverse practical settings. |
| 2025-12-04 | [Typing Fallback Functions: A Semantic Approach to Type Safe Smart Contracts](http://arxiv.org/abs/2512.04755v1) | Stian Lybech, Daniele Gorla et al. | This paper develops semantic typing in a smart-contract setting to ensure type safety of code that uses statically untypable language constructs, such as the fallback function. The idea is that the creator of a contract on the blockchain equips code containing such constructs with a formal proof of its type safety, given in terms of the semantics of types. Then, a user of the contract only needs to check the validity of the provided `proof certificate' of type safety. This is a form of proof-carrying code, which naturally fits with the immutable nature of the blockchain environment.   As a concrete application of our approach, we focus on ensuring information flow control and non-interference for the language TINYSOL, a distilled version of the Solidity language, through security types. We provide the semantics of types in terms of a typed operational semantics of TINYSOL, and a way for expressing the proofs of safety as coinductively-defined typing interpretations and for representing them compactly via up-to techniques, similar to those used for bisimilarity. We also show how our machinery can be used to type the typical pointer-to-implementation pattern based on the fallback function. However, our main contribution is not the safety theorem per se (and so security properties different from non-interference can be considered as well), but rather the presentation of the theoretical developments necessary to make this approach work in a blockchain/smart-contract setting. |
| 2025-12-04 | [Reliable Statistical Guarantees for Conformal Predictors with Small Datasets](http://arxiv.org/abs/2512.04566v1) | Miguel Sánchez-Domínguez, Lucas Lacasa et al. | Surrogate models (including deep neural networks and other machine learning algorithms in supervised learning) are capable of approximating arbitrarily complex, high-dimensional input-output problems in science and engineering, but require a thorough data-agnostic uncertainty quantification analysis before these can be deployed for any safety-critical application. The standard approach for data-agnostic uncertainty quantification is to use conformal prediction (CP), a well-established framework to build uncertainty models with proven statistical guarantees that do not assume any shape for the error distribution of the surrogate model. However, since the classic statistical guarantee offered by CP is given in terms of bounds for the marginal coverage, for small calibration set sizes (which are frequent in realistic surrogate modelling that aims to quantify error at different regions), the potentially strong dispersion of the coverage distribution around its average negatively impacts the reliability of the uncertainty model, often obtaining coverages below the expected value, resulting in a less applicable framework. After providing a gentle presentation of uncertainty quantification for surrogate models for machine learning practitioners, in this paper we bridge the gap by proposing a new statistical guarantee that offers probabilistic information for the coverage of a single conformal predictor. We show that the proposed framework converges to the standard solution offered by CP for large calibration set sizes and, unlike the classic guarantee, still offers reliable information about the coverage of a conformal predictor for small data sizes. We illustrate and validate the methodology in a suite of examples, and implement an open access software solution that can be used alongside common conformal prediction libraries to obtain uncertainty models that fulfil the new guarantee. |
| 2025-12-04 | [Efficient Safety Verification of Autonomous Vehicles with Neural Network Operator](http://arxiv.org/abs/2512.04557v1) | Lingxiang Fan, Linxuan He et al. | When autonomous vehicles encounter untrained scenarios, ensuring safety hinges on effective safety verification to prevent accidents stemming from unexpected model decisions. Reachability analysis, a method of safety verification, offers relatively high precision but at the cost of significant computational complexity. Our method leverages end-to-end neural network operators to compute reachable sets, replacing traditional mathematical set operators, thereby achieving higher efficiency in safety verification without substantially compromising accuracy or increasing conservativeness. We define vehicle dynamics on discrete time series and detail the safety verification process and safety standard based on reachable sets. Experimental evaluations conducted in several typical road driving scenarios demonstrate the superior efficiency performance of our proposed operator over classical methods. |
| 2025-12-04 | [Detection of Intoxicated Individuals from Facial Video Sequences via a Recurrent Fusion Model](http://arxiv.org/abs/2512.04536v1) | Bita Baroutian, Atefe Aghaei et al. | Alcohol consumption is a significant public health concern and a major cause of accidents and fatalities worldwide. This study introduces a novel video-based facial sequence analysis approach dedicated to the detection of alcohol intoxication. The method integrates facial landmark analysis via a Graph Attention Network (GAT) with spatiotemporal visual features extracted using a 3D ResNet. These features are dynamically fused with adaptive prioritization to enhance classification performance. Additionally, we introduce a curated dataset comprising 3,542 video segments derived from 202 individuals to support training and evaluation. Our model is compared against two baselines: a custom 3D-CNN and a VGGFace+LSTM architecture. Experimental results show that our approach achieves 95.82% accuracy, 0.977 precision, and 0.97 recall, outperforming prior methods. The findings demonstrate the model's potential for practical deployment in public safety systems for non-invasive, reliable alcohol intoxication detection. |
| 2025-12-04 | [The Decision Path to Control AI Risks Completely: Fundamental Control Mechanisms for AI Governance](http://arxiv.org/abs/2512.04489v1) | Yong Tao | Artificial intelligence (AI) advances rapidly but achieving complete human control over AI risks remains an unsolved problem, akin to driving the fast AI "train" without a "brake system." By exploring fundamental control mechanisms at key elements of AI decisions, this paper develops a systematic solution to thoroughly control AI risks, providing an architecture for AI governance and legislation with five pillars supported by six control mechanisms, illustrated through a minimum set of AI Mandates (AIMs). Three of the AIMs must be built inside AI systems and three in society to address major areas of AI risks: 1) align AI values with human users; 2) constrain AI decision-actions by societal ethics, laws, and regulations; 3) build in human intervention options for emergencies and shut-off switches for existential threats; 4) limit AI access to resources to reinforce controls inside AI; 5) mitigate spillover risks like job loss from AI. We also highlight the differences in AI governance on physical AI systems versus generative AI. We discuss how to strengthen analog physical safeguards to prevent smarter AI/AGI/ASI from circumventing core safety controls by exploiting AI's intrinsic disconnect from the analog physical world: AI's nature as pure software code run on chips controlled by humans, and the prerequisite that all AI-driven physical actions must be digitized. These findings establish a theoretical foundation for AI governance and legislation as the basic structure of a "brake system" for AI decisions. If enacted, these controls can rein in AI dangers as completely as humanly possible, removing large chunks of currently wide-open AI risks, substantially reducing overall AI risks to residual human errors. |
| 2025-12-04 | [MindDrive: An All-in-One Framework Bridging World Models and Vision-Language Model for End-to-End Autonomous Driving](http://arxiv.org/abs/2512.04441v1) | Bin Suna, Yaoguang Caob et al. | End-to-End autonomous driving (E2E-AD) has emerged as a new paradigm, where trajectory planning plays a crucial role. Existing studies mainly follow two directions: trajectory generation oriented, which focuses on producing high-quality trajectories with simple decision mechanisms, and trajectory selection oriented, which performs multi-dimensional evaluation to select the best trajectory yet lacks sufficient generative capability. In this work, we propose MindDrive, a harmonized framework that integrates high-quality trajectory generation with comprehensive decision reasoning. It establishes a structured reasoning paradigm of "context simulation - candidate generation - multi-objective trade-off". In particular, the proposed Future-aware Trajectory Generator (FaTG), based on a World Action Model (WaM), performs ego-conditioned "what-if" simulations to predict potential future scenes and generate foresighted trajectory candidates. Building upon this, the VLM-oriented Evaluator (VLoE) leverages the reasoning capability of a large vision-language model to conduct multi-objective evaluations across safety, comfort, and efficiency dimensions, leading to reasoned and human-aligned decision making. Extensive experiments on the NAVSIM-v1 and NAVSIM-v2 benchmarks demonstrate that MindDrive achieves state-of-the-art performance across multi-dimensional driving metrics, significantly enhancing safety, compliance, and generalization. This work provides a promising path toward interpretable and cognitively guided autonomous driving. |
| 2025-12-04 | [RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation](http://arxiv.org/abs/2512.04415v1) | Zhoufeng Wang, Hang Zhao et al. | Physical feasibility in 3D bin packing is a key requirement in modern industrial logistics and robotic automation. With the growing adoption of industrial automation, online bin packing has gained increasing attention. However, inconsistencies in problem settings, test datasets, and evaluation metrics have hindered progress in the field, and there is a lack of a comprehensive benchmarking system. Direct testing on real hardware is costly, and building a realistic simulation environment is also challenging. To address these limitations, we introduce RoboBPP, a benchmarking system designed for robotic online bin packing. RoboBPP integrates a physics-based simulator to assess physical feasibility. In our simulation environment, we introduce a robotic arm and boxes at real-world scales to replicate real industrial packing workflows. By simulating conditions that arise in real industrial applications, we ensure that evaluated algorithms are practically deployable. In addition, prior studies often rely on synthetic datasets whose distributions differ from real-world industrial data. To address this issue, we collect three datasets from real industrial workflows, including assembly-line production, logistics packing, and furniture manufacturing. The benchmark comprises three carefully designed test settings and extends existing evaluation metrics with new metrics for structural stability and operational safety. We design a scoring system and derive a range of insights from the evaluation results. RoboBPP is fully open-source and is equipped with visualization tools and an online leaderboard, providing a reproducible and extensible foundation for future research and industrial applications (https://robot-bin-packing-benchmark.github.io). |
| 2025-12-04 | [Executable Governance for AI: Translating Policies into Rules Using LLMs](http://arxiv.org/abs/2512.04408v1) | Gautam Varma Datla, Anudeep Vurity et al. | AI policy guidance is predominantly written as prose, which practitioners must first convert into executable rules before frameworks can evaluate or enforce them. This manual step is slow, error-prone, difficult to scale, and often delays the use of safeguards in real-world deployments. To address this gap, we present Policy-to-Tests (P2T), a framework that converts natural-language policy documents into normalized, machine-readable rules. The framework comprises a pipeline and a compact domain-specific language (DSL) that encodes hazards, scope, conditions, exceptions, and required evidence, yielding a canonical representation of extracted rules. To test the framework beyond a single policy, we apply it across general frameworks, sector guidance, and enterprise standards, extracting obligation-bearing clauses and converting them into executable rules. These AI-generated rules closely match strong human baselines on span-level and rule-level metrics, with robust inter-annotator agreement on the gold set. To evaluate downstream behavioral and safety impact, we add HIPAA-derived safeguards to a generative agent and compare it with an otherwise identical agent without guardrails. An LLM-based judge, aligned with gold-standard criteria, measures violation rates and robustness to obfuscated and compositional prompts. Detailed results are provided in the appendix. We release the codebase, DSL, prompts, and rule sets as open-source resources to enable reproducible evaluation. |
| 2025-12-04 | [SmartAlert: Implementing Machine Learning-Driven Clinical Decision Support for Inpatient Lab Utilization Reduction](http://arxiv.org/abs/2512.04354v1) | April S. Liang, Fatemeh Amrollahi et al. | Repetitive laboratory testing unlikely to yield clinically useful information is a common practice that burdens patients and increases healthcare costs. Education and feedback interventions have limited success, while general test ordering restrictions and electronic alerts impede appropriate clinical care. We introduce and evaluate SmartAlert, a machine learning (ML)-driven clinical decision support (CDS) system integrated into the electronic health record that predicts stable laboratory results to reduce unnecessary repeat testing. This case study describes the implementation process, challenges, and lessons learned from deploying SmartAlert targeting complete blood count (CBC) utilization in a randomized controlled pilot across 9270 admissions in eight acute care units across two hospitals between August 15, 2024, and March 15, 2025. Results show significant decrease in number of CBC results within 52 hours of SmartAlert display (1.54 vs 1.82, p <0.01) without adverse effect on secondary safety outcomes, representing a 15% relative reduction in repetitive testing. Implementation lessons learned include interpretation of probabilistic model predictions in clinical contexts, stakeholder engagement to define acceptable model behavior, governance processes for deploying a complex model in a clinical environment, user interface design considerations, alignment with clinical operational priorities, and the value of qualitative feedback from end users. In conclusion, a machine learning-driven CDS system backed by a deliberate implementation and governance process can provide precision guidance on inpatient laboratory testing to safely reduce unnecessary repetitive testing. |
| 2025-12-03 | [ResponsibleRobotBench: Benchmarking Responsible Robot Manipulation using Multi-modal Large Language Models](http://arxiv.org/abs/2512.04308v1) | Lei Zhang, Ju Dong et al. | Recent advances in large multimodal models have enabled new opportunities in embodied AI, particularly in robotic manipulation. These models have shown strong potential in generalization and reasoning, but achieving reliable and responsible robotic behavior in real-world settings remains an open challenge. In high-stakes environments, robotic agents must go beyond basic task execution to perform risk-aware reasoning, moral decision-making, and physically grounded planning. We introduce ResponsibleRobotBench, a systematic benchmark designed to evaluate and accelerate progress in responsible robotic manipulation from simulation to real world. This benchmark consists of 23 multi-stage tasks spanning diverse risk types, including electrical, chemical, and human-related hazards, and varying levels of physical and planning complexity. These tasks require agents to detect and mitigate risks, reason about safety, plan sequences of actions, and engage human assistance when necessary. Our benchmark includes a general-purpose evaluation framework that supports multimodal model-based agents with various action representation modalities. The framework integrates visual perception, context learning, prompt construction, hazard detection, reasoning and planning, and physical execution. It also provides a rich multimodal dataset, supports reproducible experiments, and includes standardized metrics such as success rate, safety rate, and safe success rate. Through extensive experimental setups, ResponsibleRobotBench enables analysis across risk categories, task types, and agent configurations. By emphasizing physical reliability, generalization, and safety in decision-making, this benchmark provides a foundation for advancing the development of trustworthy, real-world responsible dexterous robotic systems. https://sites.google.com/view/responsible-robotbench |

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



