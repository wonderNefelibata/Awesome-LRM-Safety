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
| 2025-10-16 | [CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions](http://arxiv.org/abs/2510.14959v1) | Lizhi Yang, Blake Werner et al. | Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed \emph{online} via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs \emph{in training}. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter. |
| 2025-10-16 | [Further Results on Safety-Critical Stabilization of Force-Controlled Nonholonomic Mobile Robots](http://arxiv.org/abs/2510.14931v1) | Bo Wang, Tianyu Han et al. | In this paper, we address the stabilization problem for force-controlled nonholonomic mobile robots under safety-critical constraints. We propose a continuous, time-invariant control law based on the gamma m-quadratic programming (gamma m-QP) framework, which unifies control Lyapunov functions (CLFs) and control barrier functions (CBFs) to enforce both stability and safety in the closed-loop system. For the first time, we construct a global, time-invariant, strict Lyapunov function for the closed-loop nonholonomic mobile robot system with a nominal stabilization controller in polar coordinates; this strict Lyapunov function then serves as the CLF in the QP design. Next, by exploiting the inherent cascaded structure of the vehicle dynamics, we develop a CBF for the mobile robot via an integrator backstepping procedure. Our main results guarantee both asymptotic stability and safety for the closed-loop system. Both the simulation and experimental results are presented to illustrate the effectiveness and performance of our approach. |
| 2025-10-16 | [Dude, Where's My (Autonomous) Car? Defining an Accessible Description Logic for Blind and Low Vision Travelers Using Autonomous Vehicles](http://arxiv.org/abs/2510.14911v1) | Paul D. S. Fink, Justin R. Brown et al. | Purpose: Autonomous vehicles (AVs) are becoming a promising transportation solution for blind and low-vision (BLV) travelers, offering the potential for greater independent mobility. This paper explores the information needs of BLV users across multiple steps of the transportation journey, including finding and navigating to, entering, and exiting vehicles independently.   Methods: A survey with 202 BLV respondents and interviews with 12 BLV individuals revealed the perspectives of BLV end-users and informed the sequencing of natural language information required for successful travel. Whereas the survey identified key information needs across the three trip segments, the interviews helped prioritize how that information should be presented in a sequence of accessible descriptions to travelers.   Results: Taken together, the survey and interviews reveal that BLV users prioritize knowing the vehicle's make and model and how to find the correct vehicle during the navigation phase. They also emphasize the importance of confirmations about the vehicle's destination and onboard safety features upon entering the vehicle. While exiting, BLV users value information about hazards and obstacles, as well as knowing which side of the vehicle to exit. Furthermore, results highlight that BLV travelers desire using their own smartphone devices when receiving information from AVs and prefer audio-based interaction.   Conclusion: The findings from this research contribute a structured framework for delivering trip-related information to BLV users, useful for designers incorporating natural language descriptions tailored to each travel segment. This work offers important contributions for sequencing transportation-related descriptions throughout the AV journey, ultimately enhancing the mobility and independence of BLV individuals. |
| 2025-10-16 | [STITCHER: Constrained Trajectory Planning in Known Environments with Real-Time Motion Primitive Search](http://arxiv.org/abs/2510.14893v1) | Helene J. Levy, Brett T. Lopez | Autonomous high-speed navigation through large, complex environments requires real-time generation of agile trajectories that are dynamically feasible, collision-free, and satisfy state or actuator constraints. Modern trajectory planning techniques primarily use numerical optimization, as they enable the systematic computation of high-quality, expressive trajectories that satisfy various constraints. However, stringent requirements on computation time and the risk of numerical instability can limit the use of optimization-based planners in safety-critical scenarios. This work presents an optimization-free planning framework called STITCHER that stitches short trajectory segments together with graph search to compute long-range, expressive, and near-optimal trajectories in real-time. STITCHER outperforms modern optimization-based planners through our innovative planning architecture and several algorithmic developments that make real-time planning possible. Extensive simulation testing is performed to analyze the algorithmic components that make up STITCHER, along with a thorough comparison with two state-of-the-art optimization planners. Simulation tests show that safe trajectories can be created within a few milliseconds for paths that span the entirety of two 50 m x 50 m environments. Hardware tests with a custom quadrotor verify that STITCHER can produce trackable paths in real-time while respecting nonconvex constraints, such as limits on tilt angle and motor forces, which are otherwise hard to include in optimization-based planners. |
| 2025-10-16 | [Where to Search: Measure the Prior-Structured Search Space of LLM Agents](http://arxiv.org/abs/2510.14846v1) | Zhuo-Yang Song | The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs. |
| 2025-10-16 | [Backdoor Unlearning by Linear Task Decomposition](http://arxiv.org/abs/2510.14845v1) | Amel Abdelraheem, Alessandro Favero et al. | Foundation models have revolutionized computer vision by enabling broad generalization across diverse tasks. Yet, they remain highly susceptible to adversarial perturbations and targeted backdoor attacks. Mitigating such vulnerabilities remains an open challenge, especially given that the large-scale nature of the models prohibits retraining to ensure safety. Existing backdoor removal approaches rely on costly fine-tuning to override the harmful behavior, and can often degrade performance on other unrelated tasks. This raises the question of whether backdoors can be removed without compromising the general capabilities of the models. In this work, we address this question and study how backdoors are encoded in the model weight space, finding that they are disentangled from other benign tasks. Specifically, this separation enables the isolation and erasure of the backdoor's influence on the model with minimal impact on clean performance. Building on this insight, we introduce a simple unlearning method that leverages such disentanglement. Through extensive experiments with CLIP-based models and common adversarial triggers, we show that, given the knowledge of the attack, our method achieves approximately perfect unlearning, while retaining, on average, 96% of clean accuracy. Additionally, we demonstrate that even when the attack and its presence are unknown, our method successfully unlearns backdoors by proper estimation using reverse-engineered triggers. Overall, our method consistently yields better unlearning and clean accuracy tradeoffs when compared to present state-of-the-art defenses. |
| 2025-10-16 | [The Pursuit of Diversity: Multi-Objective Testing of Deep Reinforcement Learning Agents](http://arxiv.org/abs/2510.14727v1) | Antony Bartlett, Cynthia Liem et al. | Testing deep reinforcement learning (DRL) agents in safety-critical domains requires discovering diverse failure scenarios. Existing tools such as INDAGO rely on single-objective optimization focused solely on maximizing failure counts, but this does not ensure discovered scenarios are diverse or reveal distinct error types. We introduce INDAGO-Nexus, a multi-objective search approach that jointly optimizes for failure likelihood and test scenario diversity using multi-objective evolutionary algorithms with multiple diversity metrics and Pareto front selection strategies. We evaluated INDAGO-Nexus on three DRL agents: humanoid walker, self-driving car, and parking agent. On average, INDAGO-Nexus discovers up to 83% and 40% more unique failures (test effectiveness) than INDAGO in the SDC and Parking scenarios, respectively, while reducing time-to-failure by up to 67% across all agents. |
| 2025-10-16 | [JASDA: Introducing Job-Aware Scheduling in Scheduler-Driven Job Atomization](http://arxiv.org/abs/2510.14599v1) | Michal Konopa, Jan Fesl et al. | The increasing complexity and temporal variability of workloads on MIG-enabled GPUs challenge the scalability of traditional centralized scheduling. Building upon the SJA concept, this paper introduces JASDA-a novel paradigm that extends SJA from a largely centralized scheduling model toward a fully decentralized negotiation process. In JASDA, jobs actively generate and score feasible subjobs in response to scheduler-announced execution windows, while the scheduler performs policy-driven clearing that balances utilization, fairness, and temporal responsiveness. This bidirectional, iterative interaction embeds feedback, calibration, and probabilistic safety directly into the scheduling loop, enabling adaptive and transparent decision-making. By coupling principles from auction theory and online optimization with the temporal granularity of GPU workloads, JASDA provides a scalable foundation for market-aware and fairness-driven resource management-bridging theoretical scheduling models with practical deployment in modern MIG-enabled environments relevant to Artificial Intelligence and Agriculture 4.0. |
| 2025-10-16 | [Assessing Socio-Cultural Alignment and Technical Safety of Sovereign LLMs](http://arxiv.org/abs/2510.14565v1) | Kyubyung Chae, Gihoon Kim et al. | Recent trends in LLMs development clearly show growing interest in the use and application of sovereign LLMs. The global debate over sovereign LLMs highlights the need for governments to develop their LLMs, tailored to their unique socio-cultural and historical contexts. However, there remains a shortage of frameworks and datasets to verify two critical questions: (1) how well these models align with users' socio-cultural backgrounds, and (2) whether they maintain safety and technical robustness without exposing users to potential harms and risks. To address this gap, we construct a new dataset and introduce an analytic framework for extracting and evaluating the socio-cultural elements of sovereign LLMs, alongside assessments of their technical robustness. Our experimental results demonstrate that while sovereign LLMs play a meaningful role in supporting low-resource languages, they do not always meet the popular claim that these models serve their target users well. We also show that pursuing this untested claim may lead to underestimating critical quality attributes such as safety. Our study suggests that advancing sovereign LLMs requires a more extensive evaluation that incorporates a broader range of well-grounded and practical criteria. |
| 2025-10-16 | [Black Holes in Asymptotic Safety: A Review of Solutions and Phenomenology](http://arxiv.org/abs/2510.14552v1) | Andrea Spina | Asymptotic Safety offers a conservative and predictive framework for quantum gravity, based on the existence of a renormalization group fixed point that ensures ultraviolet completeness without introducing new degrees of freedom. Black holes provide a natural arena in which to explore the implications of this scenario, as they probe the strongest gravitational fields and highlight the shortcomings of classical general relativity. In recent years, a variety of quantum-corrected black-hole solutions have been constructed within the Asymptotic Safety approach, either by renormalization-group improvement of classical metrics or through effective actions inspired by the flow of couplings. This review summarizes the current status of these developments. We discuss the structure and properties of the proposed solutions, their thermodynamics and evaporation, and their dynamical aspects such as quasinormal modes and shadows. |
| 2025-10-16 | [Symbol Grounding in Neuro-Symbolic AI: A Gentle Introduction to Reasoning Shortcuts](http://arxiv.org/abs/2510.14538v1) | Emanuele Marconato, Samuele Bortolotti et al. | Neuro-symbolic (NeSy) AI aims to develop deep neural networks whose predictions comply with prior knowledge encoding, e.g. safety or structural constraints. As such, it represents one of the most promising avenues for reliable and trustworthy AI. The core idea behind NeSy AI is to combine neural and symbolic steps: neural networks are typically responsible for mapping low-level inputs into high-level symbolic concepts, while symbolic reasoning infers predictions compatible with the extracted concepts and the prior knowledge. Despite their promise, it was recently shown that - whenever the concepts are not supervised directly - NeSy models can be affected by Reasoning Shortcuts (RSs). That is, they can achieve high label accuracy by grounding the concepts incorrectly. RSs can compromise the interpretability of the model's explanations, performance in out-of-distribution scenarios, and therefore reliability. At the same time, RSs are difficult to detect and prevent unless concept supervision is available, which is typically not the case. However, the literature on RSs is scattered, making it difficult for researchers and practitioners to understand and tackle this challenging problem. This overview addresses this issue by providing a gentle introduction to RSs, discussing their causes and consequences in intuitive terms. It also reviews and elucidates existing theoretical characterizations of this phenomenon. Finally, it details methods for dealing with RSs, including mitigation and awareness strategies, and maps their benefits and limitations. By reformulating advanced material in a digestible form, this overview aims to provide a unifying perspective on RSs to lower the bar to entry for tackling them. Ultimately, we hope this overview contributes to the development of reliable NeSy and trustworthy AI models. |
| 2025-10-16 | [Real-Time Surgical Instrument Defect Detection via Non-Destructive Testing](http://arxiv.org/abs/2510.14525v1) | Qurrat Ul Ain, Atif Aftab Ahmed Jilani et al. | Defective surgical instruments pose serious risks to sterility, mechanical integrity, and patient safety, increasing the likelihood of surgical complications. However, quality control in surgical instrument manufacturing often relies on manual inspection, which is prone to human error and inconsistency. This study introduces SurgScan, an AI-powered defect detection framework for surgical instruments. Using YOLOv8, SurgScan classifies defects in real-time, ensuring high accuracy and industrial scalability. The model is trained on a high-resolution dataset of 102,876 images, covering 11 instrument types and five major defect categories. Extensive evaluation against state-of-the-art CNN architectures confirms that SurgScan achieves the highest accuracy (99.3%) with real-time inference speeds of 4.2-5.8 ms per image, making it suitable for industrial deployment. Statistical analysis demonstrates that contrast-enhanced preprocessing significantly improves defect detection, addressing key limitations in visual inspection. SurgScan provides a scalable, cost-effective AI solution for automated quality control, reducing reliance on manual inspection while ensuring compliance with ISO 13485 and FDA standards, paving the way for enhanced defect detection in medical manufacturing. |
| 2025-10-16 | [Learning to Undo: Rollback-Augmented Reinforcement Learning with Reversibility Signals](http://arxiv.org/abs/2510.14503v1) | Andrejs Sorstkins, Omer Tariq et al. | This paper proposes a reversible learning framework to improve the robustness and efficiency of value based Reinforcement Learning agents, addressing vulnerability to value overestimation and instability in partially irreversible environments. The framework has two complementary core mechanisms: an empirically derived transition reversibility measure called Phi of s and a, and a selective state rollback operation. We introduce an online per state action estimator called Phi that quantifies the likelihood of returning to a prior state within a fixed horizon K. This measure is used to adjust the penalty term during temporal difference updates dynamically, integrating reversibility awareness directly into the value function. The system also includes a selective rollback operator. When an action yields an expected return markedly lower than its instantaneous estimated value and violates a predefined threshold, the agent is penalized and returns to the preceding state rather than progressing. This interrupts sub optimal high risk trajectories and avoids catastrophic steps. By combining reversibility aware evaluation with targeted rollback, the method improves safety, performance, and stability. In the CliffWalking v0 domain, the framework reduced catastrophic falls by over 99.8 percent and yielded a 55 percent increase in mean episode return. In the Taxi v3 domain, it suppressed illegal actions by greater than or equal to 99.9 percent and achieved a 65.7 percent improvement in cumulative reward, while also sharply reducing reward variance in both environments. Ablation studies confirm that the rollback mechanism is the critical component underlying these safety and performance gains, marking a robust step toward safe and reliable sequential decision making. |
| 2025-10-16 | [Structured Universal Adversarial Attacks on Object Detection for Video Sequences](http://arxiv.org/abs/2510.14460v1) | Sven Jacob, Weijia Shao et al. | Video-based object detection plays a vital role in safety-critical applications. While deep learning-based object detectors have achieved impressive performance, they remain vulnerable to adversarial attacks, particularly those involving universal perturbations. In this work, we propose a minimally distorted universal adversarial attack tailored for video object detection, which leverages nuclear norm regularization to promote structured perturbations concentrated in the background. To optimize this formulation efficiently, we employ an adaptive, optimistic exponentiated gradient method that enhances both scalability and convergence. Our results demonstrate that the proposed attack outperforms both low-rank projected gradient descent and Frank-Wolfe based attacks in effectiveness while maintaining high stealthiness. All code and data are publicly available at https://github.com/jsve96/AO-Exp-Attack. |
| 2025-10-16 | [IMAGINE: Integrating Multi-Agent System into One Model for Complex Reasoning and Planning](http://arxiv.org/abs/2510.14406v1) | Xikai Zhang, Bo Wang et al. | Although large language models (LLMs) have made significant strides across various tasks, they still face significant challenges in complex reasoning and planning. For example, even with carefully designed prompts and prior information explicitly provided, GPT-4o achieves only a 7% Final Pass Rate on the TravelPlanner dataset in the sole-planning mode. Similarly, even in the thinking mode, Qwen3-8B-Instruct and DeepSeek-R1-671B, only achieve Final Pass Rates of 5.9% and 40%, respectively. Although well-organized Multi-Agent Systems (MAS) can offer improved collective reasoning, they often suffer from high reasoning costs due to multi-round internal interactions, long per-response latency, and difficulties in end-to-end training. To address these challenges, we propose a general and scalable framework called IMAGINE, short for Integrating Multi-Agent System into One Model. This framework not only integrates the reasoning and planning capabilities of MAS into a single, compact model, but also significantly surpass the capabilities of the MAS through a simple end-to-end training. Through this pipeline, a single small-scale model is not only able to acquire the structured reasoning and planning capabilities of a well-organized MAS but can also significantly outperform it. Experimental results demonstrate that, when using Qwen3-8B-Instruct as the base model and training it with our method, the model achieves an 82.7% Final Pass Rate on the TravelPlanner benchmark, far exceeding the 40% of DeepSeek-R1-671B, while maintaining a much smaller model size. |
| 2025-10-16 | [Evaluating & Reducing Deceptive Dialogue From Language Models with Multi-turn RL](http://arxiv.org/abs/2510.14318v1) | Marwa Abdulhai, Ryan Cheng et al. | Large Language Models (LLMs) interact with millions of people worldwide in applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to quantify deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any existing metrics we test. Additionally, our benchmarking of eight state-of-the-art models indicates that LLMs naturally exhibit deceptive behavior in approximately 26% of dialogue turns, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness by as much as 31% relative to baselines. Unexpectedly, models trained with RLHF, the predominant approach for ensuring the safety of widely-deployed LLMs, still exhibit deception at a rate of 43% on average. Given that deception in dialogue is a behavior that develops over an interaction history, its effective evaluation and mitigation necessitates moving beyond single-utterance analyses. We introduce a multi-turn reinforcement learning methodology to fine-tune LLMs to reduce deceptive behaviors, leading to a 77.6% reduction compared to other instruction-tuned models. |
| 2025-10-16 | [Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies](http://arxiv.org/abs/2510.14312v1) | Mason Nakamura, Abhinav Kumar et al. | A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems. |
| 2025-10-16 | [A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space](http://arxiv.org/abs/2510.14301v1) | Bingjie Zhang, Yibo Yang et al. | Large language models (LLMs) have achieved remarkable success in diverse tasks, yet their safety alignment remains fragile during adaptation. Even when fine-tuning on benign data or with low-rank adaptation, pre-trained safety behaviors are easily degraded, leading to harmful responses in the fine-tuned models. To address this challenge, we propose GuardSpace, a guardrail framework for preserving safety alignment throughout fine-tuning, composed of two key components: a safety-sensitive subspace and a harmful-resistant null space. First, we explicitly decompose pre-trained weights into safety-relevant and safety-irrelevant components using covariance-preconditioned singular value decomposition, and initialize low-rank adapters from the safety-irrelevant ones, while freezing safety-relevant components to preserve their associated safety mechanism. Second, we construct a null space projector that restricts adapter updates from altering safe outputs on harmful prompts, thereby maintaining the original refusal behavior. Experiments with various pre-trained models on multiple downstream tasks demonstrate that GuardSpace achieves superior performance over existing methods. Notably, for Llama-2-7B-Chat fine-tuned on GSM8K, GuardSpace outperforms the state-of-the-art method AsFT, reducing the average harmful score from 14.4% to 3.6%, while improving the accuracy from from 26.0% to 28.0%. |
| 2025-10-16 | [Qwen3Guard Technical Report](http://arxiv.org/abs/2510.14276v1) | Haiquan Zhao, Chenhan Yuan et al. | As large language models (LLMs) become more capable and widely used, ensuring the safety of their outputs is increasingly critical. Existing guardrail models, though useful in static evaluation settings, face two major limitations in real-world applications: (1) they typically output only binary "safe/unsafe" labels, which can be interpreted inconsistently across diverse safety policies, rendering them incapable of accommodating varying safety tolerances across domains; and (2) they require complete model outputs before performing safety checks, making them fundamentally incompatible with streaming LLM inference, thereby preventing timely intervention during generation and increasing exposure to harmful partial outputs. To address these challenges, we present Qwen3Guard, a series of multilingual safety guardrail models with two specialized variants: Generative Qwen3Guard, which casts safety classification as an instruction-following task to enable fine-grained tri-class judgments (safe, controversial, unsafe); and Stream Qwen3Guard, which introduces a token-level classification head for real-time safety monitoring during incremental text generation. Both variants are available in three sizes (0.6B, 4B, and 8B parameters) and support up to 119 languages and dialects, providing comprehensive, scalable, and low-latency safety moderation for global LLM deployments. Evaluated across English, Chinese, and multilingual benchmarks, Qwen3Guard achieves state-of-the-art performance in both prompt and response safety classification. All models are released under the Apache 2.0 license for public use. |
| 2025-10-16 | [MorphoBench: A Benchmark with Difficulty Adaptive to Model Reasoning](http://arxiv.org/abs/2510.14265v1) | Xukai Wang, Xuanbo Liu et al. | With the advancement of powerful large-scale reasoning models, effectively evaluating the reasoning capabilities of these models has become increasingly important. However, existing benchmarks designed to assess the reasoning abilities of large models tend to be limited in scope and lack the flexibility to adapt their difficulty according to the evolving reasoning capacities of the models. To address this, we propose MorphoBench, a benchmark that incorporates multidisciplinary questions to evaluate the reasoning capabilities of large models and can adjust and update question difficulty based on the reasoning abilities of advanced models. Specifically, we curate the benchmark by selecting and collecting complex reasoning questions from existing benchmarks and sources such as Olympiad-level competitions. Additionally, MorphoBench adaptively modifies the analytical challenge of questions by leveraging key statements generated during the model's reasoning process. Furthermore, it includes questions generated using simulation software, enabling dynamic adjustment of benchmark difficulty with minimal resource consumption. We have gathered over 1,300 test questions and iteratively adjusted the difficulty of MorphoBench based on the reasoning capabilities of models such as o3 and GPT-5. MorphoBench enhances the comprehensiveness and validity of model reasoning evaluation, providing reliable guidance for improving both the reasoning abilities and scientific robustness of large models. The code has been released in https://github.com/OpenDCAI/MorphoBench. |

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



