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
| 2025-12-08 | [Auditing Games for Sandbagging](http://arxiv.org/abs/2512.07810v1) | Jordan Taylor, Sid Black et al. | Future AI systems could conceal their capabilities ('sandbagging') during evaluations, potentially misleading developers and auditors. We stress-tested sandbagging detection techniques using an auditing game. First, a red team fine-tuned five models, some of which conditionally underperformed, as a proxy for sandbagging. Second, a blue team used black-box, model-internals, or training-based approaches to identify sandbagging models. We found that the blue team could not reliably discriminate sandbaggers from benign models. Black-box approaches were defeated by effective imitation of a weaker model. Linear probes, a model-internals approach, showed more promise but their naive application was vulnerable to behaviours instilled by the red team. We also explored capability elicitation as a strategy for detecting sandbagging. Although Prompt-based elicitation was not reliable, training-based elicitation consistently elicited full performance from the sandbagging models, using only a single correct demonstration of the evaluation task. However the performance of benign models was sometimes also raised, so relying on elicitation as a detection strategy was prone to false-positives. In the short-term, we recommend developers remove potential sandbagging using on-distribution training for elicitation. In the longer-term, further research is needed to ensure the efficacy of training-based elicitation, and develop robust methods for sandbagging detection. We open source our model organisms at https://github.com/AI-Safety-Institute/sandbagging_auditing_games and select transcripts and results at https://huggingface.co/datasets/sandbagging-games/evaluation_logs . A demo illustrating the game can be played at https://sandbagging-demo.far.ai/ . |
| 2025-12-08 | [RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking of Large Language Models](http://arxiv.org/abs/2512.07761v1) | Xiqiao Xiong, Ouxiang Li et al. | Large language models are vulnerable to jailbreak attacks, threatening their safe deployment in real-world applications. This paper studies black-box multi-turn jailbreaks, aiming to train attacker LLMs to elicit harmful content from black-box models through a sequence of prompt-output interactions. Existing approaches typically rely on single turn optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate the problem as a multi-turn reinforcement learning task, directly optimizing the harmfulness of the final-turn output as the outcome reward. To mitigate sparse supervision and promote long-term attack strategies, we propose two heuristic process rewards: (1) controlling the harmfulness of intermediate outputs to prevent triggering the black-box model's rejection mechanisms, and (2) maintaining the semantic relevance of intermediate outputs to avoid drifting into irrelevant content. Experimental results on multiple benchmarks show consistently improved attack success rates across multiple models, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/RL-MTJail. Warning: This paper contains examples of harmful content. |
| 2025-12-08 | [Automated Generation of Custom MedDRA Queries Using SafeTerm Medical Map](http://arxiv.org/abs/2512.07694v1) | Francois Vandenhende, Anna Georgiou et al. | In pre-market drug safety review, grouping related adverse event terms into standardised MedDRA queries or the FDA Office of New Drugs Custom Medical Queries (OCMQs) is critical for signal detection. We present a novel quantitative artificial intelligence system that understands and processes medical terminology and automatically retrieves relevant MedDRA Preferred Terms (PTs) for a given input query, ranking them by a relevance score using multi-criteria statistical methods. The system (SafeTerm) embeds medical query terms and MedDRA PTs in a multidimensional vector space, then applies cosine similarity and extreme-value clustering to generate a ranked list of PTs. Validation was conducted against the FDA OCMQ v3.0 (104 queries), restricted to valid MedDRA PTs. Precision, recall and F1 were computed across similarity-thresholds. High recall (>95%) is achieved at moderate thresholds. Higher thresholds improve precision (up to 86%). The optimal threshold (~0.70 - 0.75) yielded recall ~50% and precision ~33%. Narrow-term PT subsets performed similarly but required slightly higher similarity thresholds. The SafeTerm AI-driven system provides a viable supplementary method for automated MedDRA query generation. A similarity threshold of ~0.60 is recommended initially, with increased thresholds for refined term selection. |
| 2025-12-08 | [Depth-Wise Activation Steering for Honest Language Models](http://arxiv.org/abs/2512.07667v1) | Gracjan Góral, Marysia Winkels et al. | Large language models sometimes assert falsehoods despite internally representing the correct answer, failures of honesty rather than accuracy, which undermines auditability and safety. Existing approaches largely optimize factual correctness or depend on retraining and brittle single-layer edits, offering limited leverage over truthful reporting. We present a training-free activation steering method that weights steering strength across network depth using a Gaussian schedule. On the MASK benchmark, which separates honesty from knowledge, we evaluate seven models spanning the LLaMA, Qwen, and Mistral families and find that Gaussian scheduling improves honesty over no-steering and single-layer baselines in six of seven models. Equal-budget ablations on LLaMA-3.1-8B-Instruct and Qwen-2.5-7B-Instruct show the Gaussian schedule outperforms random, uniform, and box-filter depth allocations, indicating that how intervention is distributed across depth materially affects outcomes beyond total strength. The method is simple, model-agnostic, requires no finetuning, and provides a low-cost control knob for eliciting truthful reporting from models' existing capabilities. |
| 2025-12-08 | [Optimization-Guided Diffusion for Interactive Scene Generation](http://arxiv.org/abs/2512.07661v1) | Shiaho Li, Naisheng Ye et al. | Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism. |
| 2025-12-08 | [Obstacle Avoidance of UAV in Dynamic Environments Using Direction and Velocity-Adaptive Artificial Potential Field](http://arxiv.org/abs/2512.07609v1) | Nikita Vaibhav Pavle, Shrreya Rajneesh et al. | The conventional Artificial Potential Field (APF) is fundamentally limited by the local minima issue and its inability to account for the kinematics of moving obstacles. This paper addresses the critical challenge of autonomous collision avoidance for Unmanned Aerial Vehicles (UAVs) operating in dynamic and cluttered airspace by proposing a novel Direction and Relative Velocity Weighted Artificial Potential Field (APF). In this approach, a bounded weighting function, $ω(θ,v_{e})$, is introduced to dynamically scale the repulsive potential based on the direction and velocity of the obstacle relative to the UAV. This robust APF formulation is integrated within a Model Predictive Control (MPC) framework to generate collision-free trajectories while adhering to kinematic constraints. Simulation results demonstrate that the proposed method effectively resolves local minima and significantly enhances safety by enabling smooth, predictive avoidance maneuvers. The system ensures superior path integrity and reliable performance, confirming its viability for autonomous navigation in complex environments. |
| 2025-12-08 | [Understanding Individual Decision-Making in Multi-Agent Reinforcement Learning: A Dynamical Systems Approach](http://arxiv.org/abs/2512.07588v1) | James Rudd-Jones, María Pérez-Ortiz et al. | Analysing learning behaviour in Multi-Agent Reinforcement Learning (MARL) environments is challenging, in particular with respect to \textit{individual} decision-making. Practitioners frequently tend to study or compare MARL algorithms from a qualitative perspective largely due to the inherent stochasticity in practical algorithms arising from random dithering exploration strategies, environment transition noise, and stochastic gradient updates to name a few. Traditional analytical approaches, such as replicator dynamics, often rely on mean-field approximations to remove stochastic effects, but this simplification, whilst able to provide general overall trends, might lead to dissonance between analytical predictions and actual realisations of individual trajectories. In this paper, we propose a novel perspective on MARL systems by modelling them as \textit{coupled stochastic dynamical systems}, capturing both agent interactions and environmental characteristics. Leveraging tools from dynamical systems theory, we analyse the stability and sensitivity of agent behaviour at individual level, which are key dimensions for their practical deployments, for example, in presence of strict safety requirements. This framework allows us, for the first time, to rigorously study MARL dynamics taking into consideration their inherent stochasticity, providing a deeper understanding of system behaviour and practical insights for the design and control of multi-agent learning processes. |
| 2025-12-08 | [Performance of the SafeTerm AI-Based MedDRA Query System Against Standardised MedDRA Queries](http://arxiv.org/abs/2512.07552v1) | Francois Vandenhende, Anna Georgiou et al. | In pre-market drug safety review, grouping related adverse event terms into SMQs or OCMQs is critical for signal detection. We assess the performance of SafeTerm Automated Medical Query (AMQ) on MedDRA SMQs. The AMQ is a novel quantitative artificial intelligence system that understands and processes medical terminology and automatically retrieves relevant MedDRA Preferred Terms (PTs) for a given input query, ranking them by a relevance score (0-1) using multi-criteria statistical methods. The system (SafeTerm) embeds medical query terms and MedDRA PTs in a multidimensional vector space, then applies cosine similarity, and extreme-value clustering to generate a ranked list of PTs. Validation was conducted against tier-1 SMQs (110 queries, v28.1). Precision, recall and F1 were computed at multiple similarity-thresholds, defined either manually or using an automated method. High recall (94%)) is achieved at moderate similarity thresholds, indicative of good retrieval sensitivity. Higher thresholds filter out more terms, resulting in improved precision (up to 89%). The optimal threshold (0.70)) yielded an overall recall of (48%) and precision of (45%) across all 110 queries. Restricting to narrow-term PTs achieved slightly better performance at an increased (+0.05) similarity threshold, confirming increased relatedness of narrow versus broad terms. The automatic threshold (0.66) selection prioritizes recall (0.58) to precision (0.29). SafeTerm AMQ achieves comparable, satisfactory performance on SMQs and sanitized OCMQs. It is therefore a viable supplementary method for automated MedDRA query generation, balancing recall and precision. We recommend using suitable MedDRA PT terminology in query formulation and applying the automated threshold method to optimise recall. Increasing similarity scores allows refined, narrow terms selection. |
| 2025-12-08 | [Data-Driven Robust Safety Verification for Markov Decision Processes](http://arxiv.org/abs/2512.07550v1) | Abhijit Mazumdar, Manuela L. Bujorianu et al. | In this paper, we propose a data-driven robust safety verification framework for stochastic dynamical systems modeled as Markov decision processes with time-varying and uncertain transition probabilities. Rather than assuming access to the exact nominal transition kernel, we consider the realistic setting where only samples from multiple system executions are available. These samples may correspond to different transition models inside an ambiguity set around the nominal transition kernel. Using these observations, we construct a unified ambiguity set that captures both inherent run-to-run variability in the transition dynamics and finite-sample statistical uncertainty. This ambiguity set is formalized through a Wasserstein-distance ball around a nominal empirical distribution and naturally induces an interval Markov decision process representation of the underlying system. Within this representation, we introduce a robust safety function that characterizes reach-avoid type probabilistic safety under all transition kernels consistent with the interval Markov decision process. We further derive high-confidence safety guarantees for the true, unknown time-varying system. A numerical example illustrates the applicability and effectiveness of the proposed approach. |
| 2025-12-08 | [The Suicide Region: Option Games and the Race to Artificial General Intelligence](http://arxiv.org/abs/2512.07526v1) | David Tan | Standard real options theory predicts delay in exercising the option to invest or deploy when extreme asset volatility or technological uncertainty are present. However, in the current race to develop artificial general intelligence (AGI), sovereign actors are exhibiting behaviors contrary to theoretical predictions: the US and China are accelerating AI investment despite acknowledging the potential for catastrophic failure from AGI misalignment. We resolve this puzzle by formalizing the AGI race as a continuous-time preemption game with endogenous existential risk. In our model, the cost of failure is no longer bounded only by the sunk cost of investment (I), but rather a systemic ruin parameter (D) that is correlated with development velocity and shared globally. As the disutility of catastrophe is embedded in both players' payoffs, the risk term mathematically cancels out of the equilibrium indifference condition. This creates a "suicide region" in the investment space where competitive pressures force rational agents to deploy AGI systems early, despite a negative risk-adjusted net present value. Furthermore, we show that "warning shots" (sub-existential disasters) will fail to deter AGI acceleration, as the winner-takes-all nature of the race remains intact. The race can only be halted if the cost of ruin is internalized, making safety research a prerequisite for economic viability. We derive the critical private liability threshold required to restore the option value of waiting and propose mechanism design interventions that can better ensure safe AGI research and socially responsible deployment. |
| 2025-12-08 | [Control of Discrete-Time Linear Systems with Charge-Balanced Inputs](http://arxiv.org/abs/2512.07506v1) | Yuzhen Qin, Zonglin Liu et al. | Electrical brain stimulation relies on externally applied currents to modulate neural activity, but safety constraints require each stimulation cycle to be charge-balanced, enforcing a zero net injected charge. However, how such charge-balanced stimulation works remains poorly understood. This paper investigates the ability of charge-balanced inputs to steer state trajectories in discrete-time linear systems. Motivated by both open-loop and adaptive neurostimulation protocols, we study two practically relevant input structures: periodic (repetitive) charge-balanced inputs and non-repetitive charge-balanced inputs. For each case, we derive novel reachability and controllability conditions. The theoretical results are further validated through numerical demonstrations of minimum-energy control input design. |
| 2025-12-08 | [From Real-World Traffic Data to Relevant Critical Scenarios](http://arxiv.org/abs/2512.07482v1) | Florian Lüttner, Nicole Neis et al. | The reliable operation of autonomous vehicles, automated driving functions, and advanced driver assistance systems across a wide range of relevant scenarios is critical for their development and deployment. Identifying a near-complete set of relevant driving scenarios for such functionalities is challenging due to numerous degrees of freedom involved, each affecting the outcomes of the driving scenario differently. Moreover, with increasing technical complexity of new functionalities, the number of potentially relevant, particularly "unknown unsafe" scenarios is increasing. To enhance validation efficiency, it is essential to identify relevant scenarios in advance, starting with simpler domains like highways before moving to more complex environments such as urban traffic. To address this, this paper focuses on analyzing lane change scenarios in highway traffic, which involve multiple degrees of freedom and present numerous safetyrelevant scenarios. We describe the process of data acquisition and processing of real-world data from public highway traffic, followed by the application of criticality measures on trajectory data to evaluate scenarios, as conducted within the AVEAS project (www.aveas.org). By linking the calculated measures to specific lane change driving scenarios and the conditions under which the data was collected, we facilitate the identification of safetyrelevant driving scenarios for various applications. Further, to tackle the extensive range of "unknown unsafe" scenarios, we propose a way to generate relevant scenarios by creating synthetic scenarios based on recorded ones. Consequently, we demonstrate and evaluate a processing chain that enables the identification of safety-relevant scenarios, the development of data-driven methods for extracting these scenarios, and the generation of synthetic critical scenarios via sampling on highways. |
| 2025-12-08 | [Permanent and transitory crime risk in variable-density hot spot analysis](http://arxiv.org/abs/2512.07467v1) | Ben Moews | Crime prevention measures, aiming for the effective and efficient spending of public resources, rely on the empirical analysis of spatial and temporal data for public safety outcomes. We perform a variable-density cluster analysis on crime incident reports in the City of Chicago for the years 2001--2022 to investigate changes in crime share composition for hot spots of different densities. Contributing to and going beyond the existing wealth of research on criminological applications in the operational research literature, we study the evolution of crime type shares in clusters over the course of two decades and demonstrate particularly notable impacts of the COVID-19 pandemic and its associated social contact avoidance measures, as well as a dependence of these effects on the primary function of city areas. Our results also indicate differences in the relative difficulty to address specific crime types, and an analysis of spatial autocorrelations further shows variations in incident uniformity between clusters and outlier areas at different distance radii. We discuss our findings in the context of the interplay between operational research and criminal justice, the practice of hot spot policing and public safety optimization, and the factors contributing to, and challenges and risks due to, data biases as an often neglected factor in criminological applications. |
| 2025-12-08 | [Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics](http://arxiv.org/abs/2512.07462v1) | Trung-Kiet Huynh, Duy-Minh Dao-Sy et al. | As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems. |
| 2025-12-08 | [AFarePart: Accuracy-aware Fault-resilient Partitioner for DNN Edge Accelerators](http://arxiv.org/abs/2512.07449v1) | Mukta Debnath, Krishnendu Guha et al. | Deep Neural Networks (DNNs) are increasingly deployed across distributed and resource-constrained platforms, such as System-on-Chip (SoC) accelerators and edge-cloud systems. DNNs are often partitioned and executed across heterogeneous processing units to optimize latency and energy. However, the reliability of these partitioned models under hardware faults and communication errors remains a critical yet underexplored topic, especially in safety-critical applications. In this paper, we propose an accuracy-aware, fault-resilient DNN partitioning framework targeting multi-objective optimization using NSGA-II, where accuracy degradation under fault conditions is introduced as a core metric alongside energy and latency. Our framework performs runtime fault injection during optimization and utilizes a feedback loop to prioritize fault-tolerant partitioning. We evaluate our approach on benchmark CNNs including AlexNet, SqueezeNet and ResNet18 on hardware accelerators, and demonstrate up to 27.7% improvement in fault tolerance with minimal increase in performance overhead. Our results highlight the importance of incorporating resilience into DNN partitioning, and thereby paving the way for robust AI inference in error-prone environments. |
| 2025-12-08 | [Systematic Evaluation of Black-Box Checking for Fast Bug Detection](http://arxiv.org/abs/2512.07434v1) | Bram Pellen, María Belén Rodríguez et al. | Combinations of active automata learning, model-based testing and model checking have been successfully used in numerous applications, e.g., for spotting bugs in implementations of major network protocols and to support refactoring of embedded controllers. However, in the large majority of these applications, model checking is only used at the very end, when no counterexample can be found anymore for the latest hypothesis model. This contrasts with the original proposal of black-box checking (BBC) by Peled, Vardi & Yannakakis, which applies model checking for all hypotheses, also the intermediate ones. In this article, we present the first systematic evaluation of the ability of BBC to find bugs quickly, based on 77 benchmarks models from real protocol implementations and controllers for which specifications of safety properties are available. Our main finding are: (a) In cases where the full model can be learned, BBC detects violations of the specifications with just 3.4% of the queries needed by an approach in which model checking is only used for the full model. (b) Even when the full model cannot be learned, BBC is still able to detect many violations of the specification. In particular, BBC manages to detect 94% of the safety properties violations in the challenging RERS 2019 industrial LTL benchmarks. (c) Our results also confirm that BBC is way more effective than existing MBT algorithms in finding deep bugs in implementations. |
| 2025-12-08 | [Training Language Models to Use Prolog as a Tool](http://arxiv.org/abs/2512.07407v1) | Niklas Mellgren, Peter Schneider-Kamp et al. | Ensuring reliable tool use is critical for safe agentic AI systems. Language models frequently produce unreliable reasoning with plausible but incorrect solutions that are difficult to verify. To address this, we investigate fine-tuning models to use Prolog as an external tool for verifiable computation. Using Group Relative Policy Optimization (GRPO), we fine-tune Qwen2.5-3B-Instruct on a cleaned GSM8K-Prolog-Prover dataset while varying (i) prompt structure, (ii) reward composition (execution, syntax, semantics, structure), and (iii) inference protocol: single-shot, best-of-N, and two agentic modes where Prolog is invoked internally or independently. Our reinforcement learning approach outperforms supervised fine-tuning, with our 3B model achieving zero-shot MMLU performance comparable to 7B few-shot results. Our findings reveal that: 1) joint tuning of prompt, reward, and inference shapes program syntax and logic; 2) best-of-N with external Prolog verification maximizes accuracy on GSM8K; 3) agentic inference with internal repair yields superior zero-shot generalization on MMLU-Stem and MMLU-Pro. These results demonstrate that grounding model reasoning in formal verification systems substantially improves reliability and auditability for safety-critical applications. The source code for reproducing our experiments is available under https://github.com/niklasmellgren/grpo-prolog-inference |
| 2025-12-08 | [Safety-Critical Control on Lie Groups Using Energy-Augmented Zeroing Control Barrier Functions](http://arxiv.org/abs/2512.07395v1) | Alessandro Letti, Riccardo Zanella et al. | We study safety-critical control on fully actuated mechanical systems by means of Zeroing Control Barrier Functions (ZCBFs) defined on Lie Groups. Specifically, we introduce and theoretically validate two classes of ZCBFs. The first enforces kinematic constraints, suitable for implementing obstacle avoidance algorithms. The second enforces kinetic energy limits along prescribed inertial-frame translational and rotational directions, relevant for ensuring safe physical interaction. Numerical simulations involving slit traversal and safe landing scenarios are presented to validate the effectiveness and versatility of the proposed methodology. |
| 2025-12-08 | [How Far are Modern Trackers from UAV-Anti-UAV? A Million-Scale Benchmark and New Baseline](http://arxiv.org/abs/2512.07385v1) | Chunhui Zhang, Li Liu et al. | Unmanned Aerial Vehicles (UAVs) offer wide-ranging applications but also pose significant safety and privacy violation risks in areas like airport and infrastructure inspection, spurring the rapid development of Anti-UAV technologies in recent years. However, current Anti-UAV research primarily focuses on RGB, infrared (IR), or RGB-IR videos captured by fixed ground cameras, with little attention to tracking target UAVs from another moving UAV platform. To fill this gap, we propose a new multi-modal visual tracking task termed UAV-Anti-UAV, which involves a pursuer UAV tracking a target adversarial UAV in the video stream. Compared to existing Anti-UAV tasks, UAV-Anti-UAV is more challenging due to severe dual-dynamic disturbances caused by the rapid motion of both the capturing platform and the target. To advance research in this domain, we construct a million-scale dataset consisting of 1,810 videos, each manually annotated with bounding boxes, a language prompt, and 15 tracking attributes. Furthermore, we propose MambaSTS, a Mamba-based baseline method for UAV-Anti-UAV tracking, which enables integrated spatial-temporal-semantic learning. Specifically, we employ Mamba and Transformer models to learn global semantic and spatial features, respectively, and leverage the state space model's strength in long-sequence modeling to establish video-level long-term context via a temporal token propagation mechanism. We conduct experiments on the UAV-Anti-UAV dataset to validate the effectiveness of our method. A thorough experimental evaluation of 50 modern deep tracking algorithms demonstrates that there is still significant room for improvement in the UAV-Anti-UAV domain. The dataset and codes will be available at {\color{magenta}https://github.com/983632847/Awesome-Multimodal-Object-Tracking}. |
| 2025-12-08 | [DGGAN: Degradation Guided Generative Adversarial Network for Real-time Endoscopic Video Enhancement](http://arxiv.org/abs/2512.07253v1) | Handing Xu, Zhenguo Nie et al. | Endoscopic surgery relies on intraoperative video, making image quality a decisive factor for surgical safety and efficacy. Yet, endoscopic videos are often degraded by uneven illumination, tissue scattering, occlusions, and motion blur, which obscure critical anatomical details and complicate surgical manipulation. Although deep learning-based methods have shown promise in image enhancement, most existing approaches remain too computationally demanding for real-time surgical use. To address this challenge, we propose a degradation-aware framework for endoscopic video enhancement, which enables real-time, high-quality enhancement by propagating degradation representations across frames. In our framework, degradation representations are first extracted from images using contrastive learning. We then introduce a fusion mechanism that modulates image features with these representations to guide a single-frame enhancement model, which is trained with a cycle-consistency constraint between degraded and restored images to improve robustness and generalization. Experiments demonstrate that our framework achieves a superior balance between performance and efficiency compared with several state-of-the-art methods. These results highlight the effectiveness of degradation-aware modeling for real-time endoscopic video enhancement. Nevertheless, our method suggests that implicitly learning and propagating degradation representation offer a practical pathway for clinical application. |

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



