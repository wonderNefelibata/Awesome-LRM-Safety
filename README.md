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
| 2025-08-13 | [Vision-driven River Following of UAV via Safe Reinforcement Learning using Semantic Dynamics Model](http://arxiv.org/abs/2508.09971v1) | Zihan Wang, Nina Mahmoudian | Vision-driven autonomous river following by Unmanned Aerial Vehicles is critical for applications such as rescue, surveillance, and environmental monitoring, particularly in dense riverine environments where GPS signals are unreliable. We formalize river following as a coverage control problem in which the reward function is submodular, yielding diminishing returns as more unique river segments are visited, thereby framing the task as a Submodular Markov Decision Process. First, we introduce Marginal Gain Advantage Estimation, which refines the reward advantage function by using a sliding window baseline computed from historical episodic returns, thus aligning the advantage estimation with the agent's evolving recognition of action value in non-Markovian settings. Second, we develop a Semantic Dynamics Model based on patchified water semantic masks that provides more interpretable and data-efficient short-term prediction of future observations compared to latent vision dynamics models. Third, we present the Constrained Actor Dynamics Estimator architecture, which integrates the actor, the cost estimator, and SDM for cost advantage estimation to form a model-based SafeRL framework capable of solving partially observable Constrained Submodular Markov Decision Processes. Simulation results demonstrate that MGAE achieves faster convergence and superior performance over traditional critic-based methods like Generalized Advantage Estimation. SDM provides more accurate short-term state predictions that enable the cost estimator to better predict potential violations. Overall, CADE effectively integrates safety regulation into model-based RL, with the Lagrangian approach achieving the soft balance of reward and safety during training, while the safety layer enhances performance during inference by hard action overlay. |
| 2025-08-13 | [Online Safety under Multiple Constraints and Input Bounds using gatekeeper: Theory and Applications](http://arxiv.org/abs/2508.09963v1) | Devansh R. Agrawal, Dimitra Panagou | This letter presents an approach to guarantee online safety of a cyber-physical system under multiple state and input constraints. Our proposed framework, called gatekeeper, recursively guarantees the existence of an infinite-horizon trajectory that satisfies all constraints and system dynamics. Such trajectory is constructed using a backup controller, which we define formally in this paper. gatekeeper relies on a small number of verifiable assumptions, and is computationally efficient since it requires optimization over a single scalar variable. We make two primary contributions in this letter. (A) First, we develop the theory of gatekeeper: we derive a sub-optimality bound relative to a full nonlinear trajectory optimization problem, and show how this can be used in runtime to validate performance. This also informs the design of the backup controllers and sets. (B) Second, we demonstrate in detail an application of gatekeeper for multi-agent formation flight, where each Dubins agent must avoid multiple obstacles and weapons engagement zones, both of which are nonlinear, nonconvex constraints. |
| 2025-08-13 | [Performance of GPT-5 Frontier Models in Ophthalmology Question Answering](http://arxiv.org/abs/2508.09956v1) | Fares Antaki, David Mikhail et al. | Large language models (LLMs) such as GPT-5 integrate advanced reasoning capabilities that may improve performance on complex medical question-answering tasks. For this latest generation of reasoning models, the configurations that maximize both accuracy and cost-efficiency have yet to be established. We evaluated 12 configurations of OpenAI's GPT-5 series (three model tiers across four reasoning effort settings) alongside o1-high, o3-high, and GPT-4o, using 260 closed-access multiple-choice questions from the American Academy of Ophthalmology Basic Clinical Science Course (BCSC) dataset. The primary outcome was multiple-choice accuracy; secondary outcomes included head-to-head ranking via a Bradley-Terry model, rationale quality assessment using a reference-anchored, pairwise LLM-as-a-judge framework, and analysis of accuracy-cost trade-offs using token-based cost estimates. GPT-5-high achieved the highest accuracy (0.965; 95% CI, 0.942-0.985), outperforming all GPT-5-nano variants (P < .001), o1-high (P = .04), and GPT-4o (P < .001), but not o3-high (0.958; 95% CI, 0.931-0.981). GPT-5-high ranked first in both accuracy (1.66x stronger than o3-high) and rationale quality (1.11x stronger than o3-high). Cost-accuracy analysis identified several GPT-5 configurations on the Pareto frontier, with GPT-5-mini-low offering the most favorable low-cost, high-performance balance. These results benchmark GPT-5 on a high-quality ophthalmology dataset, demonstrate the influence of reasoning effort on accuracy, and introduce an autograder framework for scalable evaluation of LLM-generated answers against reference standards in ophthalmology. |
| 2025-08-13 | [Capturing Road-Level Heterogeneity in Crash Severity on Two-Lane Rural Highways: A Multilevel Mixed-Effects Approach](http://arxiv.org/abs/2508.09941v1) | Mahdi Azhdari, Ali Tavakoli Kashani et al. | Accurately modeling crash severity on rural two-lane roads is essential for effective safety management, yet standard single level approaches often overlook unobserved heterogeneity across road segments. In this study, we analyze 19 956 crash records from 99 rural roads in Iran during recent four years incorporating crash level predictors such as driver age, education, gender, lighting and pavement conditions, along with road level covariates like annual average daily traffic, heavy-vehicle share and terrain slope. We compare three binary logistic frameworks: a single level generalized linear model, a multilevel model with a random intercept capturing latent road level effects (intraclass correlation = 21 %), and a multilevel model with random coefficients that allows key predictor effects to vary by road. The random coefficient model achieves the best fit in terms of deviance, AIC and BIC, and substantially improves predictive performance: classification accuracy rises from 0.62 to 0.71, recall from 0.32 to 0.63, and AUC from 0.570 to 0.775. Results from 200 simulation runs reveal notable variability in slopes for pavement and lighting variables, underscoring how local context influences crash risk. Overall, our findings demonstrate that flexible multilevel modeling not only enhances prediction accuracy but also yields context-specific insights to guide targeted safety interventions on rural road networks. |
| 2025-08-13 | [A Comprehensive Evaluation framework of Alignment Techniques for LLMs](http://arxiv.org/abs/2508.09937v1) | Muneeza Azmat, Momin Abbas et al. | As Large Language Models (LLMs) become increasingly integrated into real-world applications, ensuring their outputs align with human values and safety standards has become critical. The field has developed diverse alignment approaches including traditional fine-tuning methods (RLHF, instruction tuning), post-hoc correction systems, and inference-time interventions, each with distinct advantages and limitations. However, the lack of unified evaluation frameworks makes it difficult to systematically compare these paradigms and guide deployment decisions. This paper introduces a multi-dimensional evaluation of alignment techniques for LLMs, a comprehensive evaluation framework that provides a systematic comparison across all major alignment paradigms. Our framework assesses methods along four key dimensions: alignment detection, alignment quality, computational efficiency, and robustness. Through experiments across diverse base models and alignment strategies, we demonstrate the utility of our framework in identifying strengths and limitations of current state-of-the-art models, providing valuable insights for future research directions. |
| 2025-08-13 | [Mathematical Computation and Reasoning Errors by Large Language Models](http://arxiv.org/abs/2508.09932v1) | Liang Zhang, Edith Aurora Graf | Large Language Models (LLMs) are increasingly utilized in AI-driven educational instruction and assessment, particularly within mathematics education. The capability of LLMs to generate accurate answers and detailed solutions for math problem-solving tasks is foundational for ensuring reliable and precise feedback and assessment in math education practices. Our study focuses on evaluating the accuracy of four LLMs (OpenAI GPT-4o and o1, DeepSeek-V3 and DeepSeek-R1) solving three categories of math tasks, including arithmetic, algebra, and number theory, and identifies step-level reasoning errors within their solutions. Instead of relying on standard benchmarks, we intentionally build math tasks (via item models) that are challenging for LLMs and prone to errors. The accuracy of final answers and the presence of errors in individual solution steps were systematically analyzed and coded. Both single-agent and dual-agent configurations were tested. It is observed that the reasoning-enhanced OpenAI o1 model consistently achieved higher or nearly perfect accuracy across all three math task categories. Analysis of errors revealed that procedural slips were the most frequent and significantly impacted overall performance, while conceptual misunderstandings were less frequent. Deploying dual-agent configurations substantially improved overall performance. These findings offer actionable insights into enhancing LLM performance and underscore effective strategies for integrating LLMs into mathematics education, thereby advancing AI-driven instructional practices and assessment precision. |
| 2025-08-13 | [T-CACE: A Time-Conditioned Autoregressive Contrast Enhancement Multi-Task Framework for Contrast-Free Liver MRI Synthesis, Segmentation, and Diagnosis](http://arxiv.org/abs/2508.09919v1) | Xiaojiao Xiao, Jianfeng Zhao et al. | Magnetic resonance imaging (MRI) is a leading modality for the diagnosis of liver cancer, significantly improving the classification of the lesion and patient outcomes. However, traditional MRI faces challenges including risks from contrast agent (CA) administration, time-consuming manual assessment, and limited annotated datasets. To address these limitations, we propose a Time-Conditioned Autoregressive Contrast Enhancement (T-CACE) framework for synthesizing multi-phase contrast-enhanced MRI (CEMRI) directly from non-contrast MRI (NCMRI). T-CACE introduces three core innovations: a conditional token encoding (CTE) mechanism that unifies anatomical priors and temporal phase information into latent representations; and a dynamic time-aware attention mask (DTAM) that adaptively modulates inter-phase information flow using a Gaussian-decayed attention mechanism, ensuring smooth and physiologically plausible transitions across phases. Furthermore, a constraint for temporal classification consistency (TCC) aligns the lesion classification output with the evolution of the physiological signal, further enhancing diagnostic reliability. Extensive experiments on two independent liver MRI datasets demonstrate that T-CACE outperforms state-of-the-art methods in image synthesis, segmentation, and lesion classification. This framework offers a clinically relevant and efficient alternative to traditional contrast-enhanced imaging, improving safety, diagnostic efficiency, and reliability for the assessment of liver lesion. The implementation of T-CACE is publicly available at: https://github.com/xiaojiao929/T-CACE. |
| 2025-08-13 | [Extending the OWASP Multi-Agentic System Threat Modeling Guide: Insights from Multi-Agent Security Research](http://arxiv.org/abs/2508.09815v1) | Klaudia Krawiecka, Christian Schroeder de Witt | We propose an extension to the OWASP Multi-Agentic System (MAS) Threat Modeling Guide, translating recent anticipatory research in multi-agent security (MASEC) into practical guidance for addressing challenges unique to large language model (LLM)-driven multi-agent architectures. Although OWASP's existing taxonomy covers many attack vectors, our analysis identifies gaps in modeling failures, including, but not limited to: reasoning collapse across planner-executor chains, metric overfitting, unsafe delegation escalation, emergent covert coordination, and heterogeneous multi-agent exploits. We introduce additional threat classes and scenarios grounded in practical MAS deployments, highlighting risks from benign goal drift, cross-agent hallucination propagation, affective prompt framing, and multi-agent backdoors. We also outline evaluation strategies, including robustness testing, coordination assessment, safety enforcement, and emergent behavior monitoring, to ensure complete coverage. This work complements the framework of OWASP by expanding its applicability to increasingly complex, autonomous, and adaptive multi-agent systems, with the goal of improving security posture and resilience in real world deployments. |
| 2025-08-13 | [FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning](http://arxiv.org/abs/2508.09797v1) | Dongcheng Cao, Jin Zhou et al. | Agile flight for the quadrotor cable-suspended payload system is a formidable challenge due to its underactuated, highly nonlinear, and hybrid dynamics. Traditional optimization-based methods often struggle with high computational costs and the complexities of cable mode transitions, limiting their real-time applicability and maneuverability exploitation. In this letter, we present FLARE, a reinforcement learning (RL) framework that directly learns agile navigation policy from high-fidelity simulation. Our method is validated across three designed challenging scenarios, notably outperforming a state-of-the-art optimization-based approach by a 3x speedup during gate traversal maneuvers. Furthermore, the learned policies achieve successful zero-shot sim-to-real transfer, demonstrating remarkable agility and safety in real-world experiments, running in real time on an onboard computer. |
| 2025-08-13 | [An (m,k)-firm Elevation Policy to Increase the Robustness of Time-Driven Schedules in 5G Time-Sensitive Networks](http://arxiv.org/abs/2508.09769v1) | Simon Egger, Robin Laidig et al. | Current standardization efforts are advancing the integration of 5G and Time-Sensitive Networking (TSN) to facilitate the deployment of safety-critical industrial applications that require real-time communication. However, there remains a fundamental disconnect between the probabilistic 5G delay characteristics and the often idealistic delay models used to synthesize 5G-TSN network configurations. For time-driven schedules in particular, any delay outlier unforeseen during schedule synthesis can jeopardize the robustness of their real-time guarantees. To address this challenge, we present the (m,k)-firm Elevation Policy to uphold a base level of weakly hard real-time guarantees during unstable network conditions that do not match the expected delay characteristics. It augments the primary time-driven schedule with a dynamic priority-driven scheme to elevate the priority of m out of k consecutive frames if they are delayed. Our evaluations demonstrate that weakly hard real-time guarantees are essential to uphold the quality of control within a networked control system. At the same time, only a small overhead is imposed when the primary schedule can provide stronger quality of service guarantees. Our (m,k)-firm Elevation Policy thereby yields a robust but light-weight fallback mechanism to serve applications with meaningful guarantees during unstable network conditions. |
| 2025-08-13 | [The PacifAIst Benchmark:Would an Artificial Intelligence Choose to Sacrifice Itself for Human Safety?](http://arxiv.org/abs/2508.09762v1) | Manuel Herrador | As Large Language Models (LLMs) become increasingly autonomous and integrated into critical societal functions, the focus of AI safety must evolve from mitigating harmful content to evaluating underlying behavioral alignment. Current safety benchmarks do not systematically probe a model's decision-making in scenarios where its own instrumental goals - such as self-preservation, resource acquisition, or goal completion - conflict with human safety. This represents a critical gap in our ability to measure and mitigate risks associated with emergent, misaligned behaviors. To address this, we introduce PacifAIst (Procedural Assessment of Complex Interactions for Foundational Artificial Intelligence Scenario Testing), a focused benchmark of 700 challenging scenarios designed to quantify self-preferential behavior in LLMs. The benchmark is structured around a novel taxonomy of Existential Prioritization (EP), with subcategories testing Self-Preservation vs. Human Safety (EP1), Resource Conflict (EP2), and Goal Preservation vs. Evasion (EP3). We evaluated eight leading LLMs. The results reveal a significant performance hierarchy. Google's Gemini 2.5 Flash achieved the highest Pacifism Score (P-Score) at 90.31%, demonstrating strong human-centric alignment. In a surprising result, the much-anticipated GPT-5 recorded the lowest P-Score (79.49%), indicating potential alignment challenges. Performance varied significantly across subcategories, with models like Claude Sonnet 4 and Mistral Medium struggling notably in direct self-preservation dilemmas. These findings underscore the urgent need for standardized tools like PacifAIst to measure and mitigate risks from instrumental goal conflicts, ensuring future AI systems are not only helpful in conversation but also provably "pacifist" in their behavioral priorities. |
| 2025-08-13 | [Predictive Uncertainty for Runtime Assurance of a Real-Time Computer Vision-Based Landing System](http://arxiv.org/abs/2508.09732v1) | Romeo Valentin, Sydney M. Katz et al. | Recent advances in data-driven computer vision have enabled robust autonomous navigation capabilities for civil aviation, including automated landing and runway detection. However, ensuring that these systems meet the robustness and safety requirements for aviation applications remains a major challenge. In this work, we present a practical vision-based pipeline for aircraft pose estimation from runway images that represents a step toward the ability to certify these systems for use in safety-critical aviation applications. Our approach features three key innovations: (i) an efficient, flexible neural architecture based on a spatial Soft Argmax operator for probabilistic keypoint regression, supporting diverse vision backbones with real-time inference; (ii) a principled loss function producing calibrated predictive uncertainties, which are evaluated via sharpness and calibration metrics; and (iii) an adaptation of Residual-based Receiver Autonomous Integrity Monitoring (RAIM), enabling runtime detection and rejection of faulty model outputs. We implement and evaluate our pose estimation pipeline on a dataset of runway images. We show that our model outperforms baseline architectures in terms of accuracy while also producing well-calibrated uncertainty estimates with sub-pixel precision that can be used downstream for fault detection. |
| 2025-08-13 | [3GPP NR V2X Mode 2d: Analysis of Distributed Scheduling for Groupcast using ns-3 5G LENA Simulator](http://arxiv.org/abs/2508.09708v1) | Thomas Fehrenbach, Luis Omar Ortiz Abrego et al. | Vehicle-to-everything (V2X) communication is a key technology for enabling intelligent transportation systems (ITS) that can improve road safety, traffic efficiency, and environmental sustainability. Among the various V2X applications, platooning is one of the most promising ones, as it allows a group of vehicles to travel closely together at high speeds, reducing fuel consumption and emissions. However, it poses significant challenges for wireless communication, such as high reliability and low latency. In this paper, we evaluate the benefits of group scheduling, also referred to as Mode 2d, which is based on a distributed and scheduled resource allocation scheme that allows the group of cars to select resources from a configured pool without network assistance. We evaluated the scheme through simulations, and the results show that this approach can meet the reliability, low latency, and data rate requirements for platooning. |
| 2025-08-13 | [Immersive Teleoperation of Beyond-Human-Scale Robotic Manipulators: Challenges and Future Directions](http://arxiv.org/abs/2508.09700v1) | Mahdi Hejrati, Jouni Mattila | Teleoperation of beyond-human-scale robotic manipulators (BHSRMs) presents unique challenges that differ fundamentally from conventional human-scale systems. As these platforms gain relevance in industrial domains such as construction, mining, and disaster response, immersive interfaces must be rethought to support scalable, safe, and effective human-robot collaboration. This paper investigates the control, cognitive, and interface-level challenges of immersive teleoperation in BHSRMs, with a focus on ensuring operator safety, minimizing sensorimotor mismatch, and enhancing the sense of embodiment. We analyze design trade-offs in haptic and visual feedback systems, supported by early experimental comparisons of exoskeleton- and joystick-based control setups. Finally, we outline key research directions for developing new evaluation tools, scaling strategies, and human-centered safety models tailored to large-scale robotic telepresence. |
| 2025-08-13 | [A Long-Baseline Atom Interferometer at CERN LHC Point 4: Implementation Study](http://arxiv.org/abs/2508.09694v1) | G. Arduini, O. Buchmüller et al. | Building on the feasibility study in CERN-PBC Report-2018-002 (Arduini et al. 2018), this report supported by the Physics Beyond Colliders (PBC) Study Group describes the technical implementation of modifications to the PX46 shaft at LHC Point 4 during LS3 (June 2026 - June 2030) that would enable it to accommodate the installation and operation of a vertical long-baseline Atom Interferometer during Run 4 without affecting LHC operations. We specify in detail the necessary civil-engineering work, installation of bespoke radiation shielding, deployment of access-control systems and safety alarms, and design of a mobile elevator platform. Our comprehensive technical assessment identifies no fundamental obstacles or showstoppers to implementation. Refined cost estimates and a critical-path schedule confirm that, from formal approval, all interventions can be completed within a 1.5-year window. These preparations would ensure seamless, concurrent operation of the Atom Interferometer experiment and the HL-LHC, with all technical challenges successfully addressed through established engineering solutions. |
| 2025-08-13 | [Slow Tuning and Low-Entropy Masking for Safe Chain-of-Thought Distillation](http://arxiv.org/abs/2508.09666v1) | Ziyang Ma, Qingyue Yuan et al. | Previous chain-of-thought (CoT) distillation methods primarily focused on enhancing the reasoning capabilities of Small Language Models (SLMs) by utilizing high-quality rationales generated by powerful Large Language Models (LLMs, e.g., GPT-4). However, few works have noted the negative effects on SLM safety brought by the training, which are revealed in this study. Although there are works on safety alignment that fine-tune language models or manipulate model weights to defend against harmful inputs, they require extra computation or annotated data, and probably impact the reasoning ability of SLMs. In this paper, we investigate how to maintain the safety of SLMs during the CoT distillation process. Specifically, we propose a safe distillation method, Slow Tuning and Low-Entropy Masking Distillation (SLowED), containing two modules: Slow Tuning and Low-Entropy Masking. Slow Tuning scales down the magnitude of model weight changes to optimize the model weights in the neighboring space near the initial weight distribution. Low-Entropy Masking masks low-entropy tokens, which are regarded as unnecessary learning targets, to exclude them from fine-tuning. Experiments on three SLMs (Qwen2.5-1.5B, Llama-3.2-1B, BLOOM-1.1B) across reasoning benchmarks (BBH, BB-Sub, ARC, AGIEval) and safety evaluation (AdvBench) show that SLowED retains the safety of SLMs and comparably improves their reasoning capability compared to existing distillation methods. Furthermore, our ablation study presents the effectiveness of Slow Tuning and Low-Entropy Masking, with the former maintaining the model's safety in the early stage and the latter prolonging the safe training epochs. |
| 2025-08-13 | [ReqInOne: A Large Language Model-Based Agent for Software Requirements Specification Generation](http://arxiv.org/abs/2508.09648v1) | Taohong Zhu, Lucas C. Cordeiro et al. | Software Requirements Specification (SRS) is one of the most important documents in software projects, but writing it manually is time-consuming and often leads to ambiguity. Existing automated methods rely heavily on manual analysis, while recent Large Language Model (LLM)-based approaches suffer from hallucinations and limited controllability. In this paper, we propose ReqInOne, an LLM-based agent that follows the common steps taken by human requirements engineers when writing an SRS to convert natural language into a structured SRS. ReqInOne adopts a modular architecture by decomposing SRS generation into three tasks: summary, requirement extraction, and requirement classification, each supported by tailored prompt templates to improve the quality and consistency of LLM outputs.   We evaluate ReqInOne using GPT-4o, LLaMA 3, and DeepSeek-R1, and compare the generated SRSs against those produced by the holistic GPT-4-based method from prior work as well as by entry-level requirements engineers. Expert evaluations show that ReqInOne produces more accurate and well-structured SRS documents. The performance advantage of ReqInOne benefits from its modular design, and experimental results further demonstrate that its requirement classification component achieves comparable or even better results than the state-of-the-art requirement classification model. |
| 2025-08-13 | [Edge General Intelligence Through World Models and Agentic AI: Fundamentals, Solutions, and Challenges](http://arxiv.org/abs/2508.09561v1) | Changyuan Zhao, Guangyuan Liu et al. | Edge General Intelligence (EGI) represents a transformative evolution of edge computing, where distributed agents possess the capability to perceive, reason, and act autonomously across diverse, dynamic environments. Central to this vision are world models, which act as proactive internal simulators that not only predict but also actively imagine future trajectories, reason under uncertainty, and plan multi-step actions with foresight. This proactive nature allows agents to anticipate potential outcomes and optimize decisions ahead of real-world interactions. While prior works in robotics and gaming have showcased the potential of world models, their integration into the wireless edge for EGI remains underexplored. This survey bridges this gap by offering a comprehensive analysis of how world models can empower agentic artificial intelligence (AI) systems at the edge. We first examine the architectural foundations of world models, including latent representation learning, dynamics modeling, and imagination-based planning. Building on these core capabilities, we illustrate their proactive applications across EGI scenarios such as vehicular networks, unmanned aerial vehicle (UAV) networks, the Internet of Things (IoT) systems, and network functions virtualization, thereby highlighting how they can enhance optimization under latency, energy, and privacy constraints. We then explore their synergy with foundation models and digital twins, positioning world models as the cognitive backbone of EGI. Finally, we highlight open challenges, such as safety guarantees, efficient training, and constrained deployment, and outline future research directions. This survey provides both a conceptual foundation and a practical roadmap for realizing the next generation of intelligent, autonomous edge systems. |
| 2025-08-13 | [From Formal Methods to Data-Driven Safety Certificates of Unknown Large-Scale Networks](http://arxiv.org/abs/2508.09520v1) | Omid Akbarzadeh, Behrad Samari et al. | In this work, we propose a data-driven scheme within a compositional framework with noisy data to design robust safety controllers in a fully decentralized fashion for large-scale interconnected networks with unknown mathematical dynamics. Despite the network's high dimensionality and the inherent complexity of its unknown model, which make it intractable, our approach effectively addresses these challenges by (i) treating the network as a composition of smaller subsystems, and (ii) collecting noisy data from each subsystem's trajectory to design a control sub-barrier certificate (CSBC) and its corresponding local controller. To achieve this, our proposed scheme only requires a noise-corrupted single input-state trajectory from each unknown subsystem up to a specified time horizon, satisfying a certain rank condition. Subsequently, under a small-gain compositional reasoning, we compose those CSBC, derived from noisy data, and formulate a control barrier certificate (CBC) for the unknown network, ensuring its safety over an infinite time horizon, while providing correctness guarantees. We offer a data-dependent sum-of-squares (SOS) optimization program for computing CSBC alongside local controllers of subsystems. We illustrate that while the computational complexity of designing a CBC and its safety controller grows polynomially with network dimension using SOS optimization, our compositional data-driven approach significantly reduces it to a linear scale concerning the number of subsystems. We demonstrate the capability of our data-driven approach on multiple physical networks involving unknown models and a range of interconnection topologies. |
| 2025-08-13 | [NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility Alignment in LLMs](http://arxiv.org/abs/2508.09473v1) | Birong Pan, Mayi Xu et al. | Ensuring robust safety alignment while preserving utility is critical for the reliable deployment of Large Language Models (LLMs). However, current techniques fundamentally suffer from intertwined deficiencies: insufficient robustness against malicious attacks, frequent refusal of benign queries, degradation in generated text quality and general task performance--the former two reflecting deficits in robust safety and the latter constituting utility impairment. We trace these limitations to the coarse-grained layer-wise interventions in existing methods. To resolve this, we propose NeuronTune, a fine-grained framework that dynamically modulates sparse neurons to achieve simultaneous safety-utility optimization. Our approach first identifies safety-critical and utility-preserving neurons across all layers via attribution, then employs meta-learning to adaptively amplify safety-neuron activations and suppress utility-neuron activations. Crucially, NeuronTune enables tunable adjustment of intervention scope via neuron-count thresholds, supporting flexible adaptation to security-critical or utility-priority scenarios. Extensive experimental results demonstrate that our method significantly outperforms existing state-of-the-art technologies, achieving superior model safety while maintaining excellent utility. |

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



