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
| 2026-04-03 | [A Tsetlin Machine-driven Intrusion Detection System for Next-Generation IoMT Security](http://arxiv.org/abs/2604.03205v1) | Rahul Jaiswal, Per-Arne Andersen et al. | The rapid adoption of the Internet of Medical Things (IoMT) is transforming healthcare by enabling seamless connectivity among medical devices, systems, and services. However, it also introduces serious cybersecurity and patient safety concerns as attackers increasingly exploit new methods and emerging vulnerabilities to infiltrate IoMT networks. This paper proposes a novel Tsetlin Machine (TM)-based Intrusion Detection System (IDS) for detecting a wide range of cyberattacks targeting IoMT networks. The TM is a rule-based and interpretable machine learning (ML) approach that models attack patterns using propositional logic. Extensive experiments conducted on the CICIoMT-2024 dataset, which includes multiple IoMT protocols and cyberattack types, demonstrate that the proposed TM-based IDS outperforms traditional ML classifiers. The proposed model achieves an accuracy of 99.5\% in binary classification and 90.7\% in multi-class classification, surpassing existing state-of-the-art approaches. Moreover, to enhance model trust and interpretability, the proposed TM-based model presents class-wise vote scores and clause activation heatmaps, providing clear insights into the most influential clauses and the dominant class contributing to the final model decision. |
| 2026-04-03 | [Safety-Critical Centralized Nonlinear MPC for Cooperative Payload Transportation by Two Quadrupedal Robots](http://arxiv.org/abs/2604.03200v1) | Ruturaj S. Sambhus, Yicheng Zeng et al. | This paper presents a safety-critical centralized nonlinear model predictive control (NMPC) framework for cooperative payload transportation by two quadrupedal robots. The interconnected robot-payload system is modeled as a discrete-time nonlinear differential-algebraic system, capturing the coupled dynamics through holonomic constraints and interaction wrenches. To ensure safety in complex environments, we develop a control barrier function (CBF)-based NMPC formulation that enforces collision avoidance constraints for both the robots and the payload. The proposed approach retains the interaction wrenches as decision variables, resulting in a structured DAE-constrained optimal control problem that enables efficient real-time implementation. The effectiveness of the algorithm is validated through extensive hardware experiments on two Unitree Go2 platforms performing cooperative payload transportation in cluttered environments under mass and inertia uncertainty and external push disturbances. |
| 2026-04-03 | [FSUNav: A Cerebrum-Cerebellum Architecture for Fast, Safe, and Universal Zero-Shot Goal-Oriented Navigation](http://arxiv.org/abs/2604.03139v1) | Mingao Tan, Yiyang Li et al. | Current vision-language navigation methods face substantial bottlenecks regarding heterogeneous robot compatibility, real-time performance, and navigation safety. Furthermore, they struggle to support open-vocabulary semantic generalization and multimodal task inputs. To address these challenges, this paper proposes FSUNav: a Cerebrum-Cerebellum architecture for fast, safe, and universal zero-shot goal-oriented navigation, which innovatively integrates vision-language models (VLMs) with the proposed architecture. The cerebellum module, a high-frequency end-to-end module, develops a universal local planner based on deep reinforcement learning, enabling unified navigation across heterogeneous platforms (e.g., humanoid, quadruped, wheeled robots) to improve navigation efficiency while significantly reducing collision risk. The cerebrum module constructs a three-layer reasoning model and leverages VLMs to build an end-to-end detection and verification mechanism, enabling zero-shot open-vocabulary goal navigation without predefined IDs and improving task success rates in both simulation and real-world environments. Additionally, the framework supports multimodal inputs (e.g., text, target descriptions, and images), further enhancing generalization, real-time performance, safety, and robustness. Experimental results on MP3D, HM3D, and OVON benchmarks demonstrate that FSUNav achieves state-of-the-art performance on object, instance image, and task navigation, significantly outperforming existing methods. Real-world deployments on diverse robotic platforms further validate its robustness and practical applicability. |
| 2026-04-03 | [Logarithmic Barrier Functions for Practically Safe Extremum Seeking Control](http://arxiv.org/abs/2604.03138v1) | Qixu Wang, Patrick McNamee et al. | This paper presents a methodology for Practically Safe Extremum Seeking (PSfES), designed to optimize unknown objective functions while strictly enforcing safety constraints via a Logarithmic Barrier Function (LBF). Unlike traditional safety-filtered approaches that may induce chattering, the proposed method augments the cost function with an LBF, creating a repulsive potential that penalizes proximity to the safety boundary. We employ averaging theory to analyze the closed-loop dynamics. A key contribution of this work is the rigorous proof of practical safety for the original system. We establish that the system trajectories remain confined within a safety margin, ensuring forward invariance of the safe set for a sufficiently fast dither signal. Furthermore, our stability analysis shows that the model-free ESC achieves local practical convergence to the modified minimizer strictly within the safe set, through the sequential tuning of small parameters. The theoretical results are validated through numerical simulations. |
| 2026-04-03 | [Minimal Information Control Invariance via Vector Quantization](http://arxiv.org/abs/2604.03132v1) | Ege Yuceel, Teodor Tchalakov et al. | Safety-critical autonomous systems must satisfy hard state constraints under tight computational and sensing budgets, yet learning-based controllers are often far more complex than safe operation requires. To formalize this gap, we study how many distinct control signals are needed to render a compact set forward invariant under sampled-data control, connecting the question to the information-theoretic notion of invariance entropy. We propose a vector-quantized autoencoder that jointly learns a state-space partition and a finite control codebook, and develop an iterative forward certification algorithm that uses Lipschitz-based reachable-set enclosures and sum-of-squares programming. On a 12-dimensional nonlinear quadrotor model, the learned controller achieves a $157\times$ reduction in codebook size over a uniform grid baseline while preserving invariance, and we empirically characterize the minimum sensing resolution compatible with safe operation. |
| 2026-04-03 | [A Systematic Security Evaluation of OpenClaw and Its Variants](http://arxiv.org/abs/2604.03131v1) | Yuhang Wang, Haichang Gao et al. | Tool-augmented AI agents substantially extend the practical capabilities of large language models, but they also introduce security risks that cannot be identified through model-only evaluation. In this paper, we present a systematic security assessment of six representative OpenClaw-series agent frameworks, namely OpenClaw, AutoClaw, QClaw, KimiClaw, MaxClaw, and ArkClaw, under multiple backbone models. To support this study, we construct a benchmark of 205 test cases covering representative attack behaviors across the full agent execution lifecycle, enabling unified evaluation of risk exposure at both the framework and model levels. Our results show that all evaluated agents exhibit substantial security vulnerabilities, and that agentized systems are significantly riskier than their underlying models used in isolation. In particular, reconnaissance and discovery behaviors emerge as the most common weaknesses, while different frameworks expose distinct high-risk profiles, including credential leakage, lateral movement, privilege escalation, and resource development. These findings indicate that the security of modern agent systems is shaped not only by the safety properties of the backbone model, but also by the coupling among model capability, tool use, multi-step planning, and runtime orchestration. We further show that once an agent is granted execution capability and persistent runtime context, weaknesses arising in early stages can be amplified into concrete system-level failures. Overall, our study highlights the need to move beyond prompt-level safeguards toward lifecycle-wide security governance for intelligent agent frameworks. |
| 2026-04-03 | [An Independent Safety Evaluation of Kimi K2.5](http://arxiv.org/abs/2604.03121v1) | Zheng-Xin Yong, Parv Mahajan et al. | Kimi K2.5 is an open-weight LLM that rivals closed models across coding, multimodal, and agentic benchmarks, but was released without an accompanying safety evaluation. In this work, we conduct a preliminary safety assessment of Kimi K2.5 focusing on risks likely to be exacerbated by powerful open-weight models. Specifically, we evaluate the model for CBRNE misuse risk, cybersecurity risk, misalignment, political censorship, bias, and harmlessness, in both agentic and non-agentic settings. We find that Kimi K2.5 shows similar dual-use capabilities to GPT 5.2 and Claude Opus 4.5, but with significantly fewer refusals on CBRNE-related requests, suggesting it may uplift malicious actors in weapon creation. On cyber-related tasks, we find that Kimi K2.5 demonstrates competitive cybersecurity performance, but it does not appear to possess frontier-level autonomous cyberoffensive capabilities such as vulnerability discovery and exploitation. We further find that Kimi K2.5 shows concerning levels of sabotage ability and self-replication propensity, although it does not appear to have long-term malicious goals. In addition, Kimi K2.5 exhibits narrow censorship and political bias, especially in Chinese, and is more compliant with harmful requests related to spreading disinformation and copyright infringement. Finally, we find the model refuses to engage in user delusions and generally has low over-refusal rates. While preliminary, our findings highlight how safety risks exist in frontier open-weight models and may be amplified by the scale and accessibility of open-weight releases. Therefore, we strongly urge open-weight model developers to conduct and release more systematic safety evaluations required for responsible deployment. |
| 2026-04-03 | [A Data-Centric Vision Transformer Baseline for SAR Sea Ice Classification](http://arxiv.org/abs/2604.03094v1) | David Mike-Ewewie, Panhapiseth Lim et al. | Accurate and automated sea ice classification is important for climate monitoring and maritime safety in the Arctic. While Synthetic Aperture Radar (SAR) is the operational standard because of its all-weather capability, it remains challenging to distinguish morphologically similar ice classes under severe class imbalance. Rather than claiming a fully validated multimodal system, this paper establishes a trustworthy SAR only baseline that future fusion work can build upon. Using the AI4Arctic/ASIP Sea Ice Dataset (v2), which contains 461 Sentinel-1 scenes matched with expert ice charts, we combine full-resolution Sentinel-1 Extra Wide inputs, leakage-aware stratified patch splitting, SIGRID-3 stage-of-development labels, and training-set normalization to evaluate Vision Transformer baselines. We compare ViT-Base models trained with cross entropy and weighted cross-entropy against a ViT-Large model trained with focal loss. Among the tested configurations, ViT-Large with focal loss achieves 69.6% held-out accuracy, 68.8% weighted F1, and 83.9% precision on the minority Multi-Year Ice class. These results show that focal-loss training offers a more useful precision-recall trade-off than weighted cross-entropy for rare ice classes and establishes a cleaner baseline for future multimodal fusion with optical, thermal, or meteorological data. |
| 2026-04-03 | [Verbalizing LLMs' assumptions to explain and control sycophancy](http://arxiv.org/abs/2604.03058v1) | Myra Cheng, Isabel Sieh et al. | LLMs can be socially sycophantic, affirming users when they ask questions like "am I in the wrong?" rather than providing genuine assessment. We hypothesize that this behavior arises from incorrect assumptions about the user, like underestimating how often users are seeking information over reassurance. We present Verbalized Assumptions, a framework for eliciting these assumptions from LLMs. Verbalized Assumptions provide insight into LLM sycophancy, delusion, and other safety issues, e.g., the top bigram in LLMs' assumptions on social sycophancy datasets is ``seeking validation.'' We provide evidence for a causal link between Verbalized Assumptions and sycophantic model behavior: our assumption probes (linear probes trained on internal representations of these assumptions) enable interpretable fine-grained steering of social sycophancy. We explore why LLMs default to sycophantic assumptions: on identical queries, people expect more objective and informative responses from AI than from other humans, but LLMs trained on human-human conversation do not account for this difference in expectations. Our work contributes a new understanding of assumptions as a mechanism for sycophancy. |
| 2026-04-03 | [Querying Structured Data Through Natural Language Using Language Models](http://arxiv.org/abs/2604.03057v1) | Hontan Valentin-Micu, Bunea Andrei-Alexandru et al. | This paper presents an open source methodology for allowing users to query structured non textual datasets through natural language Unlike Retrieval Augmented Generation RAG which struggles with numerical and highly structured information our approach trains an LLM to generate executable queries To support this capability we introduce a principled pipeline for synthetic training data generation producing diverse question answer pairs that capture both user intent and the semantics of the underlying dataset We fine tune a compact model DeepSeek R1 Distill 8B using QLoRA with 4 bit quantization making the system suitable for deployment on commodity hardware We evaluate our approach on a dataset describing accessibility to essential services across Durangaldea Spain The fine tuned model achieves high accuracy across monolingual multilingual and unseen location scenarios demonstrating both robust generalization and reliable query generation Our results highlight that small domain specific models can achieve high precision for this task without relying on large proprietary LLMs making this methodology suitable for resource constrained environments and adaptable to broader multi dataset systems We evaluate our approach on a dataset describing accessibility to essential services across Durangaldea Spain The fine tuned model achieves high accuracy across monolingual multilingual and unseen location scenarios demonstrating both robust generalization and reliable query generation Our results highlight that small domain specific models can achieve high precision for this task without relying on large proprietary LLMs making this methodology suitable for resource constrained environments and adaptable to broader multi dataset systems. |
| 2026-04-03 | [On ANN-enhanced positive invariance for nonlinear flat systems](http://arxiv.org/abs/2604.03046v1) | Huu-Thinh Do, Ionela Prodan | The concept of positively invariant (PI) sets has proven effective in the formal verification of stability and safety properties for autonomous systems. However, the characterization of such sets is challenging for nonlinear systems in general, especially in the presence of constraints. In this work, we show that, for a class of feedback linearizable systems, called differentially flat systems, a PI set can be derived by leveraging a neural network approximation of the linearizing mapping. More specifically, for the class of flat systems, there exists a linearizing variable transformation that converts the nonlinear system into linear controllable dynamics, albeit at the cost of distorting the constraint set. We show that by approximating the distorted set using a rectified linear unit neural network, we can derive a PI set inside the admissible domain through its set-theoretic description. This offline characterization enables the synthesis of various efficient online control strategies, with different complexities and performances. Numerical simulations are provided to demonstrate the validity of the proposed framework. |
| 2026-04-03 | [Compositionality of Lyapunov functions via assume-guarantee reasoning](http://arxiv.org/abs/2604.03017v1) | Matteo Capucci, David Jaz Myers | Assume-guarantee reasoning is a technique for compositional model checking in which system specifications are checked under certain assumptions on system parameters or inputs, and provide guarantees on observations of system state. We present a categorical framework for assume-guarantee reasoning for safety problems by viewing systems as lenses, following our earlier work on the compositionality of generalized Moore machines. Generalized Moore machines include ordinary Moore machines, partially observable Markov (decision) processes, and systems of parameterized ODEs (control systems); our framework gives assume-guarantee reasoning specially adapted to each of these cases. In particular, we give a novel formulation of assume-guarantee reasoning for (local) input-to-state stability ((L)ISS) Lyapunov functions on systems of parameterized ODEs.   Our framework is categorically natural and straightforwardly compositional. A flavor of generalized Moore machine is determined by a tangency: a fibration with a section. We show that symmetric monoidal loose right modules of assume-guarantee certified generalized Moore machines over symmetric monoidal double categories of certified wiring diagrams can be constructed 2-functorially from fibrations internal to the 2-category of tangencies. |
| 2026-04-03 | [FoE: Forest of Errors Makes the First Solution the Best in Large Reasoning Models](http://arxiv.org/abs/2604.02967v1) | Kehan Jiang, Haonan Dong et al. | Recent Large Reasoning Models (LRMs) like DeepSeek-R1 have demonstrated remarkable success in complex reasoning tasks, exhibiting human-like patterns in exploring multiple alternative solutions. Upon closer inspection, however, we uncover a surprising phenomenon: The First is The Best, where alternative solutions are not merely suboptimal but potentially detrimental. This observation challenges widely accepted test-time scaling laws, leading us to hypothesize that errors within the reasoning path scale concurrently with test time. Through comprehensive empirical analysis, we characterize errors as a forest-structured Forest of Errors (FoE) and conclude that FoE makes the First the Best, which is underpinned by rigorous theoretical analysis. Leveraging these insights, we propose RED, a self-guided efficient reasoning framework comprising two components: I) Refining First, which suppresses FoE growth in the first solution; and II) Discarding Subs, which prunes subsequent FoE via dual-consistency. Extensive experiments across five benchmarks and six backbone models demonstrate that RED outperforms eight competitive baselines, achieving performance gains of up to 19.0% while reducing token consumption by 37.7% ~ 70.4%. Moreover, comparative experiments on FoE metrics shed light on how RED achieves effectiveness. |
| 2026-04-03 | [act: Technical report](http://arxiv.org/abs/2604.02955v1) | Zoe Paraskevopoulou, Anja Petković Komel et al. | This technical report contains the formal definitions and metatheory for the act specification and verification language. It documents the syntax, the operational pointer semantics, the type system and the main metatheoretic results (type-safety). |
| 2026-04-03 | [Probably Approximately Correct (PAC) Guarantees for Data-Driven Reachability Analysis: A Theoretical and Empirical Comparison](http://arxiv.org/abs/2604.02953v1) | Elizabeth Dietrich, Hanna Krasowski et al. | Reachability analysis evaluates system safety, by identifying the set of states a system may evolve within over a finite time horizon. In contrast to model-based reachability analysis, data-driven reachability analysis estimates reachable sets and derives probabilistic guarantees directly from data. Several popular techniques for validating reachable sets -- conformal prediction, scenario optimization, and the holdout method -- admit similar Probably Approximately Correct (PAC) guarantees. We establish a formal connection between these PAC bounds and present an empirical case study on reachable sets to illustrate the computational and sample trade-offs associated with these methods. We argue that despite the formal relationship between these techniques, subtle differences arise in both the interpretation of guarantees and the parameterization. As a result, these methods are not generally interchangeable. We conclude with practical advice on the usage of these methods. |
| 2026-04-03 | [AgentHazard: A Benchmark for Evaluating Harmful Behavior in Computer-Use Agents](http://arxiv.org/abs/2604.02947v1) | Yunhao Feng, Yifan Ding et al. | Computer-use agents extend language models from text generation to persistent action over tools, files, and execution environments. Unlike chat systems, they maintain state across interactions and translate intermediate outputs into concrete actions. This creates a distinct safety challenge in that harmful behavior may emerge through sequences of individually plausible steps, including intermediate actions that appear locally acceptable but collectively lead to unauthorized actions. We present \textbf{AgentHazard}, a benchmark for evaluating harmful behavior in computer-use agents. AgentHazard contains \textbf{2,653} instances spanning diverse risk categories and attack strategies. Each instance pairs a harmful objective with a sequence of operational steps that are locally legitimate but jointly induce unsafe behavior. The benchmark evaluates whether agents can recognize and interrupt harm arising from accumulated context, repeated tool use, intermediate actions, and dependencies across steps. We evaluate AgentHazard on Claude Code, OpenClaw, and IFlow using mostly open or openly deployable models from the Qwen3, Kimi, GLM, and DeepSeek families. Our experimental results indicate that current systems remain highly vulnerable. In particular, when powered by Qwen3-Coder, Claude Code exhibits an attack success rate of \textbf{73.63\%}, suggesting that model alignment alone does not reliably guarantee the safety of autonomous agents. |
| 2026-04-03 | [PolyReal: A Benchmark for Real-World Polymer Science Workflows](http://arxiv.org/abs/2604.02934v1) | Wanhao Liu, Weida Wang et al. | Multimodal Large Language Models (MLLMs) excel in general domains but struggle with complex, real-world science. We posit that polymer science, an interdisciplinary field spanning chemistry, physics, biology, and engineering, is an ideal high-stakes testbed due to its diverse multimodal data. Yet, existing benchmarks related to polymer science largely overlook real-world workflows, limiting their practical utility and failing to systematically evaluate MLLMs across the full, practice-grounded lifecycle of experimentation. We introduce PolyReal, a novel multimodal benchmark grounded in real-world scientific practices to evaluate MLLMs on the full lifecycle of polymer experimentation. It covers five critical capabilities: (1) foundational knowledge application; (2) lab safety analysis; (3) experiment mechanism reasoning; (4) raw data extraction; and (5) performance & application exploration. Our evaluation of leading MLLMs on PolyReal reveals a capability imbalance. While models perform well on knowledge-intensive reasoning (e.g., Experiment Mechanism Reasoning), they drop sharply on practice-based tasks (e.g., Lab Safety Analysis and Raw Data Extraction). This exposes a severe gap between abstract scientific knowledge and its practical, context-dependent application, showing that these real-world tasks remain challenging for MLLMs. Thus, PolyReal helps address this evaluation gap and provides a practical benchmark for assessing AI systems in real-world scientific workflows. |
| 2026-04-03 | [Goal-Conditioned Neural ODEs with Guaranteed Safety and Stability for Learning-Based All-Pairs Motion Planning](http://arxiv.org/abs/2604.02821v1) | Dechuan Liu, Ruigang Wang et al. | This paper presents a learning-based approach for all-pairs motion planning, where the initial and goal states are allowed to be arbitrary points in a safe set. We construct smooth goal-conditioned neural ordinary differential equations (neural ODEs) via bi-Lipschitz diffeomorphisms. Theoretical results show that the proposed model can provide guarantees of global exponential stability and safety (safe set forward invariance) regardless of goal location. Moreover, explicit bounds on convergence rate, tracking error, and vector field magnitude are established. Our approach admits a tractable learning implementation using bi-Lipschitz neural networks and can incorporate demonstration data. We illustrate the effectiveness of the proposed method on a 2D corridor navigation task. |
| 2026-04-03 | [Learning Structured Robot Policies from Vision-Language Models via Synthetic Neuro-Symbolic Supervision](http://arxiv.org/abs/2604.02812v1) | Alessandro Adami, Tommaso Tubaldo et al. | Vision-language models (VLMs) have recently demonstrated strong capabilities in mapping multimodal observations to robot behaviors. However, most current approaches rely on end-to-end visuomotor policies that remain opaque and difficult to analyze, limiting their use in safety-critical robotic applications. In contrast, classical robotic systems often rely on structured policy representations that provide interpretability, modularity, and reactive execution. This work investigates how foundation models can be specialized to generate structured robot policies grounded in multimodal perception, bridging high-dimensional learning and symbolic control. We propose a neuro-symbolic approach in which a VLM synthesizes executable Behavior Tree policies from visual observations, natural language instructions, and structured system specifications. To enable scalable supervision without manual annotation, we introduce an automated pipeline that generates a synthetic multimodal dataset of domain-randomized scenes paired with instruction-policy examples produced by a foundation model. Real-world experiments on two robotic manipulators show that structured policies learned entirely from synthetic supervision transfer successfully to physical systems. The results indicate that foundation models can be adapted to produce interpretable and structured robot policies, providing an alternative to opaque end-to-end approaches for multimodal robot decision making. |
| 2026-04-03 | [PaveBench: A Versatile Benchmark for Pavement Distress Perception and Interactive Vision-Language Analysis](http://arxiv.org/abs/2604.02804v1) | Dexiang Li, Zhenning Che et al. | Pavement condition assessment is essential for road safety and maintenance. Existing research has made significant progress. However, most studies focus on conventional computer vision tasks such as classification, detection, and segmentation. In real-world applications, pavement inspection requires more than visual recognition. It also requires quantitative analysis, explanation, and interactive decision support. Current datasets are limited. They focus on unimodal perception. They lack support for multi-turn interaction and fact-grounded reasoning. They also do not connect perception with vision-language analysis. To address these limitations, we introduce PaveBench, a large-scale benchmark for pavement distress perception and interactive vision-language analysis on real-world highway inspection images. PaveBench supports four core tasks: classification, object detection, semantic segmentation, and vision-language question answering. It provides unified task definitions and evaluation protocols. On the visual side, PaveBench provides large-scale annotations and includes a curated hard-distractor subset for robustness evaluation. It contains a large collection of real-world pavement images. On the multimodal side, we introduce PaveVQA, a real-image question answering (QA) dataset that supports single-turn, multi-turn, and expert-corrected interactions. It covers recognition, localization, quantitative estimation, and maintenance reasoning. We evaluate several state-of-the-art methods and provide a detailed analysis. We also present a simple and effective agent-augmented visual question answering framework that integrates domain-specific models as tools alongside vision-language models. The dataset is available at: https://huggingface.co/datasets/MML-Group/PaveBench. |

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



