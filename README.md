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
| 2026-01-12 | [SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations](http://arxiv.org/abs/2601.07835v1) | Mohammed Himayath Ali, Mohammed Aqib Abdullah et al. | Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains. |
| 2026-01-12 | [Video Generation Models in Robotics - Applications, Research Challenges, Future Directions](http://arxiv.org/abs/2601.07823v1) | Zhiting Mei, Tenny Yin et al. | Video generation models have emerged as high-fidelity models of the physical world, capable of synthesizing high-quality videos capturing fine-grained interactions between agents and their environments conditioned on multi-modal user inputs. Their impressive capabilities address many of the long-standing challenges faced by physics-based simulators, driving broad adoption in many problem domains, e.g., robotics. For example, video models enable photorealistic, physically consistent deformable-body simulation without making prohibitive simplifying assumptions, which is a major bottleneck in physics-based simulation. Moreover, video models can serve as foundation world models that capture the dynamics of the world in a fine-grained and expressive way. They thus overcome the limited expressiveness of language-only abstractions in describing intricate physical interactions. In this survey, we provide a review of video models and their applications as embodied world models in robotics, encompassing cost-effective data generation and action prediction in imitation learning, dynamics and rewards modeling in reinforcement learning, visual planning, and policy evaluation. Further, we highlight important challenges hindering the trustworthy integration of video models in robotics, which include poor instruction following, hallucinations such as violations of physics, and unsafe content generation, in addition to fundamental limitations such as significant data curation, training, and inference costs. We present potential future directions to address these open research challenges to motivate research and ultimately facilitate broader applications, especially in safety-critical settings. |
| 2026-01-12 | [Failure-Aware RL: Reliable Offline-to-Online Reinforcement Learning with Self-Recovery for Real-World Manipulation](http://arxiv.org/abs/2601.07821v1) | Huanyu Li, Kun Lei et al. | Post-training algorithms based on deep reinforcement learning can push the limits of robotic models for specific objectives, such as generalizability, accuracy, and robustness. However, Intervention-requiring Failures (IR Failures) (e.g., a robot spilling water or breaking fragile glass) during real-world exploration happen inevitably, hindering the practical deployment of such a paradigm. To tackle this, we introduce Failure-Aware Offline-to-Online Reinforcement Learning (FARL), a new paradigm minimizing failures during real-world reinforcement learning. We create FailureBench, a benchmark that incorporates common failure scenarios requiring human intervention, and propose an algorithm that integrates a world-model-based safety critic and a recovery policy trained offline to prevent failures during online exploration. Extensive simulation and real-world experiments demonstrate the effectiveness of FARL in significantly reducing IR Failures while improving performance and generalization during online reinforcement learning post-training. FARL reduces IR Failures by 73.1% while elevating performance by 11.3% on average during real-world RL post-training. Videos and code are available at https://failure-aware-rl.github.io. |
| 2026-01-12 | [Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification](http://arxiv.org/abs/2601.07790v1) | Yahya Masri, Emily Ma et al. | System logs are crucial for monitoring and diagnosing modern computing infrastructure, but their scale and complexity require reliable and efficient automated interpretation. Since severity levels are predefined metadata in system log messages, having a model merely classify them offers limited standalone practical value, revealing little about its underlying ability to interpret system logs. We argue that severity classification is more informative when treated as a benchmark for probing runtime log comprehension rather than as an end task. Using real-world journalctl data from Linux production servers, we evaluate nine small language models (SLMs) and small reasoning language models (SRLMs) under zero-shot, few-shot, and retrieval-augmented generation (RAG) prompting. The results reveal strong stratification. Qwen3-4B achieves the highest accuracy at 95.64% with RAG, while Gemma3-1B improves from 20.25% under few-shot prompting to 85.28% with RAG. Notably, the tiny Qwen3-0.6B reaches 88.12% accuracy despite weak performance without retrieval. In contrast, several SRLMs, including Qwen3-1.7B and DeepSeek-R1-Distill-Qwen-1.5B, degrade substantially when paired with RAG. Efficiency measurements further separate models: most Gemma and Llama variants complete inference in under 1.2 seconds per log, whereas Phi-4-Mini-Reasoning exceeds 228 seconds per log while achieving <10% accuracy. These findings suggest that (1) architectural design, (2) training objectives, and (3) the ability to integrate retrieved context under strict output constraints jointly determine performance. By emphasizing small, deployable models, this benchmark aligns with real-time requirements of digital twin (DT) systems and shows that severity classification serves as a lens for evaluating model competence and real-time deployability, with implications for root cause analysis (RCA) and broader DT integration. |
| 2026-01-12 | [Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids](http://arxiv.org/abs/2601.07718v1) | Shaoting Zhu, Ziwen Zhuang et al. | Achieving robust humanoid hiking in complex, unstructured environments requires transitioning from reactive proprioception to proactive perception. However, integrating exteroception remains a significant challenge: mapping-based methods suffer from state estimation drift; for instance, LiDAR-based methods do not handle torso jitter well. Existing end-to-end approaches often struggle with scalability and training complexity; specifically, some previous works using virtual obstacles are implemented case-by-case. In this work, we present \textit{Hiking in the Wild}, a scalable, end-to-end parkour perceptive framework designed for robust humanoid hiking. To ensure safety and training stability, we introduce two key mechanisms: a foothold safety mechanism combining scalable \textit{Terrain Edge Detection} with \textit{Foot Volume Points} to prevent catastrophic slippage on edges, and a \textit{Flat Patch Sampling} strategy that mitigates reward hacking by generating feasible navigation targets. Our approach utilizes a single-stage reinforcement learning scheme, mapping raw depth inputs and proprioception directly to joint actions, without relying on external state estimation. Extensive field experiments on a full-size humanoid demonstrate that our policy enables robust traversal of complex terrains at speeds up to 2.5 m/s. The training and deployment code is open-sourced to facilitate reproducible research and deployment on real robots with minimal hardware modifications. |
| 2026-01-12 | [Safe Navigation under Uncertain Obstacle Dynamics using Control Barrier Functions and Constrained Convex Generators](http://arxiv.org/abs/2601.07715v1) | Hugo Matias, Daniel Silvestre | This paper presents a sampled-data framework for the safe navigation of controlled agents in environments cluttered with obstacles governed by uncertain linear dynamics. Collision-free motion is achieved by combining Control Barrier Function (CBF)-based safety filtering with set-valued state estimation using Constrained Convex Generators (CCGs). At each sampling time, a CCG estimate of each obstacle is obtained using a finite-horizon guaranteed estimation scheme and propagated over the sampling interval to obtain a CCG-valued flow that describes the estimated obstacle evolution. However, since CCGs are defined indirectly - as an affine transformation of a generator set subject to equality constraints, rather than as a sublevel set of a scalar function - converting the estimated obstacle flows into CBFs is a nontrivial task. One of the main contributions of this paper is a procedure to perform this conversion, ultimately yielding a CBF via a convex optimization problem whose validity is established by the Implicit Function Theorem. The resulting obstacle-specific CBFs are then merged into a single CBF that is used to design a safe controller through the standard Quadratic Program (QP)-based approach. Since CCGs support Minkowski sums, the proposed framework also naturally handles rigid-body agents and generalizes existing CBF-based rigid-body navigation designs to arbitrary agent and obstacle geometries. While the main contribution is general, the paper primarily focuses on agents with first-order control-affine dynamics and second-order strict-feedback dynamics. Simulation examples demonstrate the effectiveness of the proposed method. |
| 2026-01-12 | [Towards Automating Blockchain Consensus Verification with IsabeLLM](http://arxiv.org/abs/2601.07654v1) | Elliot Jones, William Knottenbelt | Consensus protocols are crucial for a blockchain system as they are what allow agreement between the system's nodes in a potentially adversarial environment. For this reason, it is paramount to ensure their correct design and implementation to prevent such adversaries from carrying out malicious behaviour. Formal verification allows us to ensure the correctness of such protocols, but requires high levels of effort and expertise to carry out and thus is often omitted in the development process. In this paper, we present IsabeLLM, a tool that integrates the proof assistant Isabelle with a Large Language Model to assist and automate proofs. We demonstrate the effectiveness of IsabeLLM by using it to develop a novel model of Bitcoin's Proof of Work consensus protocol and verify its correctness. We use the DeepSeek R1 API for this demonstration and found that we were able to generate correct proofs for each of the non-trivial lemmas present in the verification. |
| 2026-01-12 | [OODEval: Evaluating Large Language Models on Object-Oriented Design](http://arxiv.org/abs/2601.07602v1) | Bingxu Xiao, Yunwei Dong et al. | Recent advances in large language models (LLMs) have driven extensive evaluations in software engineering. however, most prior work concentrates on code-level tasks, leaving software design capabilities underexplored. To fill this gap, we conduct a comprehensive empirical study evaluating 29 LLMs on object-oriented design (OOD) tasks. Owing to the lack of standardized benchmarks and metrics, we introduce OODEval, a manually constructed benchmark comprising 50 OOD tasks of varying difficulty, and OODEval-Human, the first human-rated OOD benchmark, which includes 940 undergraduate-submitted class diagrams evaluated by instructors. We further propose CLUE (Class Likeness Unified Evaluation), a unified metric set that assesses both global correctness and fine-grained design quality in class diagram generation. Using these benchmarks and metrics, we investigate five research questions: overall correctness, comparison with humans, model dimension analysis, task feature analysis, and bad case analysis. The results indicate that while LLMs achieve high syntactic accuracy, they exhibit substantial semantic deficiencies, particularly in method and relationship generation. Among the evaluated models, Qwen3-Coder-30B achieves the best overall performance, rivaling DeepSeek-R1 and GPT-4o, while Gemma3-4B-IT outperforms GPT-4o-Mini despite its smaller parameter scale. Although top-performing LLMs nearly match the average performance of undergraduates, they remain significantly below the level of the best human designers. Further analysis shows that parameter scale, code specialization, and instruction tuning strongly influence performance, whereas increased design complexity and lower requirement readability degrade it. Bad case analysis reveals common failure modes, including keyword misuse, missing classes or relationships, and omitted methods. |
| 2026-01-12 | [Peformance Isolation for Inference Processes in Edge GPU Systems](http://arxiv.org/abs/2601.07600v1) | Juan José Martín, José Flich et al. | This work analyzes the main isolation mechanisms available in modern NVIDIA GPUs: MPS, MIG, and the recent Green Contexts, to ensure predictable inference time in safety-critical applications using deep learning models. The experimental methodology includes performance tests, evaluation of partitioning impact, and analysis of temporal isolation between processes, considering both the NVIDIA A100 and Jetson Orin platforms. It is observed that MIG provides a high level of isolation. At the same time, Green Contexts represent a promising alternative for edge devices by enabling fine-grained SM allocation with low overhead, albeit without memory isolation. The study also identifies current limitations and outlines potential research directions to improve temporal predictability in shared GPUs. |
| 2026-01-12 | [GRPO with State Mutations: Improving LLM-Based Hardware Test Plan Generation](http://arxiv.org/abs/2601.07593v1) | Dimple Vijay Kochar, Nathaniel Pinckney et al. | RTL design often relies heavily on ad-hoc testbench creation early in the design cycle. While large language models (LLMs) show promise for RTL code generation, their ability to reason about hardware specifications and generate targeted test plans remains largely unexplored. We present the first systematic study of LLM reasoning capabilities for RTL verification stimuli generation, establishing a two-stage framework that decomposes test plan generation from testbench execution. Our benchmark reveals that state-of-the-art models, including DeepSeek-R1 and Claude-4.0-Sonnet, achieve only 15.7-21.7% success rates on generating stimuli that pass golden RTL designs. To improve LLM generated stimuli, we develop a comprehensive training methodology combining supervised fine-tuning with a novel reinforcement learning approach, GRPO with State Mutation (GRPO-SMu), which enhances exploration by varying input mutations. Our approach leverages a tree-based branching mutation strategy to construct training data comprising equivalent and mutated trees, moving beyond linear mutation approaches to provide rich learning signals. Training on this curated dataset, our 7B parameter model achieves a 33.3% golden test pass rate and a 13.9% mutation detection rate, representing a 17.6% absolute improvement over baseline and outperforming much larger general-purpose models. These results demonstrate that specialized training methodologies can significantly enhance LLM reasoning capabilities for hardware verification tasks, establishing a foundation for automated sub-unit testing in semiconductor design workflows. |
| 2026-01-12 | [FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments](http://arxiv.org/abs/2601.07558v1) | Chen Feng, Guiyong Zheng et al. | Autonomous 3D scanning of open-world target structures via drones remains challenging despite broad applications. Existing paradigms rely on restrictive assumptions or effortful human priors, limiting practicality, efficiency, and adaptability. Recent foundation models (FMs) offer great potential to bridge this gap. This paper investigates a critical research problem: What system architecture can effectively integrate FM knowledge for this task? We answer it with FlyCo, a principled FM-empowered perception-prediction-planning loop enabling fully autonomous, prompt-driven 3D target scanning in diverse unknown open-world environments. FlyCo directly translates low-effort human prompts (text, visual annotations) into precise adaptive scanning flights via three coordinated stages: (1) perception fuses streaming sensor data with vision-language FMs for robust target grounding and tracking; (2) prediction distills FM knowledge and combines multi-modal cues to infer the partially observed target's complete geometry; (3) planning leverages predictive foresight to generate efficient and safe paths with comprehensive target coverage. Building on this, we further design key components to boost open-world target grounding efficiency and robustness, enhance prediction quality in terms of shape accuracy, zero-shot generalization, and temporal stability, and balance long-horizon flight efficiency with real-time computability and online collision avoidance. Extensive challenging real-world and simulation experiments show FlyCo delivers precise scene understanding, high efficiency, and real-time safety, outperforming existing paradigms with lower human effort and verifying the proposed architecture's practicality. Comprehensive ablations validate each component's contribution. FlyCo also serves as a flexible, extensible blueprint, readily leveraging future FM and robotics advances. Code will be released. |
| 2026-01-12 | [Software-Hardware Co-optimization for Modular E2E AV Paradigm: A Unified Framework of Optimization Approaches, Simulation Environment and Evaluation Metrics](http://arxiv.org/abs/2601.07393v1) | Chengzhi Ji, Xingfeng Li et al. | Modular end-to-end (ME2E) autonomous driving paradigms combine modular interpretability with global optimization capability and have demonstrated strong performance. However, existing studies mainly focus on accuracy improvement, while critical system-level factors such as inference latency and energy consumption are often overlooked, resulting in increasingly complex model designs that hinder practical deployment. Prior efforts on model compression and acceleration typically optimize either the software or hardware side in isolation. Software-only optimization cannot fundamentally remove intermediate tensor access and operator scheduling overheads, whereas hardware-only optimization is constrained by model structure and precision. As a result, the real-world benefits of such optimizations are often limited. To address these challenges, this paper proposes a reusable software and hardware co-optimization and closed-loop evaluation framework for ME2E autonomous driving inference. The framework jointly integrates software-level model optimization with hardware-level computation optimization under a unified system-level objective. In addition, a multidimensional evaluation metric is introduced to assess system performance by jointly considering safety, comfort, efficiency, latency, and energy, enabling quantitative comparison of different optimization strategies. Experiments across multiple ME2E autonomous driving stacks show that the proposed framework preserves baseline-level driving performance while significantly reducing inference latency and energy consumption, achieving substantial overall system-level improvements. These results demonstrate that the proposed framework provides practical and actionable guidance for efficient deployment of ME2E autonomous driving systems. |
| 2026-01-12 | [Stochastic CHAOS: Why Deterministic Inference Kills, and Distributional Variability Is the Heartbeat of Artifical Cognition](http://arxiv.org/abs/2601.07239v1) | Tanmay Joshi, Shourya Aggarwal et al. | Deterministic inference is a comforting ideal in classical software: the same program on the same input should always produce the same output. As large language models move into real-world deployment, this ideal has been imported wholesale into inference stacks. Recent work from the Thinking Machines Lab has presented a detailed analysis of nondeterminism in LLM inference, showing how batch-invariant kernels and deterministic attention can enforce bitwise-identical outputs, positioning deterministic inference as a prerequisite for reproducibility and enterprise reliability.   In this paper, we take the opposite stance. We argue that, for LLMs, deterministic inference kills. It kills the ability to model uncertainty, suppresses emergent abilities, collapses reasoning into a single brittle path, and weakens safety alignment by hiding tail risks. LLMs implement conditional distributions over outputs, not fixed functions. Collapsing these distributions to a single canonical completion may appear reassuring, but it systematically conceals properties central to artificial cognition. We instead advocate Stochastic CHAOS, treating distributional variability as a signal to be measured and controlled.   Empirically, we show that deterministic inference is systematically misleading. Single-sample deterministic evaluation underestimates both capability and fragility, masking failure probability under paraphrases and noise. Phase-like transitions associated with emergent abilities disappear under greedy decoding. Multi-path reasoning degrades when forced onto deterministic backbones, reducing accuracy and diagnostic insight. Finally, deterministic evaluation underestimates safety risk by hiding rare but dangerous behaviors that appear only under multi-sample evaluation. |
| 2026-01-12 | [Safeguarding LLM Fine-tuning via Push-Pull Distributional Alignment](http://arxiv.org/abs/2601.07200v1) | Haozhong Wang, Zhuo Li et al. | The inherent safety alignment of Large Language Models (LLMs) is prone to erosion during fine-tuning, even when using seemingly innocuous datasets. While existing defenses attempt to mitigate this via data selection, they typically rely on heuristic, instance-level assessments that neglect the global geometry of the data distribution and fail to explicitly repel harmful patterns. To address this, we introduce Safety Optimal Transport (SOT), a novel framework that reframes safe fine-tuning from an instance-level filtering challenge to a distribution-level alignment task grounded in Optimal Transport (OT). At its core is a dual-reference ``push-pull'' weight-learning mechanism: SOT optimizes sample importance by actively pulling the downstream distribution towards a trusted safe anchor while simultaneously pushing it away from a general harmful reference. This establishes a robust geometric safety boundary that effectively purifies the training data. Extensive experiments across diverse model families and domains demonstrate that SOT significantly enhances model safety while maintaining competitive downstream performance, achieving a superior safety-utility trade-off compared to baselines. |
| 2026-01-12 | [PROTEA: Securing Robot Task Planning and Execution](http://arxiv.org/abs/2601.07186v1) | Zainab Altaweel, Mohaiminul Al Nahian et al. | Robots need task planning methods to generate action sequences for complex tasks. Recent work on adversarial attacks has revealed significant vulnerabilities in existing robot task planners, especially those built on foundation models. In this paper, we aim to address these security challenges by introducing PROTEA, an LLM-as-a-Judge defense mechanism, to evaluate the security of task plans. PROTEA is developed to address the dimensionality and history challenges in plan safety assessment. We used different LLMs to implement multiple versions of PROTEA for comparison purposes. For systemic evaluations, we created a dataset containing both benign and malicious task plans, where the harmful behaviors were injected at varying levels of stealthiness. Our results provide actionable insights for robotic system practitioners seeking to enhance robustness and security of their task planning systems. Details, dataset and demos are provided: https://protea-secure.github.io/PROTEA/ |
| 2026-01-12 | [ShowUI-Aloha: Human-Taught GUI Agent](http://arxiv.org/abs/2601.07181v1) | Yichun Zhang, Xiangwu Guo et al. | Graphical User Interfaces (GUIs) are central to human-computer interaction, yet automating complex GUI tasks remains a major challenge for autonomous agents, largely due to a lack of scalable, high-quality training data. While recordings of human demonstrations offer a rich data source, they are typically long, unstructured, and lack annotations, making them difficult for agents to learn from.To address this, we introduce ShowUI-Aloha, a comprehensive pipeline that transforms unstructured, in-the-wild human screen recordings from desktop environments into structured, actionable tasks. Our framework includes four key components: A recorder that captures screen video along with precise user interactions like mouse clicks, keystrokes, and scrolls. A learner that semantically interprets these raw interactions and the surrounding visual context, translating them into descriptive natural language captions. A planner that reads the parsed demonstrations, maintains task states, and dynamically formulates the next high-level action plan based on contextual reasoning. An executor that faithfully carries out these action plans at the OS level, performing precise clicks, drags, text inputs, and window operations with safety checks and real-time feedback. Together, these components provide a scalable solution for collecting and parsing real-world human data, demonstrating a viable path toward building general-purpose GUI agents that can learn effectively from simply observing humans. |
| 2026-01-12 | [Safe-FedLLM: Delving into the Safety of Federated Large Language Models](http://arxiv.org/abs/2601.07177v1) | Mingxiang Tao, Yu Tian et al. | Federated learning (FL) addresses data privacy and silo issues in large language models (LLMs). Most prior work focuses on improving the training efficiency of federated LLMs. However, security in open environments is overlooked, particularly defenses against malicious clients. To investigate the safety of LLMs during FL, we conduct preliminary experiments to analyze potential attack surfaces and defensible characteristics from the perspective of Low-Rank Adaptation (LoRA) weights. We find two key properties of FL: 1) LLMs are vulnerable to attacks from malicious clients in FL, and 2) LoRA weights exhibit distinct behavioral patterns that can be filtered through simple classifiers. Based on these properties, we propose Safe-FedLLM, a probe-based defense framework for federated LLMs, constructing defenses across three dimensions: Step-Level, Client-Level, and Shadow-Level. The core concept of Safe-FedLLM is to perform probe-based discrimination on the LoRA weights locally trained by each client during FL, treating them as high-dimensional behavioral features and using lightweight classification models to determine whether they possess malicious attributes. Extensive experiments demonstrate that Safe-FedLLM effectively enhances the defense capability of federated LLMs without compromising performance on benign data. Notably, our method effectively suppresses malicious data impact without significant impact on training speed, and remains effective even with many malicious clients. Our code is available at: https://github.com/dmqx/Safe-FedLLM. |
| 2026-01-12 | [Test-time Adaptive Hierarchical Co-enhanced Denoising Network for Reliable Multimodal Classification](http://arxiv.org/abs/2601.07163v1) | Shu Shen, C. L. Philip Chen et al. | Reliable learning on low-quality multimodal data is a widely concerning issue, especially in safety-critical applications. However, multimodal noise poses a major challenge in this domain and leads existing methods to suffer from two key limitations. First, they struggle to reliably remove heterogeneous data noise, hindering robust multimodal representation learning. Second, they exhibit limited adaptability and generalization when encountering previously unseen noise. To address these issues, we propose Test-time Adaptive Hierarchical Co-enhanced Denoising Network (TAHCD). On one hand, TAHCD introduces the Adaptive Stable Subspace Alignment and Sample-Adaptive Confidence Alignment to reliably remove heterogeneous noise. They account for noise at both global and instance levels and enable jointly removal of modality-specific and cross-modality noise, achieving robust learning. On the other hand, TAHCD introduces test-time cooperative enhancement, which adaptively updates the model in response to input noise in a label-free manner, improving adaptability and generalization. This is achieved by collaboratively enhancing the joint removal process of modality-specific and cross-modality noise across global and instance levels according to sample noise. Experiments on multiple benchmarks demonstrate that the proposed method achieves superior classification performance, robustness, and generalization compared with state-of-the-art reliable multimodal learning approaches. |
| 2026-01-12 | [MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models](http://arxiv.org/abs/2601.07141v1) | Xi Ye, Yiwen Liu et al. | Text-to-image (T2I) models have raised increasing safety concerns due to their capacity to generate NSFW and other banned objects. To mitigate these risks, safety filters and concept removal techniques have been introduced to block inappropriate prompts or erase sensitive concepts from the models. However, all the existing defense methods are not well prepared to handle diverse adversarial prompts. In this work, we introduce MacPrompt, a novel black-box and cross-lingual attack that reveals previously overlooked vulnerabilities in T2I safety mechanisms. Unlike existing attacks that rely on synonym substitution or prompt obfuscation, MacPrompt constructs macaronic adversarial prompts by performing cross-lingual character-level recombination of harmful terms, enabling fine-grained control over both semantics and appearance. By leveraging this design, MacPrompt crafts prompts with high semantic similarity to the original harmful inputs (up to 0.96) while bypassing major safety filters (up to 100%). More critically, it achieves attack success rates as high as 92% for sex-related content and 90% for violence, effectively breaking even state-of-the-art concept removal defenses. These results underscore the pressing need to reassess the robustness of existing T2I safety mechanisms against linguistically diverse and fine-grained adversarial strategies. |
| 2026-01-12 | [Cyclic Modulation Control of Multi-Conflict Connected Automated Traffic](http://arxiv.org/abs/2601.07114v1) | Fan Pu, Zihao Li et al. | Multi-conflict traffic is ubiquitous. Connected Automated Vehicles (CAVs) offer unprecedented opportunities to enhance safety, reduce emissions, and increase throughput through precise coordination and automation. However, existing CAV strategies remain confined to specialized scenarios, such as highway on-ramp merging or single-lane roundabouts, and traditional traffic signals sacrifice efficiency for safety via rigid phasing and all-red intervals. In this paper, we present Cyclic Modulation Control of Multi-Conflict Connected Automated Traffic (CMAT), a unified, geometry-agnostic framework that embeds each conflict point into a repeating sequence of "micro-phases". Vehicles dynamically form platoons with demand-responsive sizes and negotiate time slots for occupying conflict points, enabling collision-free traversal and high intersection utilization. CMAT aims to minimize delay, guarantee safety, and accommodate arbitrary merging, diverging, and crossing patterns without manual retuning. We formalize CMAT as a mixed-integer linear programming model constructed on a directed graph abstracted from the physical intersection layout. The performance of CMAT is evaluated across a suite of multi-conflict tests, including simple two-way crossings, four-leg intersections, complex connected intersections. The results demonstrate substantial reductions in delay and significant throughput improvements compared with state-of-the-art CAV coordination methods and traditional signal timing strategies. |

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



