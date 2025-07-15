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
| 2025-07-14 | [REST: Stress Testing Large Reasoning Models by Asking Multiple Problems at Once](http://arxiv.org/abs/2507.10541v1) | Zhuoshi Pan, Qizhi Pei et al. | Recent Large Reasoning Models (LRMs) have achieved remarkable progress on task-specific benchmarks, yet their evaluation methods remain constrained by isolated problem-solving paradigms. Existing benchmarks predominantly assess single-question reasoning through sequential testing, resulting critical limitations: (1) vulnerability to data contamination and less challenging (e.g., DeepSeek-R1 achieves 97.0% on MATH500), forcing costly and perpetual creation of new questions with large human efforts, (2) failure to evaluate models under multi-context pressure, a key requirement for real-world deployment. To bridge this gap, we present REST (Reasoning Evaluation through Simultaneous Testing), a stress-testing framework that concurrently exposes LRMs to multiple problems simultaneously. Beyond basic reasoning, REST specifically evaluates several under-tested capabilities: contextual priority allocation, cross-problem interference resistance, and dynamic cognitive load management. Our evaluation reveals several striking findings: Even state-of-the-art (SOTA) models like DeepSeek-R1 exhibit substantial performance degradation under stress testing. Crucially, REST demonstrates stronger discriminative power than existing benchmarks, revealing pronounced performance differences among models that exhibit similar, near-ceiling performance under single-question evaluations. Some key mechanistic insights emerge from our analysis: (1) the "overthinking trap" is a critical factor contributing to the performance degradation; (2) the models trained with "long2short" technique preserve more accuracy of their single-problem performance under REST, outperforming standard-trained counterparts. These results establish REST as a cost-efficient, future-proof evaluation paradigm that better reflects real-world reasoning demands while reducing reliance on continuous human annotation. |
| 2025-07-14 | [Can You Detect the Difference?](http://arxiv.org/abs/2507.10475v1) | İsmail Tarım, Aytuğ Onan | The rapid advancement of large language models (LLMs) has raised concerns about reliably detecting AI-generated text. Stylometric metrics work well on autoregressive (AR) outputs, but their effectiveness on diffusion-based models is unknown. We present the first systematic comparison of diffusion-generated text (LLaDA) and AR-generated text (LLaMA) using 2 000 samples. Perplexity, burstiness, lexical diversity, readability, and BLEU/ROUGE scores show that LLaDA closely mimics human text in perplexity and burstiness, yielding high false-negative rates for AR-oriented detectors. LLaMA shows much lower perplexity but reduced lexical fidelity. Relying on any single metric fails to separate diffusion outputs from human writing. We highlight the need for diffusion-aware detectors and outline directions such as hybrid models, diffusion-specific stylometric signatures, and robust watermarking. |
| 2025-07-14 | [Toolsuite for Implementing Multiagent Systems Based on Communication Protocols](http://arxiv.org/abs/2507.10324v1) | Amit K. Chopra, Samuel H. Christie V et al. | Interaction-Oriented Programming (IOP) is an approach to building a multiagent system by modeling the interactions between its roles via a flexible interaction protocol and implementing agents to realize the interactions of the roles they play in the protocol.   In recent years, we have developed an extensive suite of software that enables multiagent system developers to apply IOP. These include tools for efficiently verifying protocols for properties such as liveness and safety and middleware that simplifies the implementation of agents. This paper presents some of that software suite. |
| 2025-07-14 | [Multi-Robot Cooperative Herding through Backstepping Control Barrier Functions](http://arxiv.org/abs/2507.10249v1) | Kang Li, Ming Li et al. | We propose a novel cooperative herding strategy through backstepping control barrier functions (CBFs), which coordinates multiple herders to herd a group of evaders safely towards a designated goal region. For the herding system with heterogeneous groups involving herders and evaders, the behavior of the evaders can only be influenced indirectly by the herders' motion, especially when the evaders follow an inverse dynamics model and respond solely to repulsive interactions from the herders. This indirect interaction mechanism inherently renders the overall system underactuated. To address this issue, we first construct separate CBFs for the dual objectives of goal reaching and collision avoidance, which ensure both herding completion and safety guarantees. Then, we reformulate the underactuated herding dynamics into a control-affine structure and employ a backstepping approach to recursively design control inputs for the hierarchical barrier functions, avoiding taking derivatives of the higher-order system. Finally, we present a cooperative herding strategy based on backstepping CBFs that allow herders to safely herd multiple evaders into the goal region. In addition, centralized and decentralized implementations of the proposed algorithm are developed, further enhancing its flexibility and applicability. Extensive simulations and real-world experiments validate the effectiveness and safety of the proposed strategy in multi-robot herding. |
| 2025-07-14 | [FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring](http://arxiv.org/abs/2507.10134v1) | Yousef Emami, Hao Zhou et al. | Unmanned Aerial Vehicles (UAVs) are vital for public safety, particularly in wildfire monitoring, where early detection minimizes environmental impact. In UAV-Assisted Wildfire Monitoring (UAWM) systems, joint optimization of sensor transmission scheduling and velocity is critical for minimizing Age of Information (AoI) from stale sensor data. Deep Reinforcement Learning (DRL) has been used for such optimization; however, its limitations such as low sampling efficiency, simulation-to-reality gaps, and complex training render it unsuitable for time-critical applications like wildfire monitoring. This paper introduces a new online Flight Resource Allocation scheme based on LLM-Enabled In-Context Learning (FRSICL) to jointly optimize the UAV's flight control and data collection schedule along the trajectory in real time, thereby asymptotically minimizing the average AoI across ground sensors. In contrast to DRL, FRSICL generates data collection schedules and controls velocity using natural language task descriptions and feedback from the environment, enabling dynamic decision-making without extensive retraining. Simulation results confirm the effectiveness of the proposed FRSICL compared to Proximal Policy Optimization (PPO) and Nearest-Neighbor baselines. |
| 2025-07-14 | [BlueGlass: A Framework for Composite AI Safety](http://arxiv.org/abs/2507.10106v1) | Harshal Nandigramwar, Syed Qutub et al. | As AI systems become increasingly capable and ubiquitous, ensuring the safety of these systems is critical. However, existing safety tools often target different aspects of model safety and cannot provide full assurance in isolation, highlighting a need for integrated and composite methodologies. This paper introduces BlueGlass, a framework designed to facilitate composite AI safety workflows by providing a unified infrastructure enabling the integration and composition of diverse safety tools that operate across model internals and outputs. Furthermore, to demonstrate the utility of this framework, we present three safety-oriented analyses on vision-language models for the task of object detection: (1) distributional evaluation, revealing performance trade-offs and potential failure modes across distributions; (2) probe-based analysis of layer dynamics highlighting shared hierarchical learning via phase transition; and (3) sparse autoencoders identifying interpretable concepts. More broadly, this work contributes foundational infrastructure and findings for building more robust and reliable AI systems. |
| 2025-07-14 | [Foundation Model Driven Robotics: A Comprehensive Review](http://arxiv.org/abs/2507.10087v1) | Muhammad Tayyab Khan, Ammar Waheed | The rapid emergence of foundation models, particularly Large Language Models (LLMs) and Vision-Language Models (VLMs), has introduced a transformative paradigm in robotics. These models offer powerful capabilities in semantic understanding, high-level reasoning, and cross-modal generalization, enabling significant advances in perception, planning, control, and human-robot interaction. This critical review provides a structured synthesis of recent developments, categorizing applications across simulation-driven design, open-world execution, sim-to-real transfer, and adaptable robotics. Unlike existing surveys that emphasize isolated capabilities, this work highlights integrated, system-level strategies and evaluates their practical feasibility in real-world environments. Key enabling trends such as procedural scene generation, policy generalization, and multimodal reasoning are discussed alongside core bottlenecks, including limited embodiment, lack of multimodal data, safety risks, and computational constraints. Through this lens, this paper identifies both the architectural strengths and critical limitations of foundation model-based robotics, highlighting open challenges in real-time operation, grounding, resilience, and trust. The review concludes with a roadmap for future research aimed at bridging semantic reasoning and physical intelligence through more robust, interpretable, and embodied models. |
| 2025-07-14 | [TGLD: A Trust-Aware Game-Theoretic Lane-Changing Decision Framework for Automated Vehicles in Heterogeneous Traffic](http://arxiv.org/abs/2507.10075v1) | Jie Pan, Tianyi Wang et al. | Automated vehicles (AVs) face a critical need to adopt socially compatible behaviors and cooperate effectively with human-driven vehicles (HVs) in heterogeneous traffic environment. However, most existing lane-changing frameworks overlook HVs' dynamic trust levels, limiting their ability to accurately predict human driver behaviors. To address this gap, this study proposes a trust-aware game-theoretic lane-changing decision (TGLD) framework. First, we formulate a multi-vehicle coalition game, incorporating fully cooperative interactions among AVs and partially cooperative behaviors from HVs informed by real-time trust evaluations. Second, we develop an online trust evaluation method to dynamically estimate HVs' trust levels during lane-changing interactions, guiding AVs to select context-appropriate cooperative maneuvers. Lastly, social compatibility objectives are considered by minimizing disruption to surrounding vehicles and enhancing the predictability of AV behaviors, thereby ensuring human-friendly and context-adaptive lane-changing strategies. A human-in-the-loop experiment conducted in a highway on-ramp merging scenario validates our TGLD approach. Results show that AVs can effectively adjust strategies according to different HVs' trust levels and driving styles. Moreover, incorporating a trust mechanism significantly improves lane-changing efficiency, maintains safety, and contributes to transparent and adaptive AV-HV interactions. |
| 2025-07-14 | [Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks](http://arxiv.org/abs/2507.10054v1) | Emir Bosnak, Sahand Moslemi et al. | Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests. |
| 2025-07-14 | [Automating SPARQL Query Translations between DBpedia and Wikidata](http://arxiv.org/abs/2507.10045v1) | Malte Christian Bartels, Debayan Banerjee et al. | This paper investigates whether state-of-the-art Large Language Models (LLMs) can automatically translate SPARQL between popular Knowledge Graph (KG) schemas. We focus on translations between the DBpedia and Wikidata KG, and later on DBLP and OpenAlex KG. This study addresses a notable gap in KG interoperability research by rigorously evaluating LLM performance on SPARQL-to-SPARQL translation. Two benchmarks are assembled, where the first align 100 DBpedia-Wikidata queries from QALD-9-Plus; the second contains 100 DBLP queries aligned to OpenAlex, testing generalizability beyond encyclopaedic KGs. Three open LLMs: Llama-3-8B, DeepSeek-R1-Distill-Llama-70B, and Mistral-Large-Instruct-2407 are selected based on their sizes and architectures and tested with zero-shot, few-shot, and two chain-of-thought variants. Outputs were compared with gold answers, and resulting errors were categorized. We find that the performance varies markedly across models and prompting strategies, and that translations for Wikidata to DBpedia work far better than translations for DBpedia to Wikidata. |
| 2025-07-14 | [Vision-Based Anti Unmanned Aerial Technology: Opportunities and Challenges](http://arxiv.org/abs/2507.10006v1) | Guanghai Ding, Yihua Ren et al. | With the rapid advancement of UAV technology and its extensive application in various fields such as military reconnaissance, environmental monitoring, and logistics, achieving efficient and accurate Anti-UAV tracking has become essential. The importance of Anti-UAV tracking is increasingly prominent, especially in scenarios such as public safety, border patrol, search and rescue, and agricultural monitoring, where operations in complex environments can provide enhanced security. Current mainstream Anti-UAV tracking technologies are primarily centered around computer vision techniques, particularly those that integrate multi-sensor data fusion with advanced detection and tracking algorithms. This paper first reviews the characteristics and current challenges of Anti-UAV detection and tracking technologies. Next, it investigates and compiles several publicly available datasets, providing accessible links to support researchers in efficiently addressing related challenges. Furthermore, the paper analyzes the major vision-based and vision-fusion-based Anti-UAV detection and tracking algorithms proposed in recent years. Finally, based on the above research, this paper outlines future research directions, aiming to provide valuable insights for advancing the field. |
| 2025-07-14 | [3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving](http://arxiv.org/abs/2507.09993v1) | Yixun Zhang, Lizhi Wang et al. | Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. While existing 2D and 3D physical attacks typically optimize texture, they often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module to preserve geometric fidelity, and a physical augmentation module to simulate complex physical scenarios, thus enhancing attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks. These results validate 3DGAA as a practical attack framework for evaluating the safety of perception systems in autonomous driving. |
| 2025-07-14 | [Tiny Reward Models](http://arxiv.org/abs/2507.09973v1) | Sarah Pan | Large decoder-based language models have become the dominant architecture for reward modeling in reinforcement learning from human feedback (RLHF). However, as reward models are increasingly deployed in test-time strategies, their inference costs become a growing concern. We present TinyRM, a family of small, bidirectional masked language models (MLMs) with as few as 400 million parameters, that rival the capabilities of models over 175 times larger on reasoning and safety preference modeling tasks. TinyRM combines FLAN-style prompting, Directional Low-Rank Adaptation (DoRA), and layer freezing to achieve strong performance on RewardBench, despite using significantly fewer resources. Our experiments suggest that small models benefit from domain-specific tuning strategies, particularly in reasoning, where lightweight finetuning methods are especially effective. While challenges remain in building generalist models and conversational preference modeling, our preliminary results highlight the promise of lightweight bidirectional architectures as efficient, scalable alternatives for preference modeling. |
| 2025-07-14 | [Mechanistic Interpretability of LoRA-Adapted Language Models for Nuclear Reactor Safety Applications](http://arxiv.org/abs/2507.09931v1) | Yoon Pyo Lee | The integration of Large Language Models (LLMs) into safety-critical domains, such as nuclear engineering, necessitates a deep understanding of their internal reasoning processes. This paper presents a novel methodology for interpreting how an LLM encodes and utilizes domain-specific knowledge, using a Boiling Water Reactor system as a case study. We adapted a general-purpose LLM (Gemma-3-1b-it) to the nuclear domain using a parameter-efficient fine-tuning technique known as Low-Rank Adaptation. By comparing the neuron activation patterns of the base model to those of the fine-tuned model, we identified a sparse set of neurons whose behavior was significantly altered during the adaptation process. To probe the causal role of these specialized neurons, we employed a neuron silencing technique. Our results demonstrate that while silencing most of these specialized neurons individually did not produce a statistically significant effect, deactivating the entire group collectively led to a statistically significant degradation in task performance. Qualitative analysis further revealed that silencing these neurons impaired the model's ability to generate detailed, contextually accurate technical information. This paper provides a concrete methodology for enhancing the transparency of an opaque black-box model, allowing domain expertise to be traced to verifiable neural circuits. This offers a pathway towards achieving nuclear-grade artificial intelligence (AI) assurance, addressing the verification and validation challenges mandated by nuclear regulatory frameworks (e.g., 10 CFR 50 Appendix B), which have limited AI deployment in safety-critical nuclear operations. |
| 2025-07-14 | [BeePL: Correct-by-compilation kernel extensions](http://arxiv.org/abs/2507.09883v1) | Swarn Priya, Frédéric Besson et al. | eBPF is a technology that allows developers to safely extend kernel functionality without modifying kernel source code or developing loadable kernel modules. Since the kernel governs critical system operations and enforces isolation boundaries between user space and privileged data, any mechanism that modifies its behavior must meet the highest standards of safety and correctness. To this end, the eBPF toolchain includes a verifier, which statically checks safety properties such as memory access validity, bounded loops, and type correctness before loading the program into the kernel. However, the existing verifier is both overly conservative in some cases-rejecting valid programs-and unsound in others, permitting unsafe behavior that violates the intended semantics of the kernel interface.   To address these challenges, we introduce BeePL, a domain-specific language for eBPF with a formally verified type system. The BeePL type system, along with the language design, statically enforces key safety properties such as type-correct memory access, safe pointer usage, absence of unbounded loops, and structured control flow. These guarantees are backed by formal type soundness proofs, ensuring that well-typed programs satisfy the safety invariants required by the eBPF execution environment. BeePL also proves that well-typed source programs meet critical eBPF-specific properties related to memory safety, termination, and control flow, enabling high-level reasoning prior to compilation. For properties not fully enforceable statically-such as dynamic bounds and undefined behavior-BeePL inserts semantics-preserving runtime checks during compilation. We develop a verified compilation strategy that extends CompCert to generate BPF bytecode from BeePL programs, establishing a principled foundation for an end-to-end verifiable toolchain for safe kernel extensions. |
| 2025-07-14 | [Intersection of Reinforcement Learning and Bayesian Optimization for Intelligent Control of Industrial Processes: A Safe MPC-based DPG using Multi-Objective BO](http://arxiv.org/abs/2507.09864v1) | Hossein Nejatbakhsh Esfahani, Javad Mohammadpour Velni | Model Predictive Control (MPC)-based Reinforcement Learning (RL) offers a structured and interpretable alternative to Deep Neural Network (DNN)-based RL methods, with lower computational complexity and greater transparency. However, standard MPC-RL approaches often suffer from slow convergence, suboptimal policy learning due to limited parameterization, and safety issues during online adaptation. To address these challenges, we propose a novel framework that integrates MPC-RL with Multi-Objective Bayesian Optimization (MOBO). The proposed MPC-RL-MOBO utilizes noisy evaluations of the RL stage cost and its gradient, estimated via a Compatible Deterministic Policy Gradient (CDPG) approach, and incorporates them into a MOBO algorithm using the Expected Hypervolume Improvement (EHVI) acquisition function. This fusion enables efficient and safe tuning of the MPC parameters to achieve improved closed-loop performance, even under model imperfections. A numerical example demonstrates the effectiveness of the proposed approach in achieving sample-efficient, stable, and high-performance learning for control systems. |
| 2025-07-14 | [Customize Harmonic Potential Fields via Hybrid Optimization over Homotopic Paths](http://arxiv.org/abs/2507.09858v1) | Shuaikang Wang, Tiecheng Guo et al. | Safe navigation within a workspace is a fundamental skill for autonomous robots to accomplish more complex tasks. Harmonic potentials are artificial potential fields that are analytical, globally convergent and provably free of local minima. Thus, it has been widely used for generating safe and reliable robot navigation control policies. However, most existing methods do not allow customization of the harmonic potential fields nor the resulting paths, particularly regarding their topological properties. In this paper, we propose a novel method that automatically finds homotopy classes of paths that can be generated by valid harmonic potential fields. The considered complex workspaces can be as general as forest worlds consisting of numerous overlapping star-obstacles. The method is based on a hybrid optimization algorithm that searches over homotopy classes, selects the structure of each tree-of-stars within the forest, and optimizes over the continuous weight parameters for each purged tree via the projected gradient descent. The key insight is to transform the forest world to the unbounded point world via proper diffeomorphic transformations. It not only facilitates a simpler design of the multi-directional D-signature between non-homotopic paths, but also retain the safety and convergence properties. Extensive simulations and hardware experiments are conducted for non-trivial scenarios, where the navigation potentials are customized for desired homotopic properties. Project page: https://shuaikang-wang.github.io/CustFields. |
| 2025-07-14 | [Is Human-Written Data Enough? The Challenge of Teaching Reasoning to LLMs Without RL or Distillation](http://arxiv.org/abs/2507.09850v1) | Wei Du, Branislav Kisacanin et al. | Reasoning-capable language models achieve state-of-the-art performance in diverse complex tasks by generating long, explicit Chain-of-Thought (CoT) traces. While recent works show that base models can acquire such reasoning traces via reinforcement learning or distillation from stronger models like DeepSeek-R1, previous works demonstrate that even short CoT prompting without fine-tuning is able to improve reasoning. We ask whether long CoT can be induced in a base model using only prompting or minimal tuning. Using just 20 long CoT examples from the reasoning model \texttt{QwQ-32B-Preview}, we lightly fine-tune the base model \texttt{Qwen2.5-32B}. The resulting model outperforms the much larger \texttt{Qwen2.5-Math-72B-Instruct}, showing that a handful of high-quality examples can unlock strong reasoning capabilities. We further explore using CoT data from non-reasoning models and human annotators, enhanced with prompt engineering, multi-pass editing, and structural guidance. However, neither matches the performance of reasoning model traces, suggesting that certain latent qualities of expert CoT are difficult to replicate. We analyze key properties of reasoning data, such as problem difficulty, diversity, and answer length, that influence reasoning distillation. While challenges remain, we are optimistic that carefully curated human-written CoT, even in small quantities, can activate reasoning behaviors in base models. We release our human-authored dataset across refinement stages and invite further investigation into what makes small-scale reasoning supervision so effective. |
| 2025-07-14 | [Remote Safety Monitoring: Significance-Aware Status Updating for Situational Awareness](http://arxiv.org/abs/2507.09833v1) | Tasmeen Zaman Ornee, Md Kamran Chowdhury Shisher et al. | In this study, we consider a problem of remote safety monitoring, where a monitor pulls status updates from multiple sensors monitoring several safety-critical situations. Based on the received updates, multiple estimators determine the current safety-critical situations. Due to transmission errors and limited channel resources, the received status updates may not be fresh, resulting in the possibility of misunderstanding the current safety situation. In particular, if a dangerous situation is misinterpreted as safe, the safety risk is high. We study the joint design of transmission scheduling and estimation for multi-sensor, multi-channel remote safety monitoring, aiming to minimize the loss due to the unawareness of potential danger. We show that the joint design of transmission scheduling and estimation can be reduced to a sequential optimization of estimation and scheduling. The scheduling problem can be formulated as a Restless Multi-armed Bandit (RMAB) , for which it is difficult to establish indexability. We propose a low-complexity Maximum Gain First (MGF) policy and prove it is asymptotically optimal as the numbers of sources and channels scale up proportionally, without requiring the indexability condition. We also provide an information-theoretic interpretation of the transmission scheduling problem. Numerical results show that our estimation and scheduling policies achieves higher performance gain over periodic updating, randomized policy, and Maximum Age First (MAF) policy. |
| 2025-07-13 | [Measuring What Matters: A Framework for Evaluating Safety Risks in Real-World LLM Applications](http://arxiv.org/abs/2507.09820v1) | Jia Yi Goh, Shaun Khoo et al. | Most safety testing efforts for large language models (LLMs) today focus on evaluating foundation models. However, there is a growing need to evaluate safety at the application level, as components such as system prompts, retrieval pipelines, and guardrails introduce additional factors that significantly influence the overall safety of LLM applications. In this paper, we introduce a practical framework for evaluating application-level safety in LLM systems, validated through real-world deployment across multiple use cases within our organization. The framework consists of two parts: (1) principles for developing customized safety risk taxonomies, and (2) practices for evaluating safety risks in LLM applications. We illustrate how the proposed framework was applied in our internal pilot, providing a reference point for organizations seeking to scale their safety testing efforts. This work aims to bridge the gap between theoretical concepts in AI safety and the operational realities of safeguarding LLM applications in practice, offering actionable guidance for safe and scalable deployment. |

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



