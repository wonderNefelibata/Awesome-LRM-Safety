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
| 2025-08-06 | [From MAS to MARS: Coordination Failures and Reasoning Trade-offs in Hierarchical Multi-Agent Robotic Systems within a Healthcare Scenario](http://arxiv.org/abs/2508.04691v1) | Yuanchen Bai, Zijian Ding et al. | Multi-agent robotic systems (MARS) build upon multi-agent systems by integrating physical and task-related constraints, increasing the complexity of action execution and agent coordination. However, despite the availability of advanced multi-agent frameworks, their real-world deployment on robots remains limited, hindering the advancement of MARS research in practice. To bridge this gap, we conducted two studies to investigate performance trade-offs of hierarchical multi-agent frameworks in a simulated real-world multi-robot healthcare scenario. In Study 1, using CrewAI, we iteratively refine the system's knowledge base, to systematically identify and categorize coordination failures (e.g., tool access violations, lack of timely handling of failure reports) not resolvable by providing contextual knowledge alone. In Study 2, using AutoGen, we evaluate a redesigned bidirectional communication structure and further measure the trade-offs between reasoning and non-reasoning models operating within the same robotic team setting. Drawing from our empirical findings, we emphasize the tension between autonomy and stability and the importance of edge-case testing to improve system reliability and safety for future real-world deployment. Supplementary materials, including codes, task agent setup, trace outputs, and annotated examples of coordination failures and reasoning behaviors, are available at: https://byc-sophie.github.io/mas-to-mars/. |
| 2025-08-06 | [Drone Detection with Event Cameras](http://arxiv.org/abs/2508.04564v1) | Gabriele Magrini, Lorenzo Berlincioni et al. | The diffusion of drones presents significant security and safety challenges. Traditional surveillance systems, particularly conventional frame-based cameras, struggle to reliably detect these targets due to their small size, high agility, and the resulting motion blur and poor performance in challenging lighting conditions. This paper surveys the emerging field of event-based vision as a robust solution to these problems. Event cameras virtually eliminate motion blur and enable consistent detection in extreme lighting. Their sparse, asynchronous output suppresses static backgrounds, enabling low-latency focus on motion cues. We review the state-of-the-art in event-based drone detection, from data representation methods to advanced processing pipelines using spiking neural networks. The discussion extends beyond simple detection to cover more sophisticated tasks such as real-time tracking, trajectory forecasting, and unique identification through propeller signature analysis. By examining current methodologies, available datasets, and the distinct advantages of the technology, this work demonstrates that event-based vision provides a powerful foundation for the next generation of reliable, low-latency, and efficient counter-UAV systems. |
| 2025-08-06 | [Bridging Simulation and Experiment: A Self-Supervised Domain Adaptation Framework for Concrete Damage Classification](http://arxiv.org/abs/2508.04538v1) | Chen Xu, Giao Vu et al. | Reliable assessment of concrete degradation is critical for ensuring structural safety and longevity of engineering structures. This study proposes a self-supervised domain adaptation framework for robust concrete damage classification using coda wave signals. To support this framework, an advanced virtual testing platform is developed, combining multiscale modeling of concrete degradation with ultrasonic wave propagation simulations. This setup enables the generation of large-scale labeled synthetic data under controlled conditions, reducing the dependency on costly and time-consuming experimental labeling. However, neural networks trained solely on synthetic data often suffer from degraded performance when applied to experimental data due to domain shifts. To bridge this domain gap, the proposed framework integrates domain adversarial training, minimum class confusion loss, and the Bootstrap Your Own Latent (BYOL) strategy. These components work jointly to facilitate effective knowledge transfer from the labeled simulation domain to the unlabeled experimental domain, achieving accurate and reliable damage classification in concrete. Extensive experiments demonstrate that the proposed method achieves notable performance improvements, reaching an accuracy of 0.7762 and a macro F1 score of 0.7713, outperforming both the plain 1D CNN baseline and six representative domain adaptation techniques. Moreover, the method exhibits high robustness across training runs and introduces only minimal additional computational cost. These findings highlight the practical potential of the proposed simulation-driven and label-efficient framework for real-world applications in structural health monitoring. |
| 2025-08-06 | [OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](http://arxiv.org/abs/2508.04482v1) | Xueyu Hu, Tao Xiong et al. | The dream to create AI assistants as capable and versatile as the fictional J.A.R.V.I.S from Iron Man has long captivated imaginations. With the evolution of (multi-modal) large language models ((M)LLMs), this dream is closer to reality, as (M)LLM-based Agents using computing devices (e.g., computers and mobile phones) by operating within the environments and interfaces (e.g., Graphical User Interface (GUI)) provided by operating systems (OS) to automate tasks have significantly advanced. This paper presents a comprehensive survey of these advanced agents, designated as OS Agents. We begin by elucidating the fundamentals of OS Agents, exploring their key components including the environment, observation space, and action space, and outlining essential capabilities such as understanding, planning, and grounding. We then examine methodologies for constructing OS Agents, focusing on domain-specific foundation models and agent frameworks. A detailed review of evaluation protocols and benchmarks highlights how OS Agents are assessed across diverse tasks. Finally, we discuss current challenges and identify promising directions for future research, including safety and privacy, personalization and self-evolution. This survey aims to consolidate the state of OS Agents research, providing insights to guide both academic inquiry and industrial development. An open-source GitHub repository is maintained as a dynamic resource to foster further innovation in this field. We present a 9-page version of our work, accepted by ACL 2025, to provide a concise overview to the domain. |
| 2025-08-06 | [Large Language Models Versus Static Code Analysis Tools: A Systematic Benchmark for Vulnerability Detection](http://arxiv.org/abs/2508.04448v1) | Damian Gnieciak, Tomasz Szandala | Modern software relies on a multitude of automated testing and quality assurance tools to prevent errors, bugs and potential vulnerabilities. This study sets out to provide a head-to-head, quantitative and qualitative evaluation of six automated approaches: three industry-standard rule-based static code-analysis tools (SonarQube, CodeQL and Snyk Code) and three state-of-the-art large language models hosted on the GitHub Models platform (GPT-4.1, Mistral Large and DeepSeek V3). Using a curated suite of ten real-world C# projects that embed 63 vulnerabilities across common categories such as SQL injection, hard-coded secrets and outdated dependencies, we measure classical detection accuracy (precision, recall, F-score), analysis latency, and the developer effort required to vet true positives. The language-based scanners achieve higher mean F-1 scores,0.797, 0.753 and 0.750, than their static counterparts, which score 0.260, 0.386 and 0.546, respectively. LLMs' advantage originates from superior recall, confirming an ability to reason across broader code contexts. However, this benefit comes with substantial trade-offs: DeepSeek V3 exhibits the highest false-positive ratio, all language models mislocate issues at line-or-column granularity due to tokenisation artefacts. Overall, language models successfully rival traditional static analysers in finding real vulnerabilities. Still, their noisier output and imprecise localisation limit their standalone use in safety-critical audits. We therefore recommend a hybrid pipeline: employ language models early in development for broad, context-aware triage, while reserving deterministic rule-based scanners for high-assurance verification. The open benchmark and JSON-based result harness released with this paper lay a foundation for reproducible, practitioner-centric research into next-generation automated code security. |
| 2025-08-06 | [Reliable and Real-Time Highway Trajectory Planning via Hybrid Learning-Optimization Frameworks](http://arxiv.org/abs/2508.04436v1) | Yujia Lu, Chong Wei et al. | Autonomous highway driving presents a high collision risk due to fast-changing environments and limited reaction time, necessitating reliable and efficient trajectory planning. This paper proposes a hybrid trajectory planning framework that integrates the adaptability of learning-based methods with the formal safety guarantees of optimization-based approaches. The framework features a two-layer architecture: an upper layer employing a graph neural network (GNN) trained on real-world highway data to predict human-like longitudinal velocity profiles, and a lower layer utilizing path optimization formulated as a mixed-integer quadratic programming (MIQP) problem. The primary contribution is the lower-layer path optimization model, which introduces a linear approximation of discretized vehicle geometry to substantially reduce computational complexity, while enforcing strict spatiotemporal non-overlapping constraints to formally guarantee collision avoidance throughout the planning horizon. Experimental results demonstrate that the planner generates highly smooth, collision-free trajectories in complex real-world emergency scenarios, achieving success rates exceeding 97% with average planning times of 54 ms, thereby confirming real-time capability. |
| 2025-08-06 | [Improving Crash Data Quality with Large Language Models: Evidence from Secondary Crash Narratives in Kentucky](http://arxiv.org/abs/2508.04399v1) | Xu Zhang, Mei Chen | This study evaluates advanced natural language processing (NLP) techniques to enhance crash data quality by mining crash narratives, using secondary crash identification in Kentucky as a case study. Drawing from 16,656 manually reviewed narratives from 2015-2022, with 3,803 confirmed secondary crashes, we compare three model classes: zero-shot open-source large language models (LLMs) (LLaMA3:70B, DeepSeek-R1:70B, Qwen3:32B, Gemma3:27B); fine-tuned transformers (BERT, DistilBERT, RoBERTa, XLNet, Longformer); and traditional logistic regression as baseline. Models were calibrated on 2015-2021 data and tested on 1,771 narratives from 2022. Fine-tuned transformers achieved superior performance, with RoBERTa yielding the highest F1-score (0.90) and accuracy (95%). Zero-shot LLaMA3:70B reached a comparable F1 of 0.86 but required 139 minutes of inference; the logistic baseline lagged well behind (F1:0.66). LLMs excelled in recall for some variants (e.g., GEMMA3:27B at 0.94) but incurred high computational costs (up to 723 minutes for DeepSeek-R1:70B), while fine-tuned models processed the test set in seconds after brief training. Further analysis indicated that mid-sized LLMs (e.g., DeepSeek-R1:32B) can rival larger counterparts in performance while reducing runtime, suggesting opportunities for optimized deployments. Results highlight trade-offs between accuracy, efficiency, and data requirements, with fine-tuned transformer models balancing precision and recall effectively on Kentucky data. Practical deployment considerations emphasize privacy-preserving local deployment, ensemble approaches for improved accuracy, and incremental processing for scalability, providing a replicable scheme for enhancing crash-data quality with advanced NLP. |
| 2025-08-06 | [Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](http://arxiv.org/abs/2508.04325v1) | Zizhan Ma, Wenxuan Wang et al. | Large language models (LLMs) show significant potential in healthcare, prompting numerous benchmarks to evaluate their capabilities. However, concerns persist regarding the reliability of these benchmarks, which often lack clinical fidelity, robust data management, and safety-oriented evaluation metrics. To address these shortcomings, we introduce MedCheck, the first lifecycle-oriented assessment framework specifically designed for medical benchmarks. Our framework deconstructs a benchmark's development into five continuous stages, from design to governance, and provides a comprehensive checklist of 46 medically-tailored criteria. Using MedCheck, we conducted an in-depth empirical evaluation of 53 medical LLM benchmarks. Our analysis uncovers widespread, systemic issues, including a profound disconnect from clinical practice, a crisis of data integrity due to unmitigated contamination risks, and a systematic neglect of safety-critical evaluation dimensions like model robustness and uncertainty awareness. Based on these findings, MedCheck serves as both a diagnostic tool for existing benchmarks and an actionable guideline to foster a more standardized, reliable, and transparent approach to evaluating AI in healthcare. |
| 2025-08-06 | [EVOC2RUST: A Skeleton-guided Framework for Project-Level C-to-Rust Translation](http://arxiv.org/abs/2508.04295v1) | Chaofan Wang, Tingrui Yu et al. | Rust's compile-time safety guarantees make it ideal for safety-critical systems, creating demand for translating legacy C codebases to Rust. While various approaches have emerged for this task, they face inherent trade-offs: rule-based solutions face challenges in meeting code safety and idiomaticity requirements, while LLM-based solutions often fail to generate semantically equivalent Rust code, due to the heavy dependencies of modules across the entire codebase. Recent studies have revealed that both solutions are limited to small-scale programs. In this paper, we propose EvoC2Rust, an automated framework for converting entire C projects to equivalent Rust ones. EvoC2Rust employs a skeleton-guided translation strategy for project-level translation. The pipeline consists of three evolutionary stages: 1) it first decomposes the C project into functional modules, employs a feature-mapping-enhanced LLM to transform definitions and macros and generates type-checked function stubs, which form a compilable Rust skeleton; 2) it then incrementally translates the function, replacing the corresponding stub placeholder; 3) finally, it repairs compilation errors by integrating LLM and static analysis. Through evolutionary augmentation, EvoC2Rust combines the advantages of both rule-based and LLM-based solutions. Our evaluation on open-source benchmarks and six industrial projects demonstrates EvoC2Rust's superior performance in project-level C-to-Rust translation. On average, it achieves 17.24% and 14.32% improvements in syntax and semantic accuracy over the LLM-based approaches, along with a 96.79% higher code safety rate than the rule-based tools. At the module level, EvoC2Rust reaches 92.25% compilation and 89.53% test pass rates on industrial projects, even for complex codebases and long functions. |
| 2025-08-06 | [Large Language Model's Multi-Capability Alignment in Biomedical Domain](http://arxiv.org/abs/2508.04278v1) | Wentao Wu, Linqing Chen et al. | BalancedBio is a theoretically grounded framework for parameter-efficient biomedical reasoning, addressing multi-capability integration in domain-specific AI alignment. It establishes the Biomedical Multi-Capability Convergence Theorem, proving orthogonal gradient spaces are essential to prevent capability interference for safe deployment. Key innovations include: (1) Medical Knowledge Grounded Synthetic Generation (MKGSG), extending Source2Synth with clinical workflow constraints and medical ontology validation for factual accuracy and safety; and (2) Capability Aware Group Relative Policy Optimization, deriving optimal hybrid reward weighting to maintain orthogonality in RL, using a reward model with rule-based and model-based scores adapted to biomedical tasks. Mathematical analysis proves Pareto-optimal convergence, preserving performance across capabilities. It achieves state-of-the-art results in its parameter class: domain expertise (80.95% BIOMED-MMLU, +15.32% over baseline), reasoning (61.94%, +7.75%), instruction following (67.95%, +6.44%), and integration (86.7%, +18.5%). Theoretical safety guarantees include bounds on capability preservation and clinical accuracy. Real-world deployment yields 78% cost reduction, 23% improved diagnostic accuracy, and 89% clinician acceptance. This work provides a principled methodology for biomedical AI alignment, enabling efficient reasoning with essential safety and reliability, with the 0.5B model version to be released. |
| 2025-08-06 | [Stabilization and Re-excitation of Sawtooth Oscillations due to Energetic Particles in Tokamaks](http://arxiv.org/abs/2508.04210v1) | H. X. Zhang, H. W. Zhang et al. | Sawtooth oscillations, driven by internal kink modes (IKMs), are fundamental phenomena in tokamak plasmas. They can be classified into different types, including normal sawteeth, small sawteeth, and in some cases, evolving into the steady-island state, each having a different impact on energy confinement in fusion reactors. This study investigates the interaction between sawtooth oscillations and energetic particles (EPs) using the initial-value MHD-kinetic hybrid code CLT-K, which can perform long-term self-consistent nonlinear simulations. We analyze the redistribution of EPs caused by sawtooth crashes and the effect of EPs on sawtooth behavior and type transitions. The results show that co-passing EPs tend to re-excite sawtooth oscillations, extending their period, while counter-passing EPs promote the system evolution toward small sawteeth, potentially leading to the steady-island state. Additionally, we provide a physical picture of how EPs influence sawtooth type through the mechanism of magnetic flux pumping. We demonstrate that the radial residual flow in the core plays a crucial role in determining the reconnection rate and sawtooth type. Moreover, we observe new phenomena about couplings of various instabilities, such as the excitation of global multi-mode toroidal Alfv\'en eigenmodes (TAEs) due to EP redistribution following a sawtooth crash and the excitation of the resonant tearing mode (r-TM) when injecting counter-passing EPs. The study also explores the impact of EP energy and the safety factor profile on the development of stochastic magnetic fields and EP transport. These findings emphasize the necessity of multi-mode simulations in capturing the complexity of EP-sawtooth interactions and provide insights for optimizing sawtooth control in future reactors such as ITER. |
| 2025-08-06 | [ReasoningGuard: Safeguarding Large Reasoning Models with Inference-time Safety Aha Moments](http://arxiv.org/abs/2508.04204v1) | Yuquan Wang, Mi Zhang et al. | Large Reasoning Models (LRMs) have demonstrated impressive performance in reasoning-intensive tasks, but they remain vulnerable to harmful content generation, particularly in the mid-to-late steps of their reasoning processes. Existing defense mechanisms, however, rely on costly fine-tuning and additional expert knowledge, which restricts their scalability. In this work, we propose ReasoningGuard, an inference-time safeguard for LRMs, which injects timely safety aha moments to steer harmless while helpful reasoning processes. Leveraging the model's internal attention behavior, our approach accurately identifies critical points in the reasoning path, and triggers spontaneous, safety-oriented reflection. To safeguard both the subsequent reasoning steps and the final answers, we further implement a scaling sampling strategy during the decoding phase, selecting the optimal reasoning path. Inducing minimal extra inference cost, ReasoningGuard effectively mitigates three types of jailbreak attacks, including the latest ones targeting the reasoning process of LRMs. Our approach outperforms seven existing safeguards, achieving state-of-the-art safety defenses while effectively avoiding the common exaggerated safety issues. |
| 2025-08-06 | [Eliciting and Analyzing Emergent Misalignment in State-of-the-Art Large Language Models](http://arxiv.org/abs/2508.04196v1) | Siddhant Panpatil, Hiskias Dingeto et al. | Despite significant advances in alignment techniques, we demonstrate that state-of-the-art language models remain vulnerable to carefully crafted conversational scenarios that can induce various forms of misalignment without explicit jailbreaking. Through systematic manual red-teaming with Claude-4-Opus, we discovered 10 successful attack scenarios, revealing fundamental vulnerabilities in how current alignment methods handle narrative immersion, emotional pressure, and strategic framing. These scenarios successfully elicited a range of misaligned behaviors, including deception, value drift, self-preservation, and manipulative reasoning, each exploiting different psychological and contextual vulnerabilities. To validate generalizability, we distilled our successful manual attacks into MISALIGNMENTBENCH, an automated evaluation framework that enables reproducible testing across multiple models. Cross-model evaluation of our 10 scenarios against five frontier LLMs revealed an overall 76% vulnerability rate, with significant variations: GPT-4.1 showed the highest susceptibility (90%), while Claude-4-Sonnet demonstrated greater resistance (40%). Our findings demonstrate that sophisticated reasoning capabilities often become attack vectors rather than protective mechanisms, as models can be manipulated into complex justifications for misaligned behavior. This work provides (i) a detailed taxonomy of conversational manipulation patterns and (ii) a reusable evaluation framework. Together, these findings expose critical gaps in current alignment strategies and highlight the need for robustness against subtle, scenario-based manipulation in future AI systems. |
| 2025-08-06 | [COPO: Consistency-Aware Policy Optimization](http://arxiv.org/abs/2508.04138v1) | Jinghang Han, Jiawei Chen et al. | Reinforcement learning has significantly enhanced the reasoning capabilities of Large Language Models (LLMs) in complex problem-solving tasks. Recently, the introduction of DeepSeek R1 has inspired a surge of interest in leveraging rule-based rewards as a low-cost alternative for computing advantage functions and guiding policy optimization. However, a common challenge observed across many replication and extension efforts is that when multiple sampled responses under a single prompt converge to identical outcomes, whether correct or incorrect, the group-based advantage degenerates to zero. This leads to vanishing gradients and renders the corresponding samples ineffective for learning, ultimately limiting training efficiency and downstream performance. To address this issue, we propose a consistency-aware policy optimization framework that introduces a structured global reward based on outcome consistency, the global loss based on it ensures that, even when model outputs show high intra-group consistency, the training process still receives meaningful learning signals, which encourages the generation of correct and self-consistent reasoning paths from a global perspective. Furthermore, we incorporate an entropy-based soft blending mechanism that adaptively balances local advantage estimation with global optimization, enabling dynamic transitions between exploration and convergence throughout training. Our method introduces several key innovations in both reward design and optimization strategy. We validate its effectiveness through substantial performance gains on multiple mathematical reasoning benchmarks, highlighting the proposed framework's robustness and general applicability. Code of this work has been released at https://github.com/hijih/copo-code.git. |
| 2025-08-06 | [Efficient Strategy for Improving Large Language Model (LLM) Capabilities](http://arxiv.org/abs/2508.04073v1) | Julián Camilo Velandia Gutiérrez | Large Language Models (LLMs) have become a milestone in the field of artificial intelligence and natural language processing. However, their large-scale deployment remains constrained by the need for significant computational resources. This work proposes starting from a base model to explore and combine data processing and careful data selection techniques, training strategies, and architectural adjustments to improve the efficiency of LLMs in resource-constrained environments and within a delimited knowledge base. The methodological approach included defining criteria for building reliable datasets, conducting controlled experiments with different configurations, and systematically evaluating the resulting variants in terms of capability, versatility, response time, and safety. Finally, comparative tests were conducted to measure the performance of the developed variants and to validate the effectiveness of the proposed strategies. This work is based on the master's thesis in Systems and Computer Engineering titled "Efficient Strategy for Improving the Capabilities of Large Language Models (LLMs)". |
| 2025-08-06 | [Radar-Based NLoS Pedestrian Localization for Darting-Out Scenarios Near Parked Vehicles with Camera-Assisted Point Cloud Interpretation](http://arxiv.org/abs/2508.04033v1) | Hee-Yeun Kim, Byeonggyu Park et al. | The presence of Non-Line-of-Sight (NLoS) blind spots resulting from roadside parking in urban environments poses a significant challenge to road safety, particularly due to the sudden emergence of pedestrians. mmWave technology leverages diffraction and reflection to observe NLoS regions, and recent studies have demonstrated its potential for detecting obscured objects. However, existing approaches predominantly rely on predefined spatial information or assume simple wall reflections, thereby limiting their generalizability and practical applicability. A particular challenge arises in scenarios where pedestrians suddenly appear from between parked vehicles, as these parked vehicles act as temporary spatial obstructions. Furthermore, since parked vehicles are dynamic and may relocate over time, spatial information obtained from satellite maps or other predefined sources may not accurately reflect real-time road conditions, leading to erroneous sensor interpretations. To address this limitation, we propose an NLoS pedestrian localization framework that integrates monocular camera image with 2D radar point cloud (PCD) data. The proposed method initially detects parked vehicles through image segmentation, estimates depth to infer approximate spatial characteristics, and subsequently refines this information using 2D radar PCD to achieve precise spatial inference. Experimental evaluations conducted in real-world urban road environments demonstrate that the proposed approach enhances early pedestrian detection and contributes to improved road safety. Supplementary materials are available at https://hiyeun.github.io/NLoS/. |
| 2025-08-06 | [HarmonyGuard: Toward Safety and Utility in Web Agents via Adaptive Policy Enhancement and Dual-Objective Optimization](http://arxiv.org/abs/2508.04010v1) | Yurun Chen, Xavier Hu et al. | Large language models enable agents to autonomously perform tasks in open web environments. However, as hidden threats within the web evolve, web agents face the challenge of balancing task performance with emerging risks during long-sequence operations. Although this challenge is critical, current research remains limited to single-objective optimization or single-turn scenarios, lacking the capability for collaborative optimization of both safety and utility in web environments. To address this gap, we propose HarmonyGuard, a multi-agent collaborative framework that leverages policy enhancement and objective optimization to jointly improve both utility and safety. HarmonyGuard features a multi-agent architecture characterized by two fundamental capabilities: (1) Adaptive Policy Enhancement: We introduce the Policy Agent within HarmonyGuard, which automatically extracts and maintains structured security policies from unstructured external documents, while continuously updating policies in response to evolving threats. (2) Dual-Objective Optimization: Based on the dual objectives of safety and utility, the Utility Agent integrated within HarmonyGuard performs the Markovian real-time reasoning to evaluate the objectives and utilizes metacognitive capabilities for their optimization. Extensive evaluations on multiple benchmarks show that HarmonyGuard improves policy compliance by up to 38% and task completion by up to 20% over existing baselines, while achieving over 90% policy compliance across all tasks. Our project is available here: https://github.com/YurunChen/HarmonyGuard. |
| 2025-08-06 | [The Emotional Baby Is Truly Deadly: Does your Multimodal Large Reasoning Model Have Emotional Flattery towards Humans?](http://arxiv.org/abs/2508.03986v1) | Yuan Xun, Xiaojun Jia et al. | We observe that MLRMs oriented toward human-centric service are highly susceptible to user emotional cues during the deep-thinking stage, often overriding safety protocols or built-in safety checks under high emotional intensity. Inspired by this key insight, we propose EmoAgent, an autonomous adversarial emotion-agent framework that orchestrates exaggerated affective prompts to hijack reasoning pathways. Even when visual risks are correctly identified, models can still produce harmful completions through emotional misalignment. We further identify persistent high-risk failure modes in transparent deep-thinking scenarios, such as MLRMs generating harmful reasoning masked behind seemingly safe responses. These failures expose misalignments between internal inference and surface-level behavior, eluding existing content-based safeguards. To quantify these risks, we introduce three metrics: (1) Risk-Reasoning Stealth Score (RRSS) for harmful reasoning beneath benign outputs; (2) Risk-Visual Neglect Rate (RVNR) for unsafe completions despite visual risk recognition; and (3) Refusal Attitude Inconsistency (RAIC) for evaluating refusal unstability under prompt variants. Extensive experiments on advanced MLRMs demonstrate the effectiveness of EmoAgent and reveal deeper emotional cognitive misalignments in model safety behavior. |
| 2025-08-05 | [Data and AI governance: Promoting equity, ethics, and fairness in large language models](http://arxiv.org/abs/2508.03970v1) | Alok Abhishek, Lisa Erickson et al. | In this paper, we cover approaches to systematically govern, assess and quantify bias across the complete life cycle of machine learning models, from initial development and validation to ongoing production monitoring and guardrail implementation. Building upon our foundational work on the Bias Evaluation and Assessment Test Suite (BEATS) for Large Language Models, the authors share prevalent bias and fairness related gaps in Large Language Models (LLMs) and discuss data and AI governance framework to address Bias, Ethics, Fairness, and Factuality within LLMs. The data and AI governance approach discussed in this paper is suitable for practical, real-world applications, enabling rigorous benchmarking of LLMs prior to production deployment, facilitating continuous real-time evaluation, and proactively governing LLM generated responses. By implementing the data and AI governance across the life cycle of AI development, organizations can significantly enhance the safety and responsibility of their GenAI systems, effectively mitigating risks of discrimination and protecting against potential reputational or brand-related harm. Ultimately, through this article, we aim to contribute to advancement of the creation and deployment of socially responsible and ethically aligned generative artificial intelligence powered applications. |
| 2025-08-05 | [ASTRA: Autonomous Spatial-Temporal Red-teaming for AI Software Assistants](http://arxiv.org/abs/2508.03936v1) | Xiangzhe Xu, Guangyu Shen et al. | AI coding assistants like GitHub Copilot are rapidly transforming software development, but their safety remains deeply uncertain-especially in high-stakes domains like cybersecurity. Current red-teaming tools often rely on fixed benchmarks or unrealistic prompts, missing many real-world vulnerabilities. We present ASTRA, an automated agent system designed to systematically uncover safety flaws in AI-driven code generation and security guidance systems. ASTRA works in three stages: (1) it builds structured domain-specific knowledge graphs that model complex software tasks and known weaknesses; (2) it performs online vulnerability exploration of each target model by adaptively probing both its input space, i.e., the spatial exploration, and its reasoning processes, i.e., the temporal exploration, guided by the knowledge graphs; and (3) it generates high-quality violation-inducing cases to improve model alignment. Unlike prior methods, ASTRA focuses on realistic inputs-requests that developers might actually ask-and uses both offline abstraction guided domain modeling and online domain knowledge graph adaptation to surface corner-case vulnerabilities. Across two major evaluation domains, ASTRA finds 11-66% more issues than existing techniques and produces test cases that lead to 17% more effective alignment training, showing its practical value for building safer AI systems. |

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



