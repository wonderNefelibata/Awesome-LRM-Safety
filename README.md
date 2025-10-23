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
| 2025-10-22 | [Learning and Simulating Building Evacuation Patterns for Enhanced Safety Design Using Generative Models](http://arxiv.org/abs/2510.19623v1) | Jin Han, Zhe Zheng et al. | Evacuation simulation is essential for building safety design, ensuring properly planned evacuation routes. However, traditional evacuation simulation relies heavily on refined modeling with extensive parameters, making it challenging to adopt such methods in a rapid iteration process in early design stages. Thus, this study proposes DiffEvac, a novel method to learn building evacuation patterns based on Generative Models (GMs), for efficient evacuation simulation and enhanced safety design. Initially, a dataset of 399 diverse functional layouts and corresponding evacuation heatmaps of buildings was established. Then, a decoupled feature representation is proposed to embed physical features like layouts and occupant density for GMs. Finally, a diffusion model based on image prompts is proposed to learn evacuation patterns from simulated evacuation heatmaps. Compared to existing research using Conditional GANs with RGB representation, DiffEvac achieves up to a 37.6% improvement in SSIM, 142% in PSNR, and delivers results 16 times faster, thereby cutting simulation time to 2 minutes. Case studies further demonstrate that the proposed method not only significantly enhances the rapid design iteration and adjustment process with efficient evacuation simulation but also offers new insights and technical pathways for future safety optimization in intelligent building design. The research implication is that the approach lowers the modeling burden, enables large-scale what-if exploration, and facilitates coupling with multi-objective design tools. |
| 2025-10-22 | [A Concrete Roadmap towards Safety Cases based on Chain-of-Thought Monitoring](http://arxiv.org/abs/2510.19476v1) | Julian Schulz | As AI systems approach dangerous capability levels where inability safety cases become insufficient, we need alternative approaches to ensure safety. This paper presents a roadmap for constructing safety cases based on chain-of-thought (CoT) monitoring in reasoning models and outlines our research agenda. We argue that CoT monitoring might support both control and trustworthiness safety cases. We propose a two-part safety case: (1) establishing that models lack dangerous capabilities when operating without their CoT, and (2) ensuring that any dangerous capabilities enabled by a CoT are detectable by CoT monitoring. We systematically examine two threats to monitorability: neuralese and encoded reasoning, which we categorize into three forms (linguistic drift, steganography, and alien reasoning) and analyze their potential drivers. We evaluate existing and novel techniques for maintaining CoT faithfulness. For cases where models produce non-monitorable reasoning, we explore the possibility of extracting a monitorable CoT from a non-monitorable CoT. To assess the viability of CoT monitoring safety cases, we establish prediction markets to aggregate forecasts on key technical milestones influencing their feasibility. |
| 2025-10-22 | [AutoMT: A Multi-Agent LLM Framework for Automated Metamorphic Testing of Autonomous Driving Systems](http://arxiv.org/abs/2510.19438v1) | Linfeng Liang, Chenkai Tan et al. | Autonomous Driving Systems (ADS) are safety-critical, where failures can be severe. While Metamorphic Testing (MT) is effective for fault detection in ADS, existing methods rely heavily on manual effort and lack automation. We present AutoMT, a multi-agent MT framework powered by Large Language Models (LLMs) that automates the extraction of Metamorphic Relations (MRs) from local traffic rules and the generation of valid follow-up test cases. AutoMT leverages LLMs to extract MRs from traffic rules in Gherkin syntax using a predefined ontology. A vision-language agent analyzes scenarios, and a search agent retrieves suitable MRs from a RAG-based database to generate follow-up cases via computer vision. Experiments show that AutoMT achieves up to 5 x higher test diversity in follow-up case generation compared to the best baseline (manual expert-defined MRs) in terms of validation rate, and detects up to 20.55% more behavioral violations. While manual MT relies on a fixed set of predefined rules, AutoMT automatically extracts diverse metamorphic relations that augment real-world datasets and help uncover corner cases often missed during in-field testing and data collection. Its modular architecture separating MR extraction, filtering, and test generation supports integration into industrial pipelines and potentially enables simulation-based testing to systematically cover underrepresented or safety-critical scenarios. |
| 2025-10-22 | [LoongRL:Reinforcement Learning for Advanced Reasoning over Long Contexts](http://arxiv.org/abs/2510.19363v1) | Siyuan Wang, Gaokai Zhang et al. | Reasoning over long contexts is essential for large language models. While reinforcement learning (RL) enhances short-context reasoning by inducing "Aha" moments in chain-of-thought, the advanced thinking patterns required for long-context reasoning remain largely unexplored, and high-difficulty RL data are scarce. In this paper, we introduce LoongRL, a data-driven RL method for advanced long-context reasoning. Central to LoongRL is KeyChain, a synthesis approach that transforms short multi-hop QA into high-difficulty long-context tasks by inserting UUID chains that hide the true question among large collections of distracting documents. Solving these tasks requires the model to trace the correct chain step-by-step, identify the true question, retrieve relevant facts and reason over them to answer correctly. RL training on KeyChain data induces an emergent plan-retrieve-reason-recheck reasoning pattern that generalizes far beyond training length. Models trained at 16K effectively solve 128K tasks without prohibitive full-length RL rollout costs. On Qwen2.5-7B and 14B, LoongRL substantially improves long-context multi-hop QA accuracy by +23.5% and +21.1% absolute gains. The resulting LoongRL-14B reaches a score of 74.2, rivaling much larger frontier models such as o3-mini (74.5) and DeepSeek-R1 (74.9). It also improves long-context retrieval, passes all 128K needle-in-a-haystack stress tests, and preserves short-context reasoning capabilities. |
| 2025-10-22 | [SORA-ATMAS: Adaptive Trust Management and Multi-LLM Aligned Governance for Future Smart Cities](http://arxiv.org/abs/2510.19327v1) | Usama Antuley, Shahbaz Siddiqui et al. | The rapid evolution of smart cities has increased the reliance on intelligent interconnected services to optimize infrastructure, resources, and citizen well-being. Agentic AI has emerged as a key enabler by supporting autonomous decision-making and adaptive coordination, allowing urban systems to respond in real time to dynamic conditions. Its benefits are evident in areas such as transportation, where the integration of traffic data, weather forecasts, and safety sensors enables dynamic rerouting and a faster response to hazards. However, its deployment across heterogeneous smart city ecosystems raises critical governance, risk, and compliance (GRC) challenges, including accountability, data privacy, and regulatory alignment within decentralized infrastructures. Evaluation of SORA-ATMAS with three domain agents (Weather, Traffic, and Safety) demonstrated that its governance policies, including a fallback mechanism for high-risk scenarios, effectively steer multiple LLMs (GPT, Grok, DeepSeek) towards domain-optimized, policy-aligned outputs, producing an average MAE reduction of 35% across agents. Results showed stable weather monitoring, effective handling of high-risk traffic plateaus 0.85, and adaptive trust regulation in Safety/Fire scenarios 0.65. Runtime profiling of a 3-agent deployment confirmed scalability, with throughput between 13.8-17.2 requests per second, execution times below 72~ms, and governance delays under 100 ms, analytical projections suggest maintained performance at larger scales. Cross-domain rules ensured safe interoperability, with traffic rerouting permitted only under validated weather conditions. These findings validate SORA-ATMAS as a regulation-aligned, context-aware, and verifiable governance framework that consolidates distributed agent outputs into accountable, real-time decisions, offering a resilient foundation for smart-city management. |
| 2025-10-22 | [Vision-Based Mistake Analysis in Procedural Activities: A Review of Advances and Challenges](http://arxiv.org/abs/2510.19292v1) | Konstantinos Bacharidis, Antonis A. Argyros | Mistake analysis in procedural activities is a critical area of research with applications spanning industrial automation, physical rehabilitation, education and human-robot collaboration. This paper reviews vision-based methods for detecting and predicting mistakes in structured tasks, focusing on procedural and executional errors. By leveraging advancements in computer vision, including action recognition, anticipation and activity understanding, vision-based systems can identify deviations in task execution, such as incorrect sequencing, use of improper techniques, or timing errors. We explore the challenges posed by intra-class variability, viewpoint differences and compositional activity structures, which complicate mistake detection. Additionally, we provide a comprehensive overview of existing datasets, evaluation metrics and state-of-the-art methods, categorizing approaches based on their use of procedural structure, supervision levels and learning strategies. Open challenges, such as distinguishing permissible variations from true mistakes and modeling error propagation are discussed alongside future directions, including neuro-symbolic reasoning and counterfactual state modeling. This work aims to establish a unified perspective on vision-based mistake analysis in procedural activities, highlighting its potential to enhance safety, efficiency and task performance across diverse domains. |
| 2025-10-22 | [TARMAC: A Taxonomy for Robot Manipulation in Chemistry](http://arxiv.org/abs/2510.19289v1) | Kefeng Huang, Jonathon Pipe et al. | Chemistry laboratory automation aims to increase throughput, reproducibility, and safety, yet many existing systems still depend on frequent human intervention. Advances in robotics have reduced this dependency, but without a structured representation of the required skills, autonomy remains limited to bespoke, task-specific solutions with little capacity to transfer beyond their initial design. Current experiment abstractions typically describe protocol-level steps without specifying the robotic actions needed to execute them. This highlights the lack of a systematic account of the manipulation skills required for robots in chemistry laboratories. To address this gap, we introduce TARMAC - a Taxonomy for Robot Manipulation in Chemistry - a domain-specific framework that defines and organizes the core manipulations needed in laboratory practice. Based on annotated teaching-lab demonstrations and supported by experimental validation, TARMAC categorizes actions according to their functional role and physical execution requirements. Beyond serving as a descriptive vocabulary, TARMAC can be instantiated as robot-executable primitives and composed into higher-level macros, enabling skill reuse and supporting scalable integration into long-horizon workflows. These contributions provide a structured foundation for more flexible and autonomous laboratory automation. More information is available at https://tarmac-paper.github.io/ |
| 2025-10-22 | [A Study on Delay Assessment for Heterogenous Traffic in VANET](http://arxiv.org/abs/2510.19267v1) | Shama Siddiqu, Indrakshi Dey | Vehicular Ad hoc Networks (VANETs) comprise of multi-priority hetero-genous nodes, both stationary and/or mobile. The data generated by these nodes may include messages relating to information, safety, entertainment, traffic management and emergency alerts. The data in the network needs dif-ferentiated service based on the priority/urgency. Media Access Control (MAC) protocols hold a significant value for managing the data priority. This paper studies a comparison of 802.11p which is a standard PHY and MAC protocol for VANET with a fragmentation-based protocol, FROG-MAC. The major design principle of 802.11-p is to allow direct Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication without associa-tion, using Enhanced Distributed Channel Access (EDCA) to prioritize safety-critical messages. However, if non-critical messages already start to transmit, the nodes with critical data have to wait. FROG-MAC reduces this delay by transmitting normal packets in fragments with short pauses between them, al-lowing urgent packets to access the channel during these intervals. Simula-tions have been performed to assess the delay and throughput for high and low priority data. We report that FROG-MAC improves both the performance parameters due to offering an early channel access to the emergency traffic. |
| 2025-10-22 | [LAPRAD: LLM-Assisted PRotocol Attack Discovery](http://arxiv.org/abs/2510.19264v1) | R. Can Aygun, Yehuda Afek et al. | With the goal of improving the security of Internet protocols, we seek faster, semi-automatic methods to discover new vulnerabilities in protocols such as DNS, BGP, and others. To this end, we introduce the LLM-Assisted Protocol Attack Discovery (LAPRAD) methodology, enabling security researchers with some DNS knowledge to efficiently uncover vulnerabilities that would otherwise be hard to detect.   LAPRAD follows a three-stage process. In the first, we consult an LLM (GPT-o1) that has been trained on a broad corpus of DNS-related sources and previous DDoS attacks to identify potential exploits. In the second stage, a different LLM automatically constructs the corresponding attack configurations using the ReACT approach implemented via LangChain (DNS zone file generation). Finally, in the third stage, we validate the attack's functionality and effectiveness.   Using LAPRAD, we uncovered three new DDoS attacks on the DNS protocol and rediscovered two recently reported ones that were not included in the LLM's training data. The first new attack employs a bait-and-switch technique to trick resolvers into caching large, bogus DNSSEC RRSIGs, reducing their serving capacity to as little as 6%. The second exploits large DNSSEC encryption algorithms (RSA-4096) with multiple keys, thereby bypassing a recently implemented default RRSet limit. The third leverages ANY-type responses to produce a similar effect.   These variations of a cache-flushing DDoS attack, called SigCacheFlush, circumvent existing patches, severely degrade resolver query capacity, and impact the latest versions of major DNS resolver implementations. |
| 2025-10-22 | [Trace: Securing Smart Contract Repository Against Access Control Vulnerability](http://arxiv.org/abs/2510.19254v1) | Chong Chen, Jiachi Chen et al. | Smart contract vulnerabilities, particularly improper Access Control that allows unauthorized execution of restricted functions, have caused billions of dollars in losses. GitHub hosts numerous smart contract repositories containing source code, documentation, and configuration files-these serve as intermediate development artifacts that must be compiled and packaged before deployment. Third-party developers often reference, reuse, or fork code from these repositories during custom development. However, if the referenced code contains vulnerabilities, it can introduce significant security risks. Existing tools for detecting smart contract vulnerabilities are limited in their ability to handle complex repositories, as they typically require the target contract to be compilable to generate an abstract representation for further analysis. This paper presents TRACE, a tool designed to secure non-compilable smart contract repositories against access control vulnerabilities. TRACE employs LLMs to locate sensitive functions involving critical operations (e.g., transfer) within the contract and subsequently completes function snippets into a fully compilable contract. TRACE constructs a function call graph from the abstract syntax tree (AST) of the completed contract. It uses the control flow graph (CFG) of each function as node information. The nodes of the sensitive functions are then analyzed to detect Access Control vulnerabilities. Experimental results demonstrate that TRACE outperforms state-of-the-art tools on an open-sourced CVE dataset, detecting 14 out of 15 CVEs. In addition, it achieves 89.2% precision on 5,000 recent on-chain contracts, far exceeding the best existing tool at 76.9%. On 83 real-world repositories, TRACE achieves 87.0% precision, significantly surpassing DeepSeek-R1's 14.3%. |
| 2025-10-22 | [Laser fabrication of Ti stent and facile MEMS flow sensor integration for implantable respiration monitoring](http://arxiv.org/abs/2510.19228v1) | Muhammad Salman Al Farisi, Takuya Kawata et al. | Animal experiments play a vital role in drug discovery and development by providing essential data on a drug's efficacy, safety, and physiological effects before advancing to human clinical trials. In this study, we propose a stent-based flow sensor designed to measure airflow in the airways of laboratory animals. The stent was fabricated from biocompatible Ti using a combination of fiber laser digital processing and an origami-inspired folding technique. The sensing structure was developed through standard micro-electromechanical systems (MEMS) microfabrication technology. To integrate the sensing structure with the metallic stent, a facile insertion process was employed, where the sensor film was positioned at the stent's center using its natural buckling mechanism. Once fabricated, the stent implant was expanded and installed within an airway-mimicking tube to validate its functionality. A proof-of-concept trial using an artificial ventilator successfully demonstrated real-time respiration monitoring, confirming the feasibility of the proposed system for airflow measurement in preclinical studies. This stent-based sensor offers a promising approach for enhancing respiratory assessments in laboratory animals, potentially improving the accuracy of drug evaluations and respiratory disease research. |
| 2025-10-22 | [OpenGuardrails: An Open-Source Context-Aware AI Guardrails Platform](http://arxiv.org/abs/2510.19169v1) | Thomas Wang, Haowen Li | As large language models (LLMs) become increasingly integrated into real-world applications, safeguarding them against unsafe, malicious, or privacy-violating content is critically important. We present OpenGuardrails, the first open-source project to provide both a context-aware safety and manipulation detection model and a deployable platform for comprehensive AI guardrails. OpenGuardrails protects against content-safety risks, model-manipulation attacks (e.g., prompt injection, jailbreaking, code-interpreter abuse, and the generation/execution of malicious code), and data leakage. Content-safety and model-manipulation detection are implemented by a unified large model, while data-leakage identification and redaction are performed by a separate lightweight NER pipeline (e.g., Presidio-style models or regex-based detectors). The system can be deployed as a security gateway or an API-based service, with enterprise-grade, fully private deployment options. OpenGuardrails achieves state-of-the-art (SOTA) performance on safety benchmarks, excelling in both prompt and response classification across English, Chinese, and multilingual tasks. All models are released under the Apache 2.0 license for public use. |
| 2025-10-22 | [Subliminal Corruption: Mechanisms, Thresholds, and Interpretability](http://arxiv.org/abs/2510.19152v1) | Reya Vir, Sarvesh Bhatnagar | As machine learning models are increasingly fine-tuned on synthetic data, there is a critical risk of subtle misalignments spreading through interconnected AI systems. This paper investigates subliminal corruption, which we define as undesirable traits are transmitted through semantically neutral data, bypassing standard safety checks. While this phenomenon has been identified, a quantitative understanding of its dynamics is missing. To address this gap, we present a systematic study of the scaling laws, thresholds, and mechanisms of subliminal corruption using a teacher-student setup with GPT-2. Our experiments reveal three key findings: (1) subliminal corruption causes behavioral crossover, degrading the model's overall alignment, not just the targeted trait; (2) alignment fails in a sharp phase transition at a critical threshold of poisoned data, rather than degrading gradually; and (3) interpretability analysis shows the corruption mechanism mimics the model's natural fine-tuning process, making it difficult to detect. These results demonstrate a critical vulnerability in AI systems that rely on synthetic data and highlight the need for new safety protocols that can account for latent threats. |
| 2025-10-21 | [A Cross-Environment and Cross-Embodiment Path Planning Framework via a Conditional Diffusion Model](http://arxiv.org/abs/2510.19128v1) | Mehran Ghafarian Tamizi, Homayoun Honari et al. | Path planning for a robotic system in high-dimensional cluttered environments needs to be efficient, safe, and adaptable for different environments and hardware. Conventional methods face high computation time and require extensive parameter tuning, while prior learning-based methods still fail to generalize effectively. The primary goal of this research is to develop a path planning framework capable of generalizing to unseen environments and new robotic manipulators without the need for retraining. We present GADGET (Generalizable and Adaptive Diffusion-Guided Environment-aware Trajectory generation), a diffusion-based planning model that generates joint-space trajectories conditioned on voxelized scene representations as well as start and goal configurations. A key innovation is GADGET's hybrid dual-conditioning mechanism that combines classifier-free guidance via learned scene encoding with classifier-guided Control Barrier Function (CBF) safety shaping, integrating environment awareness with real-time collision avoidance directly in the denoising process. This design supports zero-shot transfer to new environments and robotic embodiments without retraining. Experimental results show that GADGET achieves high success rates with low collision intensity in spherical-obstacle, bin-picking, and shelf environments, with CBF guidance further improving safety. Moreover, comparative evaluations indicate strong performance relative to both sampling-based and learning-based baselines. Furthermore, GADGET provides transferability across Franka Panda, Kinova Gen3 (6/7-DoF), and UR5 robots, and physical execution on a Kinova Gen3 demonstrates its ability to generate safe, collision-free trajectories in real-world settings. |
| 2025-10-21 | [A Configurable Simulation Framework for Safety Assessment of Vulnerable Road Users](http://arxiv.org/abs/2510.19097v1) | Zhitong He, Yaobin Chen et al. | Ensuring the safety of vulnerable road users (VRUs), including pedestrians, cyclists, electric scooter riders, and motorcyclists, remains a major challenge for advanced driver assistance systems (ADAS) and connected and automated vehicles (CAV) technologies. Real-world VRU tests are expensive and sometimes cannot capture or repeat rare and hazardous events. In this paper, we present a lightweight, configurable simulation framework that follows European New Car Assessment Program (Euro NCAP) VRU testing protocols. A rule-based finite-state machine (FSM) is developed as a motion planner to provide vehicle automation during the VRU interaction. We also integrate ego-vehicle perception and idealized Vehicle-to-Everything (V2X) awareness to demonstrate safety margins in different scenarios. This work provides an extensible platform for rapid and repeatable VRU safety validation, paving the way for broader case-study deployment in diverse, user-defined settings, which will be essential for building a more VRU-friendly and sustainable intelligent transportation system. |
| 2025-10-21 | [When Can We Trust LLMs in Mental Health? Large-Scale Benchmarks for Reliable LLM Evaluation](http://arxiv.org/abs/2510.19032v1) | Abeer Badawi, Elahe Rahimi et al. | Evaluating Large Language Models (LLMs) for mental health support is challenging due to the emotionally and cognitively complex nature of therapeutic dialogue. Existing benchmarks are limited in scale, reliability, often relying on synthetic or social media data, and lack frameworks to assess when automated judges can be trusted. To address the need for large-scale dialogue datasets and judge reliability assessment, we introduce two benchmarks that provide a framework for generation and evaluation. MentalBench-100k consolidates 10,000 one-turn conversations from three real scenarios datasets, each paired with nine LLM-generated responses, yielding 100,000 response pairs. MentalAlign-70k}reframes evaluation by comparing four high-performing LLM judges with human experts across 70,000 ratings on seven attributes, grouped into Cognitive Support Score (CSS) and Affective Resonance Score (ARS). We then employ the Affective Cognitive Agreement Framework, a statistical methodology using intraclass correlation coefficients (ICC) with confidence intervals to quantify agreement, consistency, and bias between LLM judges and human experts. Our analysis reveals systematic inflation by LLM judges, strong reliability for cognitive attributes such as guidance and informativeness, reduced precision for empathy, and some unreliability in safety and relevance. Our contributions establish new methodological and empirical foundations for reliable, large-scale evaluation of LLMs in mental health. We release the benchmarks and codes at: https://github.com/abeerbadawi/MentalBench/ |
| 2025-10-21 | [Plural Voices, Single Agent: Towards Inclusive AI in Multi-User Domestic Spaces](http://arxiv.org/abs/2510.19008v1) | Joydeep Chandra, Satyam Kumar Navneet | Domestic AI agents faces ethical, autonomy, and inclusion challenges, particularly for overlooked groups like children, elderly, and Neurodivergent users. We present the Plural Voices Model (PVM), a novel single-agent framework that dynamically negotiates multi-user needs through real-time value alignment, leveraging diverse public datasets on mental health, eldercare, education, and moral reasoning. Using human+synthetic curriculum design with fairness-aware scenarios and ethical enhancements, PVM identifies core values, conflicts, and accessibility requirements to inform inclusive principles. Our privacy-focused prototype features adaptive safety scaffolds, tailored interactions (e.g., step-by-step guidance for Neurodivergent users, simple wording for children), and equitable conflict resolution. In preliminary evaluations, PVM outperforms multi-agent baselines in compliance (76% vs. 70%), fairness (90% vs. 85%), safety-violation rate (0% vs. 7%), and latency. Design innovations, including video guidance, autonomy sliders, family hubs, and adaptive safety dashboards, demonstrate new directions for ethical and inclusive domestic AI, for building user-centered agentic systems in plural domestic contexts. Our Codes and Model are been open sourced, available for reproduction: https://github.com/zade90/Agora |
| 2025-10-21 | [Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search](http://arxiv.org/abs/2510.18939v1) | Howard Yen, Ashwin Paranjape et al. | Long-horizon agentic search requires iteratively exploring the web over long trajectories and synthesizing information across many sources, and is the foundation for enabling powerful applications like deep research systems. In this work, we show that popular agentic search frameworks struggle to scale to long trajectories primarily due to context limitations-they accumulate long, noisy content, hit context window and tool budgets, or stop early. Then, we introduce SLIM (Simple Lightweight Information Management), a simple framework that separates retrieval into distinct search and browse tools, and periodically summarizes the trajectory, keeping context concise while enabling longer, more focused searches. On long-horizon tasks, SLIM achieves comparable performance at substantially lower cost and with far fewer tool calls than strong open-source baselines across multiple base models. Specifically, with o3 as the base model, SLIM achieves 56% on BrowseComp and 31% on HLE, outperforming all open-source frameworks by 8 and 4 absolute points, respectively, while incurring 4-6x fewer tool calls. Finally, we release an automated fine-grained trajectory analysis pipeline and error taxonomy for characterizing long-horizon agentic search frameworks; SLIM exhibits fewer hallucinations than prior systems. We hope our analysis framework and simple tool design inform future long-horizon agents. |
| 2025-10-21 | [EffiReasonTrans: RL-Optimized Reasoning for Code Translation](http://arxiv.org/abs/2510.18863v1) | Yanlin Wang, Rongyi Ou et al. | Code translation is a crucial task in software development and maintenance. While recent advancements in large language models (LLMs) have improved automated code translation accuracy, these gains often come at the cost of increased inference latency, hindering real-world development workflows that involve human-in-the-loop inspection. To address this trade-off, we propose EffiReasonTrans, a training framework designed to improve translation accuracy while balancing inference latency. We first construct a high-quality reasoning-augmented dataset by prompting a stronger language model, DeepSeek-R1, to generate intermediate reasoning and target translations. Each (source code, reasoning, target code) triplet undergoes automated syntax and functionality checks to ensure reliability. Based on this dataset, we employ a two-stage training strategy: supervised fine-tuning on reasoning-augmented samples, followed by reinforcement learning to further enhance accuracy and balance inference latency. We evaluate EffiReasonTrans on six translation pairs. Experimental results show that it consistently improves translation accuracy (up to +49.2% CA and +27.8% CodeBLEU compared to the base model) while reducing the number of generated tokens (up to -19.3%) and lowering inference latency in most cases (up to -29.0%). Ablation studies further confirm the complementary benefits of the two-stage training framework. Additionally, EffiReasonTrans demonstrates improved translation accuracy when integrated into agent-based frameworks. Our code and data are available at https://github.com/DeepSoftwareAnalytics/EffiReasonTrans. |
| 2025-10-21 | [Lyapunov-Aware Quantum-Inspired Reinforcement Learning for Continuous-Time Vehicle Control: A Feasibility Study](http://arxiv.org/abs/2510.18852v1) | Nutkritta Kraipatthanapong, Natthaphat Thathong et al. | This paper presents a novel Lyapunov-Based Quantum Reinforcement Learning (LQRL) framework that integrates quantum policy optimization with Lyapunov stability analysis for continuous-time vehicle control. The proposed approach combines the representational power of variational quantum circuits (VQCs) with a stability-aware policy gradient mechanism to ensure asymptotic convergence and safe decision-making under dynamic environments. The vehicle longitudinal control problem was formulated as a continuous-state reinforcement learning task, where the quantum policy network generates control actions subject to Lyapunov stability constraints. Simulation experiments were conducted in a closed-loop adaptive cruise control scenario using a quantum-inspired policy trained under stability feedback. The results demonstrate that the LQRL framework successfully embeds Lyapunov stability verification into quantum policy learning, enabling interpretable and stability-aware control performance. Although transient overshoot and Lyapunov divergence were observed under aggressive acceleration, the system maintained bounded state evolution, validating the feasibility of integrating safety guarantees within quantum reinforcement learning architectures. The proposed framework provides a foundational step toward provably safe quantum control in autonomous systems and hybrid quantum-classical optimization domains. |

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



