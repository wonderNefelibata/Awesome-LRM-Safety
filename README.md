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
| 2026-04-22 | [AVISE: Framework for Evaluating the Security of AI Systems](http://arxiv.org/abs/2604.20833v1) | Mikko Lempinen, Joni Kemppainen et al. | As artificial intelligence (AI) systems are increasingly deployed across critical domains, their security vulnerabilities pose growing risks of high-profile exploits and consequential system failures. Yet systematic approaches to evaluating AI security remain underdeveloped. In this paper, we introduce AVISE (AI Vulnerability Identification and Security Evaluation), a modular open-source framework for identifying vulnerabilities in and evaluating the security of AI systems and models. As a demonstration of the framework, we extend the theory-of-mind-based multi-turn Red Queen attack into an Adversarial Language Model (ALM) augmented attack and develop an automated Security Evaluation Test (SET) for discovering jailbreak vulnerabilities in language models. The SET comprises 25 test cases and an Evaluation Language Model (ELM) that determines whether each test case was able to jailbreak the target model, achieving 92% accuracy, an F1-score of 0.91, and a Matthews correlation coefficient of 0.83. We evaluate nine recently released language models of diverse sizes with the SET and find that all are vulnerable to the augmented Red Queen attack to varying degrees. AVISE provides researchers and industry practitioners with an extensible foundation for developing and deploying automated SETs, offering a concrete step toward more rigorous and reproducible AI security evaluation. |
| 2026-04-22 | [A Hough transform approach to safety-aware scalar field mapping using Gaussian Processes](http://arxiv.org/abs/2604.20799v1) | Muzaffar Qureshi, Trivikram Satharasi et al. | This paper presents a framework for mapping unknown scalar fields using a sensor-equipped autonomous robot operating in unsafe environments. The unsafe regions are defined as regions of high-intensity, where the field value exceeds a predefined safety threshold. For safe and efficient mapping of the scalar field, the sensor-equipped robot must avoid high-intensity regions during the measurement process. In this paper, the scalar field is modeled as a sample from a Gaussian process (GP), which enables Bayesian inference and provides closed-form expressions for both the predictive mean and the uncertainty. Concurrently, the spatial structure of the high-intensity regions is estimated in real-time using the Hough transform (HT), leveraging the evolving GP posterior. A safe sampling strategy is then employed to guide the robot towards safe measurement locations, using probabilistic safety guarantees on the evolving GP posterior. The estimated high-intensity regions also facilitate the design of safe motion plans for the robot. The effectiveness of the approach is verified through two numerical simulation studies and an indoor experiment for mapping a light-intensity field using a wheeled mobile robot. |
| 2026-04-22 | [DAIRE: A lightweight AI model for real-time detection of Controller Area Network attacks in the Internet of Vehicles](http://arxiv.org/abs/2604.20771v1) | Shahid Alam, Amina Jameel et al. | The Internet of Vehicles (IoV) is advancing modern transportation by improving safety, efficiency, and intelligence. However, the reliance on the Controller Area Network (CAN) introduces critical security risks, as CAN-based communication is highly vulnerable to cyberattacks. Addressing this challenge, we propose DAIRE (Detecting Attacks in IoV in REal-time), a lightweight machine learning framework designed for real-time detection and classification of CAN attacks. DAIRE is built on a lightweight artificial neural network (ANN) where each layer contains Ni = i x c neurons, with Ni representing the number of neurons in the ith layer and c corresponding to the total number of attack classes. Other hyperparameters are determined empirically to ensure real-time operation. To support the detection and classification of various IoV attacks, such as Denial-of-Service, Fuzzy, and Spoofing, DAIRE employs the sparse categorical cross-entropy loss function and root mean square propagation for loss minimization. In contrast to more resource-intensive architectures, DAIRE leverages a lightweight ANN to reduce computational demands while still delivering strong performance. Experimental results on the CICIoV2024 and Car-Hacking datasets demonstrate DAIRE's effectiveness, achieving an average detection rate of 99.88%, a false positive rate of 0.02%, and an overall accuracy of 99.96%. Furthermore, DAIRE significantly outperforms state-of-the-art approaches in inference speed, with a classification time of just 0.03 ms per sample. These results highlight DAIRE's effectiveness in detecting IoV cyberattacks and its practical suitability for real-time deployment in vehicular systems, underscoring its vital role in strengthening automotive cybersecurity. |
| 2026-04-22 | [RespondeoQA: a Benchmark for Bilingual Latin-English Question Answering](http://arxiv.org/abs/2604.20738v1) | Marisa Hudspeth, Patrick J. Burns et al. | We introduce a benchmark dataset for question answering and translation in bilingual Latin and English settings, containing about 7,800 question-answer pairs. The questions are drawn from Latin pedagogical sources, including exams, quizbowl-style trivia, and textbooks ranging from the 1800s to the present. After automated extraction, cleaning, and manual review, the dataset covers a diverse range of question types: knowledge- and skill-based, multihop reasoning, constrained translation, and mixed language pairs. To our knowledge, this is the first QA benchmark centered on Latin. As a case study, we evaluate three large language models -- LLaMa 3, Qwen QwQ, and OpenAI's o3-mini -- finding that all perform worse on skill-oriented questions. Although the reasoning models perform better on scansion and literary-device tasks, they offer limited improvement overall. QwQ performs slightly better on questions asked in Latin, but LLaMa3 and o3-mini are more task dependent. This dataset provides a new resource for assessing model capabilities in a specialized linguistic and cultural domain, and the creation process can be easily adapted for other languages. The dataset is available at: https://github.com/slanglab/RespondeoQA |
| 2026-04-22 | [Interval POMDP Shielding for Imperfect-Perception Agents](http://arxiv.org/abs/2604.20728v1) | William Scarbro, Ravi Mangal | Autonomous systems that rely on learned perception can make unsafe decisions when sensor readings are misclassified. We study shielding for this setting: given a proposed action, a shield blocks actions that could violate safety. We consider the common case where system dynamics are known but perception uncertainty must be estimated from finite labeled data. From these data we build confidence intervals for the probabilities of perception outcomes and use them to model the system as a finite Interval Partially Observable Markov Decision Process with discrete states and actions. We then propose an algorithm to compute a conservative set of beliefs over the underlying state that is consistent with the observations seen so far.   This enables us to construct a runtime shield that comes with a finite-horizon guarantee: with high probability over the training data, if the true perception uncertainty rates lie within the learned intervals, then every action admitted by the shield satisfies a stated lower bound on safety. Experiments on four case studies show that our shielding approach (and variants derived from it) improves the safety of the system over state-of-the-art baselines. |
| 2026-04-22 | [SoK: The Next Frontier in AV Security: Systematizing Perception Attacks and the Emerging Threat of Multi-Sensor Fusion](http://arxiv.org/abs/2604.20621v1) | Shahriar Rahman Khan, Tariqul Islam et al. | Autonomous vehicles (AVs) increasingly rely on multi-sensor perception pipelines that combine data from cameras, lidar, radar, and other modalities to interpret the environment. This SoK systematizes 48 peer-reviewed studies on perception-layer attacks against AVs, tracking the field's evolution from single-sensor exploits to complex cross-modal threats that compromise multi-sensor fusion (MSF). We develop a unified taxonomy of 20 attack vectors organized by sensor type, attack stage, medium, and perception module, revealing patterns that expose underexplored vulnerabilities in fusion logic and cross-sensor dependencies. Our analysis identifies key research gaps, including limited real-world testing, short-term evaluation bias, and the absence of defenses that account for inter-sensor consistency. To illustrate one such gap, we validate a fusion-level vulnerability through a proof-of-concept simulation combining infrared and lidar spoofing. The findings highlight a fundamental shift in AV security: as systems fuse more sensors for robustness, attackers exploit the very redundancy meant to ensure safety. We conclude with directions for fusion-aware defense design and a research agenda for trustworthy perception in autonomous systems. |
| 2026-04-22 | [Evaluating Assurance Cases as Text-Attributed Graphs for Structure and Provenance Analysis](http://arxiv.org/abs/2604.20577v1) | Fariz Ikhwantri, Dusica Marijan | An assurance case is a structured argument document that justifies claims about a system's requirements or properties, which are supported by evidence. In regulated domains, these are crucial for meeting compliance and safety requirements to industry standards. We propose a graph diagnostic framework for analysing the structure and provenance of assurance cases. We focus on two main tasks: (1) link prediction, to learn and identify connections between argument elements, and (2) graph classification, to differentiate between assurance cases created by a state-of-the-art large language model and those created by humans, aiming to detect bias. We compiled a publicly available dataset of assurance cases, represented as graphs with nodes and edges, supporting both link prediction and provenance analysis. Experiments show that graph neural networks (GNNs) achieve strong link prediction performance (ROC-AUC 0.760) on real assurance cases and generalise well across domains and semi-supervised settings. For provenance detection, GNNs effectively distinguish human-authored from LLM-generated cases (F1 0.94). We observed that LLM-generated assurance cases have different hierarchical linking patterns compared to human-authored cases. Furthermore, existing GNN explanation methods show only moderate faithfulness, revealing a gap between predicted reasoning and the true argument structure. |
| 2026-04-22 | [PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting](http://arxiv.org/abs/2604.20576v1) | Jumin Kim, Seungmin Baek et al. | As DRAM scaling exacerbates RowHammer, DDR5 introduces per-row activation counting (PRAC) to track aggressor activity. However, PRAC indiscriminately increments counters on every activation -- including benign refreshes -- while relying solely on explicit RFM operations for resets. Consequently, counters saturate even in an idle bank, triggering cascading mitigations and degrading performance. This vulnerability arises from a fundamental mismatch: PRAC tracks the aggressor but aims to protect the victim.   We present Per-Victim-row hAmmered Counting (PVAC), a victim-based counting mechanism that aligns the counter semantics with the physical disturbance mechanism of RowHammer. PVAC increments the counters of victim rows, resets the activated row, and naturally bounds counter values under normal refresh. To enable efficient victim-based updates, PVAC employs a dedicated counter subarray (CSA) that performs all counter resets and increments concurrently with normal accesses, without timing overhead. We further devise an energy-efficient CSA layout that minimizes refresh-induced counter accesses. Through victim-based counting, PVAC supports higher hammering tolerance than PRAC while maintaining the same worst-case safety guarantee. Across benign workloads and adversarial attack patterns, PVAC avoids spurious Alerts, eliminates PRAC timing penalties, and achieves higher performance and lower energy consumption than prior PRAC-based defenses. |
| 2026-04-22 | [Multi-Objective RIS Deployment Optimization for Physical Layer Security in ISAC Networks](http://arxiv.org/abs/2604.20537v1) | Wenqing Dai, Jan Herbst et al. | Reconfigurable Intelligent Surfaces (RIS) have emerged as a key enabler for programmable wireless environments in future Beyond-5G (B5G) and 6G networks. In the meantime, Integrated Sensing and Communication (ISAC) and Physical-Layer Security (PLS) are becoming essential functionalities for next-generation wireless systems, particularly in safety and mission-critical applications. However, jointly optimizing RIS-assisted systems to support communication, sensing, and security introduces complex and often conflicting design trade-offs. In this work, a multi-objective optimization framework for RIS-assisted networks is proposed, aiming to jointly analyze communication performance, sensing accuracy, and security-related channel properties in a unified system perspective. The proposed model jointly considers RIS deployment location, orientation, surface size, and an ISAC configuration weight that controls the allocation of RIS reflection gain between communication and sensing tasks. Simulation results reveal inherent trade-offs among communication reliability, sensing accuracy, and security performance. The proposed framework provides valuable insights into the interplay between communication, sensing, and security, and enables the design of efficient RIS deployment and configuration strategies for secure ISAC-enabled 6G wireless networks. |
| 2026-04-22 | [Bounding Transient Instability in Sensor Data Injected Nonlinear Stochastic Flight Dynamics](http://arxiv.org/abs/2604.20517v1) | Surya Ratna Prakash D, Soumyendu Raha | Transient instability in nonlinear stochastic dynamical systems is a fundamental limitation in safety-critical aerospace applications, particularly during powered descent and landing where failure is driven by finite-time excursions rather than asymptotic divergence. Classical notions of mean-square or asymptotic stability are therefore insufficient for certification and design. This paper develops a logarithmic-norm-based framework for finite-time transient stability analysis of nonlinear Ito stochastic differential equations. The approach extends matrix measures to nonlinear mappings in a Lipschitz sense, enabling efficient characterization of instantaneous perturbation growth without local linearization. Using Ito calculus, bounds on the mean and variance of transient growth are derived, providing conditions for non-positive finite-time mean growth and probabilistic bounds on instability events. The analysis highlights a key distinction between mean and sample-path behavior, showing that stability in expectation does not guarantee pathwise finite-time safety, and that almost-sure transient stability cannot generally be ensured under stochastic diffusion. The framework is extended to data-constrained stochastic dynamics in navigation and estimation, revealing a trade-off between estimation consistency and transient robustness due to continuous data injection. Demonstrations with flight-like lunar lander telemetry show that similar mean trajectories can exhibit significantly different transient stability behaviour, and that mission failure correlates with accumulation of transient instability over short critical intervals. These results motivate probabilistic finite-time stability metrics for safety-critical autonomous systems. |
| 2026-04-22 | [Mythos and the Unverified Cage: Z3-Based Pre-Deployment Verification for Frontier-Model Sandbox Infrastructure](http://arxiv.org/abs/2604.20496v1) | Dominik Blain | The April 2026 Claude Mythos sandbox escape exposed a critical weakness in frontier AI containment: the infrastructure surrounding advanced models remains susceptible to formally characterizable arithmetic vulnerabilities. Anthropic has not publicly characterized the escape vector; some secondary accounts hypothesize a CWE-190 arithmetic vulnerability in sandbox networking code. We treat this as unverified and analyze the vulnerability class rather than the specific escape. This paper presents COBALT, a Z3 SMT-based formal verification engine for identifying CWE-190/191/195 arithmetic vulnerability patterns in C/C++ infrastructure prior to deployment.   We distinguish two classes of contribution. Validated: COBALT detects arithmetic vulnerability patterns in production codebases, producing SAT verdicts with concrete witnesses and UNSAT guarantees under explicit safety bounds. We demonstrate this on four production case studies: NASA cFE, wolfSSL, Eclipse Mosquitto, and NASA F Prime, with reproducible encodings, verified solver output, and acknowledged security outcomes. Proposed: a four-layer containment framework consisting of COBALT, VERDICT, DIRECTIVE-4, and SENTINEL, mapping pre-deployment verification, pre-execution constraints, output control, and runtime monitoring to the failure modes exposed by the Mythos incident.   Under explicit assumptions, we further argue that the publicly reported Mythos escape class is consistent with a Z3-expressible CWE-190 arithmetic formulation and that pre-deployment formal analysis would have been capable of surfacing the relevant pattern. The broader claim is infrastructural: frontier-model safety cannot depend on behavioral safeguards alone; the containment stack itself must be subjected to formal verification. |
| 2026-04-22 | [CCTVBench: Contrastive Consistency Traffic VideoQA Benchmark for Multimodal LLMs](http://arxiv.org/abs/2604.20460v1) | Xingcheng Zhou, Hao Guo et al. | Safety-critical traffic reasoning requires contrastive consistency: models must detect true hazards when an accident occurs, and reliably reject plausible-but-false hypotheses under near-identical counterfactual scenes. We present CCTVBench, a Contrastive Consistency Traffic VideoQA Benchmark built on paired real accident videos and world-model-generated counterfactual counterparts, together with minimally different, mutually exclusive hypothesis questions. CCTVBench enforces a single structured decision pattern over each video question quadruple and provides actionable diagnostics that decompose failures into positive omission, positive swap, negative hallucination, and mutual-exclusivity violation, while separating video versus question consistency. Experiments across open-source and proprietary video LLMs reveal a large and persistent gap between standard per-instance QA metrics and quadruple-level contrastive consistency, with unreliable none-of-the-above rejection as a key bottleneck. Finally, we introduce C-TCD, a contrastive decoding approach leveraging a semantically exclusive counterpart video as the contrast input at inference time, improving both instance-level QA and contrastive consistency. |
| 2026-04-22 | [MedSkillAudit: A Domain-Specific Audit Framework for Medical Research Agent Skills](http://arxiv.org/abs/2604.20441v1) | Yingyong Hou, Xinyuan Lao et al. | Background: Agent skills are increasingly deployed as modular, reusable capability units in AI agent systems. Medical research agent skills require safeguards beyond general-purpose evaluation, including scientific integrity, methodological validity, reproducibility, and boundary safety. This study developed and preliminarily evaluated a domain-specific audit framework for medical research agent skills, with a focus on reliability against expert review. Methods: We developed MedSkillAudit (skill-auditor@1.0), a layered framework assessing skill release readiness before deployment. We evaluated 75 skills across five medical research categories (15 per category). Two experts independently assigned a quality score (0-100), an ordinal release disposition (Production Ready / Limited Release / Beta Only / Reject), and a high-risk failure flag. System-expert agreement was quantified using ICC(2,1) and linearly weighted Cohen's kappa, benchmarked against the human inter-rater baseline. Results: The mean consensus quality score was 72.4 (SD = 13.0); 57.3% of skills fell below the Limited Release threshold. MedSkillAudit achieved ICC(2,1) = 0.449 (95% CI: 0.250-0.610), exceeding the human inter-rater ICC of 0.300. System-consensus score divergence (SD = 9.5) was smaller than inter-expert divergence (SD = 12.4), with no directional bias (Wilcoxon p = 0.613). Protocol Design showed the strongest category-level agreement (ICC = 0.551); Academic Writing showed a negative ICC (-0.567), reflecting a structural rubric-expert mismatch. Conclusions: Domain-specific pre-deployment audit may provide a practical foundation for governing medical research agent skills, complementing general-purpose quality checks with structured audit workflows tailored to scientific use cases. |
| 2026-04-22 | [Phonon driven non-equilibrium triggers for thermal runaway in battery electrodes](http://arxiv.org/abs/2604.20424v1) | Harry Mclean, Francis Huw Davies et al. | Thermal runaway in lithium-ion batteries is governed by the poorly-understood initiation phase, where localised heating introduces instability. Here we identify the three key components that trigger thermal runaway, decreases in local conductivity, heat capacity changes, and intercalation heating, which significantly increase temperature gradients that accelerate battery degradation. Using a multiscale framework that links atomistic phonon calculations with grain-resolved thermal modelling, we identify large thermal gradients across grain boundaries arising from external heating events and intercalation-dependent thermal properties of Li$_x$ZrS$_2$. The observed changes in thermal conductivity are due to charge redistribution and bond-strength modulation of the host, in contrast to the existing theory of lithium rattler mechanics. Internal heating events driven by intercalation gives rise to local thermal gradients, finite-speed thermal wave interference, and internal thermal fluctuations that generate mechanical strain and sub-grain thermal breakdown. These results show that the trigger for thermal runaway is controlled by internal grain architecture and composition, as well as the external environment. Our findings establish materials and electrode design rules for suppressing hotspot formation and improving battery safety during fast charging. |
| 2026-04-22 | [OVPD: A Virtual-Physical Fusion Testing Dataset of OnSite Auton-omous Driving Challenge](http://arxiv.org/abs/2604.20423v1) | Yuhang Zhang, Jiarui Zhang et al. | The rapid iteration of autonomous driving algorithms has created a growing demand for high-fidelity, replayable, and diagnosable testing data. However, many public datasets lack real vehicle dynamics feedback and closed-loop interaction with surrounding traffic and road infrastructure, limiting their ability to reflect deployment readiness. To address this gap, we present OVPD (OnSite Virtual-Physical Dataset), a virtual-physical fusion testing dataset released from the 2025 OnSite Autonomous Driving Challenge. Centered on real-vehicle-in-the-loop testing, OVPD integrates virtual background traffic with vehicle-infrastructure perception to build controllable and interactive closed-loop test environments on a proving ground. The dataset contains 20 testing clips from 20 teams over a scenario chain of 15 atomic scenarios, totaling nearly 3 hours of multi-modal data, including vehicle trajectories and states, control commands, and digital-twin-rendered surround-view observations. OVPD supports long-tail planning and decision-making validation, open-loop or platform-enabled closed-loop evaluation, and comprehensive assessment across safety, efficiency, comfort, rule compliance, and traffic impact, providing actionable evidence for failure diagnosis and iterative improvement. The dataset is available via: https://huggingface.co/datasets/Yuhang253820/Onsite_OPVD |
| 2026-04-22 | [WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning](http://arxiv.org/abs/2604.20398v1) | Juyong Jiang, Chenglin Cai et al. | While Large Language Models (LLMs) excel at function-level code generation, project-level tasks such as generating functional and visually aesthetic multi-page websites remain highly challenging. Existing works are often limited to single-page static websites, while agentic frameworks typically rely on multi-turn execution with proprietary models, leading to substantial token costs, high latency, and brittle integration. Training a small LLM end-to-end with reinforcement learning (RL) is a promising alternative, yet it faces a critical bottleneck in designing reliable and computationally feasible rewards for website generation. Unlike single-file coding tasks that can be verified by unit tests, website generation requires evaluating inherently subjective aesthetics, cross-page interactions, and functional correctness. To this end, we propose WebGen-R1, an end-to-end RL framework tailored for project-level website generation. We first introduce a scaffold-driven structured generation paradigm that constrains the large open-ended action space and preserves architectural integrity. We then design a novel cascaded multimodal reward that seamlessly couples structural guarantees with execution-grounded functional feedback and vision-based aesthetic supervision. Extensive experiments demonstrate that our WebGen-R1 substantially transforms a 7B base model from generating nearly nonfunctional websites into producing deployable, aesthetically aligned multi-page websites. Remarkably, our WebGen-R1 not only consistently outperforms heavily scaled open-source models (up to 72B), but also rivals the state-of-the-art DeepSeek-R1 (671B) in functional success, while substantially exceeding it in valid rendering and aesthetic alignment. These results position WebGen-R1 as a viable path for scaling small open models from function-level code generation to project-level web application generation. |
| 2026-04-22 | [Graph2Counsel: Clinically Grounded Synthetic Counseling Dialogue Generation from Client Psychological Graphs](http://arxiv.org/abs/2604.20382v1) | Aishik Mandal, Hiba Arnaout et al. | Rising demand for mental health support has increased interest in using Large Language Models (LLMs) for counseling. However, adapting LLMs to this high-risk safety-critical domain is hindered by the scarcity of real-world counseling data due to privacy constraints. Synthetic datasets provide a promising alternative, but existing approaches often rely on unstructured or semi-structured text inputs and overlook structural dependencies between a client's cognitive, emotional, and behavioral states, often producing psychologically inconsistent interactions and reducing data realism and quality. We introduce Graph2Counsel, a framework for generating synthetic counseling sessions grounded in Client Psychological Graphs (CPGs) that encode relationships among clients' thoughts, emotions, and behaviors. Graph2Counsel employs a structured prompting pipeline guided by counselor strategies and CPG, and explores prompting strategies including CoT (Wei et al., 2022) and Multi-Agent Feedback (Li et al., 2025a). Graph2Counsel produces 760 sessions from 76 CPGs across diverse client profiles. In expert evaluation, our dataset outperforms prior datasets on specificity, counselor competence, authenticity, conversational flow, and safety, with substantial inter-annotator agreement (Krippendorff's $α$ = 0.70). Fine-tuning an open-source model on this dataset improves performance on CounselingBench (Nguyen et al., 2025) and CounselBench (Li et al., 2025b), showing downstream utility. We also make our code and data public. |
| 2026-04-22 | [A Vision-Language-Action Model for Adaptive Ultrasound-Guided Needle Insertion and Needle Tracking](http://arxiv.org/abs/2604.20347v1) | Yuelin Zhang, Qingpeng Ding et al. | Ultrasound (US)-guided needle insertion is a critical yet challenging procedure due to dynamic imaging conditions and difficulties in needle visualization. Many methods have been proposed for automated needle insertion, but they often rely on hand-crafted pipelines with modular controllers, whose performance degrades in challenging cases. In this paper, a Vision-Language-Action (VLA) model is proposed for adaptive and automated US-guided needle insertion and tracking on a robotic ultrasound (RUS) system. This framework provides a unified approach to needle tracking and needle insertion control, enabling real-time, dynamically adaptive adjustment of insertion based on the obtained needle position and environment awareness. To achieve real-time and end-to-end tracking, a Cross-Depth Fusion (CDF) tracking head is proposed, integrating shallow positional and deep semantic features from the large-scale vision backbone. To adapt the pretrained vision backbone for tracking tasks, a Tracking-Conditioning (TraCon) register is introduced for parameter-efficient feature conditioning. After needle tracking, an uncertainty-aware control policy and an asynchronous VLA pipeline are presented for adaptive needle insertion control, ensuring timely decision-making for improved safety and outcomes. Extensive experiments on both needle tracking and insertion show that our method consistently outperforms state-of-the-art trackers and manual operation, achieving higher tracking accuracy, improved insertion success rates, and reduced procedure time, highlighting promising directions for RUS-based intelligent intervention. |
| 2026-04-22 | [FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory](http://arxiv.org/abs/2604.20300v1) | Yingjie Gu, Bo Xiong et al. | For LLM agents, memory management critically impacts efficiency, quality, and security. While much research focuses on retention, selective forgetting--inspired by human cognitive processes (hippocampal indexing/consolidation theory and Ebbinghaus forgetting curve)--remains underexplored. We argue that in resource-constrained environments, a well-designed forgetting mechanism is as crucial as remembering, delivering benefits across three dimensions: (1) efficiency via intelligent memory pruning, (2) quality by dynamically updating outdated preferences and context, and (3) security through active forgetting of malicious inputs, sensitive data, and privacy-compromising content. Our framework establishes a taxonomy of forgetting mechanisms: passive decay-based, active deletion-based, safety-triggered, and adaptive reinforcement-based. Building on advances in LLM agent architectures and vector databases, we present detailed specifications, implementation strategies, and empirical validation from controlled experiments. Results show significant improvements: access efficiency (+8.49%), content quality (+29.2% signal-to-noise ratio), and security performance (100% elimination of security risks). Our work bridges cognitive neuroscience and AI systems, offering practical solutions for real-world deployment while addressing ethical and regulatory compliance. The paper concludes with challenges and future directions, establishing selective forgetting as a fundamental capability for next-generation LLM agents operating in real-world, resource-constrained scenarios. Our contributions align with AI-native memory systems and responsible AI development. |
| 2026-04-22 | [Generative Augmentation of Imbalanced Flight Records for Flight Diversion Prediction: A Multi-objective Optimisation Framework](http://arxiv.org/abs/2604.20288v1) | Karim Aly, Alexei Sharpanskykh et al. | Flight diversions are rare but high-impact events in aviation, making their reliable prediction vital for both safety and operational efficiency. However, their scarcity in historical records impedes the training of machine learning models utilised to predict them. This study addresses this scarcity gap by investigating how generative models can augment historical flight data with synthetic diversion records to enhance model training and improve predictive accuracy. We propose a multi-objective optimisation framework coupled with automated hyperparameter search to identify optimal configurations for three deep generative models: Tabular Variational Autoencoder (TVAE), Conditional Tabular Generative Adversarial Network (CTGAN), and CopulaGAN, with the Gaussian Copula (GC) model serving as a statistical baseline. The quality of the synthetic data was examined through a six-stage evaluation framework encompassing realism, diversity, operational validity, statistical similarity, fidelity, and predictive utility. Results show that the optimised models significantly outperform their non-optimised counterparts, and that synthetic augmentation substantially improves diversion prediction compared to models trained solely on real data. These findings demonstrate the effectiveness of hyperparameter-optimised generative models for advancing predictive modelling of rare events in air transportation. |

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



