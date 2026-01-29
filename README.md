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
| 2026-01-28 | [Reward Models Inherit Value Biases from Pretraining](http://arxiv.org/abs/2601.20838v1) | Brian Christian, Jessica A. F. Thompson et al. | Reward models (RMs) are central to aligning large language models (LLMs) with human values but have received less attention than pre-trained and post-trained LLMs themselves. Because RMs are initialized from LLMs, they inherit representations that shape their behavior, but the nature and extent of this influence remain understudied. In a comprehensive study of 10 leading open-weight RMs using validated psycholinguistic corpora, we show that RMs exhibit significant differences along multiple dimensions of human value as a function of their base model. Using the "Big Two" psychological axes, we show a robust preference of Llama RMs for "agency" and a corresponding robust preference of Gemma RMs for "communion." This phenomenon holds even when the preference data and finetuning process are identical, and we trace it back to the logits of the respective instruction-tuned and pre-trained models. These log-probability differences themselves can be formulated as an implicit RM; we derive usable implicit reward scores and show that they exhibit the very same agency/communion difference. We run experiments training RMs with ablations for preference data source and quantity, which demonstrate that this effect is not only repeatable but surprisingly durable. Despite RMs being designed to represent human preferences, our evidence shows that their outputs are influenced by the pretrained LLMs on which they are based. This work underscores the importance of safety and alignment efforts at the pretraining stage, and makes clear that open-source developers' choice of base model is as much a consideration of values as of performance. |
| 2026-01-28 | [Learning Contextual Runtime Monitors for Safe AI-Based Autonomy](http://arxiv.org/abs/2601.20666v1) | Alejandro Luque-Cerpa, Mengyuan Wang et al. | We introduce a novel framework for learning context-aware runtime monitors for AI-based control ensembles. Machine-learning (ML) controllers are increasingly deployed in (autonomous) cyber-physical systems because of their ability to solve complex decision-making tasks. However, their accuracy can degrade sharply in unfamiliar environments, creating significant safety concerns. Traditional ensemble methods aim to improve robustness by averaging or voting across multiple controllers, yet this often dilutes the specialized strengths that individual controllers exhibit in different operating contexts. We argue that, rather than blending controller outputs, a monitoring framework should identify and exploit these contextual strengths. In this paper, we reformulate the design of safe AI-based control ensembles as a contextual monitoring problem. A monitor continuously observes the system's context and selects the controller best suited to the current conditions. To achieve this, we cast monitor learning as a contextual learning task and draw on techniques from contextual multi-armed bandits. Our approach comes with two key benefits: (1) theoretical safety guarantees during controller selection, and (2) improved utilization of controller diversity. We validate our framework in two simulated autonomous driving scenarios, demonstrating significant improvements in both safety and performance compared to non-contextual baselines. |
| 2026-01-28 | [Dependable Connectivity for Industrial Wireless Communication Networks](http://arxiv.org/abs/2601.20580v1) | Nurul Huda Mahmood, Onel L. A. Lopez et al. | Dependability - a system's ability to consistently provide reliable services by ensuring safety and maintainability in the face of internal or external disruptions - is a fundamental requirement for industrial wireless communication networks (IWCNs). While 5G ultra-reliable low-latency communication (URLLC) addresses some aspects of this challenge, its evolution toward holistic dependability in 6G must encompass reliability, availability, safety, and security. This paper provides a comprehensive framework for dependable IWCNs, bridging theory and practice. We first establish the theoretical foundations of dependability, including outlining its key attributes and presenting analytical tools to study it. Next, we explore practical enablers, such as adaptive multiple access schemes leveraging real-time monitoring and time-sensitive networking to ensure end-to-end determinism. A case study demonstrates how intelligent wake-up protocols improve event detection probability by orders of magnitude compared to conventional duty cycling. Finally, we outline open challenges and future directions for a 6G-driven dependable IWCN. |
| 2026-01-28 | [Reinforcement Unlearning via Group Relative Policy Optimization](http://arxiv.org/abs/2601.20568v1) | Efstratios Zaradoukas, Bardh Prenkaj et al. | During pretraining, LLMs inadvertently memorize sensitive or copyrighted data, posing significant compliance challenges under legal frameworks like the GDPR and the EU AI Act. Fulfilling these mandates demands techniques that can remove information from a deployed model without retraining from scratch. Existing unlearning approaches attempt to address this need, but often leak the very data they aim to erase, sacrifice fluency and robustness, or depend on costly external reward models. We introduce PURGE (Policy Unlearning through Relative Group Erasure), a novel method grounded in the Group Relative Policy Optimization framework that formulates unlearning as a verifiable problem. PURGE uses an intrinsic reward signal that penalizes any mention of forbidden concepts, allowing safe and consistent unlearning. Our approach reduces token usage per target by up to a factor of 46 compared with SotA methods, while improving fluency by 5.48 percent and adversarial robustness by 12.02 percent over the base model. On the Real World Knowledge Unlearning (RWKU) benchmark, PURGE achieves 11 percent unlearning effectiveness while preserving 98 percent of original utility. PURGE shows that framing LLM unlearning as a verifiable task, enables more reliable, efficient, and scalable forgetting, suggesting a promising new direction for unlearning research that combines theoretical guarantees, improved safety, and practical deployment efficiency. |
| 2026-01-28 | [Interpreting Emergent Extreme Events in Multi-Agent Systems](http://arxiv.org/abs/2601.20538v1) | Ling Tang, Jilin Mei et al. | Large language model-powered multi-agent systems have emerged as powerful tools for simulating complex human-like systems. The interactions within these systems often lead to extreme events whose origins remain obscured by the black box of emergence. Interpreting these events is critical for system safety. This paper proposes the first framework for explaining emergent extreme events in multi-agent systems, aiming to answer three fundamental questions: When does the event originate? Who drives it? And what behaviors contribute to it? Specifically, we adapt the Shapley value to faithfully attribute the occurrence of extreme events to each action taken by agents at different time steps, i.e., assigning an attribution score to the action to measure its influence on the event. We then aggregate the attribution scores along the dimensions of time, agent, and behavior to quantify the risk contribution of each dimension. Finally, we design a set of metrics based on these contribution scores to characterize the features of extreme events. Experiments across diverse multi-agent system scenarios (economic, financial, and social) demonstrate the effectiveness of our framework and provide general insights into the emergence of extreme phenomena. |
| 2026-01-28 | [Challenges in Android Data Disclosure: An Empirical Study](http://arxiv.org/abs/2601.20459v1) | Mugdha Khedkar, Michael Schlichtig et al. | Current legal frameworks enforce that Android developers accurately report the data their apps collect. However, large codebases can make this reporting challenging. This paper employs an empirical approach to understand developers' experience with Google Play Store's Data Safety Section (DSS) form.   We first survey 41 Android developers to understand how they categorize privacy-related data into DSS categories and how confident they feel when completing the DSS form. To gain a broader and more detailed view of the challenges developers encounter during the process, we complement the survey with an analysis of 172 online developer discussions, capturing the perspectives of 642 additional developers. Together, these two data sources represent insights from 683 developers.   Our findings reveal that developers often manually classify the privacy-related data their apps collect into the data categories defined by Google-or, in some cases, omit classification entirely-and rely heavily on existing online resources when completing the form. Moreover, developers are generally confident in recognizing the data their apps collect, yet they lack confidence in translating this knowledge into DSS-compliant disclosures. Key challenges include issues in identifying privacy-relevant data to complete the form, limited understanding of the form, and concerns about app rejection due to discrepancies with Google's privacy requirements.   These results underscore the need for clearer guidance and more accessible tooling to support developers in meeting privacy-aware reporting obligations. |
| 2026-01-28 | [Reducing End-to-End Latency of Cause-Effect Chains with Shared Cache Analysis](http://arxiv.org/abs/2601.20427v1) | Yixuan Zhu, Yinkang Gao et al. | Cause-effect chains, as a widely used modeling method in real-time embedded systems, are extensively applied in various safety-critical domains. End-to-end latency, as a key real-time attribute of cause-effect chains, is crucial in many applications. But the analysis of end-to-end latency for cause-effect chains on multicore platforms with shared caches still presents an unresolved issue. Traditional methods typically assume that the worst-case execution time (WCET) of each task in the cause-effect chain is known. However, in the absence of scheduling information, these methods often assume that all shared cache accesses result in misses, leading to an overestimation of WCET and, consequently, affecting the accuracy of end-to-end latency. However, effectively integrating scheduling information into the WCET analysis process of the chains may introduce two challenges: first, how to leverage the structural characteristics of the chains to optimize shared cache analysis, and second, how to improve analysis accuracy while avoiding state space explosion.   To address these issues, this paper proposes a novel end-to-end latency analysis framework designed for multi-chain systems on multicore platforms with shared caches. This framework extracts scheduling information and structural characteristics of cause-effect chains, constructing fine-grained and scalable inter-core memory access contexts at the basic block level for time-sensitive shared cache analysis. This results in more accurate WCET (TSC-WCET) estimates, which are then used to derive the end-to-end latency. Finally, we conduct experiments on dual-core and quad-core systems with various cache configurations, which show that under certain settings, the average maximum end-to-end latency of cause-effect chains is reduced by up to 34% and 26%. |
| 2026-01-28 | [Unsupervised Anomaly Detection in Multi-Agent Trajectory Prediction via Transformer-Based Models](http://arxiv.org/abs/2601.20367v1) | Qing Lyu, Zhe Fu et al. | Identifying safety-critical scenarios is essential for autonomous driving, but the rarity of such events makes supervised labeling impractical. Traditional rule-based metrics like Time-to-Collision are too simplistic to capture complex interaction risks, and existing methods lack a systematic way to verify whether statistical anomalies truly reflect physical danger. To address this gap, we propose an unsupervised anomaly detection framework based on a multi-agent Transformer that models normal driving and measures deviations through prediction residuals. A dual evaluation scheme has been proposed to assess both detection stability and physical alignment: Stability is measured using standard ranking metrics in which Kendall Rank Correlation Coefficient captures rank agreement and Jaccard index captures the consistency of the top-K selected items; Physical alignment is assessed through correlations with established Surrogate Safety Measures (SSM). Experiments on the NGSIM dataset demonstrate our framework's effectiveness: We show that the maximum residual aggregator achieves the highest physical alignment while maintaining stability. Furthermore, our framework identifies 388 unique anomalies missed by Time-to-Collision and statistical baselines, capturing subtle multi-agent risks like reactive braking under lateral drift. The detected anomalies are further clustered into four interpretable risk types, offering actionable insights for simulation and testing. |
| 2026-01-28 | [Dual-Modality IoT Framework for Integrated Access Control and Environmental Safety Monitoring with Real-Time Cloud Analytics](http://arxiv.org/abs/2601.20366v1) | Abdul Hasib, A. S. M. Ahsanul Sarkar Akib et al. | The integration of physical security systems with environmental safety monitoring represents a critical advancement in smart infrastructure management. Traditional approaches maintain these systems as independent silos, creating operational inefficiencies, delayed emergency responses, and increased management complexity. This paper presents a comprehensive dual-modality Internet of Things framework that seamlessly integrates RFID-based access control with multi-sensor environmental safety monitoring through a unified cloud architecture. The system comprises two coordinated subsystems: Subsystem 1 implements RFID authentication with servo-actuated gate control and real-time Google Sheets logging, while Subsystem 2 provides comprehensive safety monitoring incorporating flame detection, water flow measurement, LCD status display, and personnel identification. Both subsystems utilize ESP32 microcontrollers for edge processing and wireless connectivity. Experimental evaluation over 45 days demonstrates exceptional performance metrics: 99.2\% RFID authentication accuracy with 0.82-second average response time, 98.5\% flame detection reliability within 5-meter range, and 99.8\% cloud data logging success rate. The system maintains operational integrity during network disruptions through intelligent local caching mechanisms and achieves total implementation cost of 5,400 BDT (approximately \$48), representing an 82\% reduction compared to commercial integrated solutions. This research establishes a practical framework for synergistic security-safety integration, demonstrating that professional-grade performance can be achieved through careful architectural design and component optimization while maintaining exceptional cost-effectiveness and accessibility for diverse application scenarios. |
| 2026-01-28 | [Beyond Speedup -- Utilizing KV Cache for Sampling and Reasoning](http://arxiv.org/abs/2601.20326v1) | Zeyu Xing, Xing Li et al. | KV caches, typically used only to speed up autoregressive decoding, encode contextual information that can be reused for downstream tasks at no extra cost. We propose treating the KV cache as a lightweight representation, eliminating the need to recompute or store full hidden states. Despite being weaker than dedicated embeddings, KV-derived representations are shown to be sufficient for two key applications: \textbf{(i) Chain-of-Embedding}, where they achieve competitive or superior performance on Llama-3.1-8B-Instruct and Qwen2-7B-Instruct; and \textbf{(ii) Fast/Slow Thinking Switching}, where they enable adaptive reasoning on Qwen3-8B and DeepSeek-R1-Distil-Qwen-14B, reducing token generation by up to $5.7\times$ with minimal accuracy loss. Our findings establish KV caches as a free, effective substrate for sampling and reasoning, opening new directions for representation reuse in LLM inference. Code: https://github.com/cmd2001/ICLR2026_KV-Embedding. |
| 2026-01-28 | [Neural Cooperative Reach-While-Avoid Certificates for Interconnected Systems](http://arxiv.org/abs/2601.20324v1) | Jingyuan Zhou, Haoze Wu et al. | Providing formal guarantees for neural network-based controllers in large-scale interconnected systems remains a fundamental challenge. In particular, using neural certificates to capture cooperative interactions and verifying these certificates at scale is crucial for the safe deployment of such controllers. However, existing approaches fall short on both fronts. To address these limitations, we propose neural cooperative reach-while-avoid certificates with Dynamic-Localized Vector Control Lyapunov and Barrier Functions, which capture cooperative dynamics through state-dependent neighborhood structures and provide decentralized certificates for global exponential stability and safety. Based on the certificates, we further develop a scalable training and verification framework that jointly synthesizes controllers and neural certificates via a constrained optimization objective, and leverages a sufficient condition to ensure formal guarantees considering modeling error. To improve scalability, we introduce a structural reuse mechanism to transfer controllers and certificates between substructure-isomorphic systems. The proposed methodology is validated with extensive experiments on multi-robot coordination and vehicle platoons. Results demonstrate that our framework ensures certified cooperative reach-while-avoid while maintaining strong control performance. |
| 2026-01-28 | [A Data-Driven Krasovskii-Based Approach for Safety Controller Design of Time-Delayed Uncertain Polynomial Systems](http://arxiv.org/abs/2601.20298v1) | Omid Akbarzadeh, MohammadHossein Ashoori et al. | We develop a data-driven framework for the synthesis of robust Krasovskii control barrier certificates (RK-CBC) and corresponding robust safety controllers (R-SC) for discrete-time input-affine uncertain polynomial systems with unknown dynamics, while explicitly accounting for unknown-but-bounded disturbances and time-invariant delays using only observed input-state data. Although control barrier certificates have been extensively studied for safety analysis of control systems, existing work on unknown systems with time delays, particularly in the presence of disturbances, remains limited. The challenge of safety synthesis for such systems stems from two main factors: first, the system's mathematical model is unavailable; and second, the safety conditions should explicitly incorporate the effects of time delays on system evolution during the synthesis process, while remaining robust to unknown disturbances. To address these challenges, we develop a data-driven framework based on Krasovskii control barrier certificates, extending the classical CBC formulation for delay-free systems to explicitly account for time delays by aggregating delayed components within the barrier construction. The proposed framework relies solely on input-state data collected over a finite time horizon, enabling the direct synthesis of RK-CBC and R-SC from observed trajectories without requiring an explicit system model. The synthesis is cast as a data-driven sum-of-squares (SOS) optimization program, yielding a structured design methodology. As a result, robust safety is guaranteed in the presence of unknown disturbances and time delays over an infinite time horizon. The effectiveness of the proposed method is demonstrated through three case studies, including two physical systems. |
| 2026-01-28 | [SoftHateBench: Evaluating Moderation Models Against Reasoning-Driven, Policy-Compliant Hostility](http://arxiv.org/abs/2601.20256v1) | Xuanyu Su, Diana Inkpen et al. | Online hate on social media ranges from overt slurs and threats (\emph{hard hate speech}) to \emph{soft hate speech}: discourse that appears reasonable on the surface but uses framing and value-based arguments to steer audiences toward blaming or excluding a target group. We hypothesize that current moderation systems, largely optimized for surface toxicity cues, are not robust to this reasoning-driven hostility, yet existing benchmarks do not measure this gap systematically. We introduce \textbf{\textsc{SoftHateBench}}, a generative benchmark that produces soft-hate variants while preserving the underlying hostile standpoint. To generate soft hate, we integrate the \emph{Argumentum Model of Topics} (AMT) and \emph{Relevance Theory} (RT) in a unified framework: AMT provides the backbone argument structure for rewriting an explicit hateful standpoint into a seemingly neutral discussion while preserving the stance, and RT guides generation to keep the AMT chain logically coherent. The benchmark spans \textbf{7} sociocultural domains and \textbf{28} target groups, comprising \textbf{4,745} soft-hate instances. Evaluations across encoder-based detectors, general-purpose LLMs, and safety models show a consistent drop from hard to soft tiers: systems that detect explicit hostility often fail when the same stance is conveyed through subtle, reasoning-based language. \textcolor{red}{\textbf{Disclaimer.} Contains offensive examples used solely for research.} |
| 2026-01-28 | [How AI Impacts Skill Formation](http://arxiv.org/abs/2601.20245v1) | Judy Hanwen Shen, Alex Tamkin | AI assistance produces significant productivity gains across professional domains, particularly for novice workers. Yet how this assistance affects the development of skills required to effectively supervise AI remains unclear. Novice workers who rely heavily on AI to complete unfamiliar tasks may compromise their own skill acquisition in the process. We conduct randomized experiments to study how developers gained mastery of a new asynchronous programming library with and without the assistance of AI. We find that AI use impairs conceptual understanding, code reading, and debugging abilities, without delivering significant efficiency gains on average. Participants who fully delegated coding tasks showed some productivity improvements, but at the cost of learning the library. We identify six distinct AI interaction patterns, three of which involve cognitive engagement and preserve learning outcomes even when participants receive AI assistance. Our findings suggest that AI-enhanced productivity is not a shortcut to competence and AI assistance should be carefully adopted into workflows to preserve skill formation -- particularly in safety-critical domains. |
| 2026-01-28 | [High-Resolution Mapping of Port Dynamics from Open-Access AIS Data in Tokyo Bay](http://arxiv.org/abs/2601.20211v1) | Moritz Hütten | Knowledge about vessel activity in port areas and around major industrial zones provides insights into economic trends, supports decision-making for shipping and port operators, and contributes to maritime safety. Vessel data from terrestrial receivers of the Automatic Identification System (AIS) have become increasingly openly available, and we demonstrate that such data can be used to infer port activities at high resolution and with precision comparable to official statistics. We analyze open-access AIS data from a three-month period in 2024 for Tokyo Bay, located in Japan's most densely populated urban region. Accounting for uneven data coverage, we reconstruct vessel activity in Tokyo Bay at $\sim\,$30~m resolution and identify 161 active berths across seven major port areas in the bay. During the analysis period, we find an average of $35\pm17_{\text{stat}}$ vessels moving within the bay at any given time, and $293\pm22_{\text{stat}}+65_{\text{syst}}-10_{\text{syst}}$ vessels entering or leaving the bay daily, with an average gross tonnage of $11{,}860^{+280}_{-\;\,50}$. These figures indicate an accelerating long-term trend toward fewer but larger vessels in Tokyo Bay's commercial traffic. Furthermore, we find that in dense urban environments, radio shadows in vessel AIS data can reveal the precise locations of inherently passive receiver stations. |
| 2026-01-28 | [Securing AI Agents in Cyber-Physical Systems: A Survey of Environmental Interactions, Deepfake Threats, and Defenses](http://arxiv.org/abs/2601.20184v1) | Mohsen Hatami, Van Tuan Pham et al. | The increasing integration of AI agents into cyber-physical systems (CPS) introduces new security risks that extend beyond traditional cyber or physical threat models. Recent advances in generative AI enable deepfake and semantic manipulation attacks that can compromise agent perception, reasoning, and interaction with the physical environment, while emerging protocols such as the Model Context Protocol (MCP) further expand the attack surface through dynamic tool use and cross-domain context sharing. This survey provides a comprehensive review of security threats targeting AI agents in CPS, with a particular focus on environmental interactions, deepfake-driven attacks, and MCP-mediated vulnerabilities. We organize the literature using the SENTINEL framework, a lifecycle-aware methodology that integrates threat characterization, feasibility analysis under CPS constraints, defense selection, and continuous validation. Through an end-to-end case study grounded in a real-world smart grid deployment, we quantitatively illustrate how timing, noise, and false-positive costs constrain deployable defenses, and why detection mechanisms alone are insufficient as decision authorities in safety-critical CPS. The survey highlights the role of provenance- and physics-grounded trust mechanisms and defense-in-depth architectures, and outlines open challenges toward trustworthy AI-enabled CPS. |
| 2026-01-28 | [What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering](http://arxiv.org/abs/2601.20164v1) | Jim Maar, Denis Paperno et al. | Prior work suggests that language models, while trained on next token prediction, show implicit planning behavior: they may select the next token in preparation to a predicted future token, such as a likely rhyming word, as supported by a prior qualitative study of Claude 3.5 Haiku using a cross-layer transcoder. We propose much simpler techniques for assessing implicit planning in language models. With case studies on rhyme poetry generation and question answering, we demonstrate that our methodology easily scales to many models. Across models, we find that the generated rhyme (e.g. "-ight") or answer to a question ("whale") can be manipulated by steering at the end of the preceding line with a vector, affecting the generation of intermediate tokens leading up to the rhyme or answer word. We show that implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters. Our methodology offers a widely applicable direct way to study implicit planning abilities of LLMs. More broadly, understanding planning abilities of language models can inform decisions in AI safety and control. |
| 2026-01-28 | [Supporting Informed Self-Disclosure: Design Recommendations for Presenting AI-Estimates of Privacy Risks to Users](http://arxiv.org/abs/2601.20161v1) | Isadora Krsek, Meryl Ye et al. | People candidly discuss sensitive topics online under the perceived safety of anonymity; yet, for many, this perceived safety is tenuous, as miscalibrated risk perceptions can lead to over-disclosure. Recent advances in Natural Language Processing (NLP) afford an unprecedented opportunity to present users with quantified disclosure-based re-identification risk (i.e., "population risk estimates", PREs). How can PREs be presented to users in a way that promotes informed decision-making, mitigating risk without encouraging unnecessary self-censorship? Using design fictions and comic-boarding, we story-boarded five design concepts for presenting PREs to users and evaluated them through an online survey with N = 44 Reddit users. We found participants had detailed conceptions of how PREs may impact risk awareness and motivation, but envisioned needing additional context and support to effectively interpret and act on risks. We distill our findings into four key design recommendations for how best to present users with quantified privacy risks to support informed disclosure decision-making. |
| 2026-01-27 | [Counterfactual Cultural Cues Reduce Medical QA Accuracy in LLMs: Identifier vs Context Effects](http://arxiv.org/abs/2601.20102v1) | Amirhossein Haji Mohammad Rezaei, Zahra Shakeri | Engineering sustainable and equitable healthcare requires medical language models that do not change clinically correct diagnoses when presented with non-decisive cultural information. We introduce a counterfactual benchmark that expands 150 MedQA test items into 1650 variants by inserting culture-related (i) identifier tokens, (ii) contextual cues, or (iii) their combination for three groups (Indigenous Canadian, Middle-Eastern Muslim, Southeast Asian), plus a length-matched neutral control, where a clinician verified that the gold answer remains invariant in all variants. We evaluate GPT-5.2, Llama-3.1-8B, DeepSeek-R1, and MedGemma (4B/27B) under option-only and brief-explanation prompting. Across models, cultural cues significantly affect accuracy (Cochran's Q, $p<10^-14$), with the largest degradation when identifier and context co-occur (up to 3-7 percentage points under option-only prompting), while neutral edits produce smaller, non-systematic changes. A human-validated rubric ($κ=0.76$) applied via an LLM-as-judge shows that more than half of culturally grounded explanations end in an incorrect answer, linking culture-referential reasoning to diagnostic failure. We release prompts and augmentations to support evaluation and mitigation of culturally induced diagnostic errors. |
| 2026-01-27 | [Techno-economic optimization of a heat-pipe microreactor, part II: multi-objective optimization analysis](http://arxiv.org/abs/2601.20079v1) | Paul Seurin, Dean Price | Heat-pipe microreactors (HPMRs) are compact and transportable nuclear power systems exhibiting inherent safety, well-suited for deployment in remote regions where access is limited and reliance on costly fossil fuels is prevalent. In prior work, we developed a design optimization framework that incorporates techno-economic considerations through surrogate modeling and reinforcement learning (RL)-based optimization, focusing solely on minimizing the levelized cost of electricity (LCOE) by using a bottom-up cost estimation approach. In this study, we extend that framework to a multi-objective optimization that uses the Pareto Envelope Augmented with Reinforcement Learning (PEARL) algorithm. The objectives include minimizing both the rod-integrated peaking factor ($F_{Δh}$) and LCOE -- subject to safety and operational constraints. We evaluate three cost scenarios: (1) a high-cost axial and drum reflectors, (2) a low-cost axial reflector, and (3) low-cost axial and drum reflectors. Our findings indicate that reducing the solid moderator radius, pin pitch, and drum coating angle -- all while increasing the fuel height -- effectively lowers $F_{Δh}$. Across all three scenarios, four key strategies consistently emerged for optimizing LCOE: (1) minimizing the axial reflector contribution when costly, (2) reducing control drum reliance, (3) substituting expensive tri-structural isotropic (TRISO) fuel with axial reflector material priced at the level of graphite, and (4) maximizing fuel burnup. While PEARL demonstrates promise in navigating trade-offs across diverse design scenarios, discrepancies between surrogate model predictions and full-order simulations remain. Further improvements are anticipated through constraint relaxation and surrogate development, constituting an ongoing area of investigation. |

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



