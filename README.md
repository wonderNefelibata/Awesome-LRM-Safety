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
| 2026-05-22 | [A Novel Approach for the Counting of Wood Logs Using cGANs and Image Processing Techniques](http://arxiv.org/abs/2605.23775v1) | João VC Mazzochin, Giovani Bernardes Vitor et al. | This study tackles the challenge of precise wood log counting, where applications of the proposed methodology can span from automated approaches for materials management, surveillance, and safety science to wood traffic monitoring, wood volume estimation, and others. We introduce an approach leveraging Conditional Generative Adversarial Networks (cGANs) for eucalyptus log segmentation in images, incorporating specialized image processing techniques to handle noise and intersections, coupled with the Connected Components Algorithm for efficient counting. To support this research, we created and made publicly available a comprehensive database of 466 images containing approximately 13,048 eucalyptus logs, which served for both training and validation purposes. Our method demonstrated robust performance, achieving an average Accuracy_pixel of 96.4% and Accuracy_logs of 92.3%, with additional measures such as F1 scores ranging from 0.879 to 0.933 and IoU values between 0.784 and 0.875, further validating its effectiveness. The implementation proves to be efficient with an average processing time of 0.713s per image on an NVIDIA T4 GPU, making it suitable for realtime applications. The practical implications of this method are significant for operational forestry, enabling more accurate inventory management, reducing human errors in manual counting, and optimizing resource allocation. Furthermore, the segmentation capabilities of the model provide a foundation for advanced applications such as eucalyptus stack volume estimation, contributing to a more comprehensive and refined analysis of forestry operations. The methodology's success in handling complex scenarios, including intersecting logs and varying environmental conditions, positions it as a valuable tool for practical applications across related industrial sectors. |
| 2026-05-22 | [NLG Evaluation: Past, Present, Future](http://arxiv.org/abs/2605.23715v1) | Ehud Reiter | Natural Language Generation (NLG) evaluation has changed dramatically since 1990, and will continue to evolve in the future. In 1990, when NLG had close ties to linguistics, there was very little formal experimental evaluation in the modern sense. In 2026, when NLG is closely linked to machine learning, experimental evaluation is expected and indeed fundamental to research. Many evaluation techniques were developed over this period, including most recently LLM-as-Judge. I expect NLG evaluation will continue to evolve in the future. In particular, impact, qualitative, and safety evaluation will become more important as large numbers of people routinely use NLG technology. |
| 2026-05-22 | [Graph-based Complexity Forecasts in UK En Route Airspace Using Relevant Aircraft Interactions](http://arxiv.org/abs/2605.23696v1) | Edward Henderson, George De Ath et al. | Effectively managing Air Traffic Control Officer (ATCO) workload is crucial in maintaining operational safety. Group supervisors use tools that estimate upcoming traffic load to aid decision-making. However, industry-standard models can fail to capture the nuances of upcoming air traffic complexity. This study presents a probabilistic approach to forecast the complexity of an airspace sector using the number of relevant aircraft pairs, i.e., those that require monitoring or deconfliction by a controller, as a proxy measure for ATCO workload. We adapted an existing filter algorithm to make it suitable for use in London Middle Sector (LMS), a complex airspace sector with multiple flows of traffic above some of the busiest airports in Europe. Through iterative feedback with ATCOs, the algorithm was refined and extended to handle specific geometric and operational considerations. The updated algorithm outperformed the original, with an F1-score of 0.84 compared to 0.69 on a labelled set of 50 traffic scenarios. To produce forecasts of future numbers of relevant aircraft pairs in the sector, a graph representation of the LMS route network was constructed, standardising the spatial fidelity of route legs. The forecasting method accounts for uncertainty in aircraft arrival times by modelling the probability of each aircraft occupying route segments at future query times. When combined with historic distributions of relevant interactions and a live operational data stream, predictions of upcoming ATCO workload could be made up to 45 minutes in advance. The proposed method to forecast upcoming workload showed a significantly stronger correlation with actual relevant interactions (Spearman's $ρ= 0.68$) than a standard traffic volume prediction ($ρ= 0.55$). The resulting data-driven tool shows promise for use by group supervisors to inform sector configuration and ATCO rostering decisions. |
| 2026-05-22 | [Relevant Walk Search for Explaining Graph Neural Networks](http://arxiv.org/abs/2605.23673v1) | Ping Xiong, Thomas Schnake et al. | Graph Neural Networks (GNNs) have become important machine learning tools for graph analysis, and its explainability is crucial for safety, fairness, and robustness. Layer-wise relevance propagation for GNNs (GNN-LRP) evaluates the relevance of \emph{walks} to reveal important information flows in the network, and provides higher-order explanations, which have been shown to be superior to the lower-order, i.e., node-/edge-level, explanations. However, identifying relevant walks by GNN-LRP requires {\em exponential} computational complexity with respect to the network depth, which we will remedy in this paper. Specifically, we propose {\em polynomial-time} algorithms for finding top-$K$ relevant walks, which drastically reduces the computation and thus increases the applicability of GNN-LRP to large-scale problems. Our proposed algorithms are based on the \emph{max-product} algorithm -- a common tool for finding the maximum likelihood configurations in probabilistic graphical models -- and can find the most relevant walks exactly at the neuron level and approximately at the node level. Our experiments demonstrate the performance of our algorithms at scale and their utility across application domains, i.e., on epidemiology, molecular, and natural language benchmarks. We provide our codes under \href{https://github.com/xiong-ping/rel_walk_gnnlrp}{github.com/xiong-ping/rel\_walk\_gnnlrp}. |
| 2026-05-22 | [RF Instrument Agent (RFIA): Empowering RF Instruments with Natural Language Understanding, Scheduling and Execution of Complex Tasks](http://arxiv.org/abs/2605.23636v1) | Chunhui Li, Wei Fan | Modern radio-frequency (RF) instruments, such as vector network analyzers (VNAs), already provide mature remote-control interfaces. However, practical RF measurement workflows still rely on manual operation or custom scripting, which is time-consuming and expertise-intensive. This paper presents RF Instrument Agent (RFIA), a natural-language agent framework for reliable task-driven RF instrument control. RFIA adopts a decoupled intent--planning--execution architecture, where the LLM is used only for task understanding and high-level planning, while instrument-facing operations are handled by a deterministic runtime. Verified skills, workflow templates, RF analysis tools, instrument-specific rules, and retrieval-assisted SCPI knowledge are organized in a structured knowledge base, and hybrid execution graphs are used for closed-loop measurement tasks. A hardware-in-the-loop prototype is implemented on a commercial VNA and evaluated using a 16-task benchmark covering configuration, query, acquisition, rule-aware operation, RF-data analysis, and closed-loop measurement. RFIA handles all benchmark tasks under predefined execution and safety policies, including one expected safety rejection. Hardware-in-the-loop results with both a 230B-scale MiniMax-M2.7 model and a smaller 27B-scale Qwen3.6-27B model confirm that the decoupled architecture supports reliable natural-language RF measurement automation across different LLM backends. |
| 2026-05-22 | [Formally Verified Liveness with Multiparty Session Types in Rocq](http://arxiv.org/abs/2605.23633v1) | Omer Keskin, Nobuko Yoshida et al. | Multiparty session types (MPST) offer a framework for the description of communication-based protocols involving multiple participants. In the top-down approach to MPST, the communication pattern of the session is described using a global type. Then the global type is projected on to a local type for each participant, and the individual processes making up the session are type-checked against these projections. Typed sessions possess certain desirable properties such as safety, deadlock-freedom and liveness.   In this work, we present the first mechanised proof of liveness for synchronous multiparty session types in the Rocq Proof Assistant. Building on recent work, we represent global and local types as coinductive trees using the paco library. We use a coinductively defined subtyping relation on local types together with another coinductively defined plain-merge projection relation relating local and global types. We then associate collections of local types, or local type contexts, with global types using this projection and subtyping relations, and prove an operational correspondence between a local type context and its associated global type. We utilise this association relation to prove the safety and liveness of associated local type contexts and, consequently, the multiparty sessions typed by these contexts.   Besides clarifying the often informal proofs found in the MPST literature, our Rocq mechanisation also enables the certification of liveness properties of communication protocols. Our contribution amounts to around 14K lines of Rocq code, available at https://github.com/omerskeskin/mpstlive . |
| 2026-05-22 | [Safety, Liveness, and Fairness in Quantitative Argumentation Dialogues](http://arxiv.org/abs/2605.23578v1) | Arunavo Ganguly, Julian Alfredo Mendez et al. | We introduce notions of safety, liveness, and fairness, as commonly used in temporal reasoning, to quantitative (bipolar) argumentation dialogues where repeated inferences are drawn from argumentation graphs with weighted nodes. Between inferences, these graphs undergo updates. Strong and weak safety capture that arguments' (final) strengths remain above a specific threshold of justification and always reach the threshold eventually, respectively. Liveness requires that arguments' strengths fluctuate across the threshold of justification. Fairness notions assess how safe arguments are spread within a sequence of argumentation graphs. We formally show how these notions are related, and discuss some analytical challenges with respect to providing general guarantees for our properties. |
| 2026-05-22 | [TactileReflex: Noise-Statistics-Driven Vision-Tactile Reflex Control for Force-Sensitive Manipulation](http://arxiv.org/abs/2605.23568v1) | Ziyan Feng, Yulong Fu et al. | Manipulating fragile deformable containers, such as disposable plastic cups filled with liquid, demands real-time grip-force adaptation within an extremely narrow force margin: insufficient force causes slip, while excessive force irreversibly deforms the thin wall. Existing approaches struggle to achieve such force-sensitive manipulation tasks. We propose a noise-statistics-based calibration-driven reflex control paradigm with vision-based tactile sensing: by analyzing the sensor's intrinsic noise characteristics (via a brief static-hold-and-unload protocol), we directly derive all controller thresholds, eliminating external force calibration, trial-and-error manual tuning, or material-specific physical models. Instantiating this paradigm, we present TactileReflex, a three-channel closed-loop controller that extracts three image-level proxies, shear intensity ($S_y$), contact intensity ($F_n$), and center of pressure ($C$), from dual visuo-tactile sensors and drives prioritized reflex channels at ~12 Hz for slip suppression, weight-adaptive release, and force protection. Each channel closes the loop directly on its proxy via noise-derived thresholds. Ablation demonstrates that only the full three-channel system is able to prevent irreversible container deformation (5/5 success vs. at most 1/5 for partial configurations). In a dynamic pouring task, fixed-effort baselines fail in all 10 attempts due to pose drift, while TactileReflex achieves 9/10 success across two water volumes. As a self-contained and interpretable controller, TactileReflex can serve as a plug-and-play safety layer beneath high-level manipulation pipelines, including haptic-free VR teleoperation and vision-language-action (VLA) policies. |
| 2026-05-22 | [SafeSABR: Risk-Calibrated Adaptive Bitrate Streaming over Starlink Networks](http://arxiv.org/abs/2605.23560v1) | Hongjun Xie, Jiahang Zhu et al. | Starlink, as a representative low Earth orbit (LEO) satellite broadband system, makes high-bitrate video streaming possible in regions where terrestrial broadband is unavailable. However, its access links exhibit rapid throughput fluctuations caused by satellite mobility and handovers. Existing learned adaptive bitrate (ABR) algorithms can achieve high average quality of experience (QoE), yet high-bitrate Starlink streaming exposes severe session-level rebuffering that is not captured by average QoE alone. To address it, this paper proposes SafeSABR, a risk-calibrated learned ABR framework for Starlink networks. SafeSABR formulates Starlink ABR as a QoE--severe-risk tradeoff and follows a three-stage design: behavior-cloning pretraining learns a high-QoE ABR prior, risk-calibrated reinforcement learning (RL) fine-tuning reduces severe-tail action tendencies, and a runtime safety auditor uses safe-capacity lower bounds to check policy-requested bitrates before execution. Experiments on real Starlink traces compare SafeSABR with online, prediction-assisted, and learned ABR baselines. Compared with advanced methods, SafeSABR reduces severe-stall sessions from 22.8% to 7.2% and worst-5% session rebuffering from 54.30 s to 22.68 s, with a 1.8% QoE cost. Component analyses further show that risk-calibrated fine-tuning and safe-capacity auditing reduce unsafe bitrate decisions and downstream severe-session rebuffering. These results show that combining risk-calibrated policy learning with decision-aware safe throughput forecasting can move learned ABR toward a safer QoE--severe-risk operating point under volatile Starlink networks. |
| 2026-05-22 | [MISRust: Mapping MISRA-C++ Coding Guidelines to the Rust Programming Language](http://arxiv.org/abs/2605.23490v1) | Marius Molz, Niels Schneider et al. | The Rust programming language is increasingly being considered for safety-critical system development. However, established safety standards such as ISO 26262 require the use of coding guidelines that do not yet exist for Rust. This paper systematically examines each of the 179 MISRA C++ 2023 coding guidelines and classifies them into 6 categories based on their applicability to Rust. Our approach analyzes the rationale behind each MISRA rule to determine whether it remains valid in the Rust programming context. We find that 47.75% of the 111 as-is applicable MISRA rules are automatically enforced by Rust's language design, eliminating the need for explicit guideline enforcement. Furthermore, our analysis explicitly distinguishes between safe and unsafe Rust. We find that 69 guidelines are still relevant and still require either direct application or adaptation for Rust. Importantly, 36 of these rules are automatically satisfied when only using the safe subset of the Rust language. However, they are required again if unsafe Rust features are introduced. We also identify specific areas where new Rust-specific guidelines are needed. Where a guideline does not directly translate, we propose Rust-specific adaptations that preserve its intent. All mapping results and supporting artifacts are publicly available as open-source materials at https://github.com/embedded-software-laboratory/MISRust. |
| 2026-05-22 | [CBANet: A Compact Attention-Based CNN-BiLSTM Network for Aggressive Driving Event Detection](http://arxiv.org/abs/2605.23471v1) | Hanadi Alhamdan, Ghadah Alosaimi et al. | Aggressive driving is a major cause of traffic accidents and poses a serious threat to road safety. Although deep learning methods have shown promising results in detecting risky driving behaviours from vehicle sensor data, their performance in real-world conditions is often limited by severe data imbalance, large variability between drivers, and the lack of physically interpretable vehicle dynamics representations. In this paper, we propose an enhanced deep learning framework for aggressive driving detection using multivariate vehicle dynamics signals. Instead of relying solely on raw measurements, the proposed approach constructs engineered dynamic features that capture steering, acceleration, and braking behaviour. To address the extreme rarity of aggressive events in naturalistic driving data, we introduce a stable training strategy that combines controlled SMOTE-based oversampling with a class-weighted loss formulation, and evaluates focal loss variants for imbalance handling. Furthermore, a safety-oriented decision strategy based on class-specific threshold calibration is adopted to better reflect the asymmetric risks of missed detections and false alarms in real-world applications. The proposed framework is evaluated on a newly collected naturalistic driving dataset. Extensive experiments show that the proposed method consistently outperforms standard deep learning baselines with significant improvements in minority-class recall and safety-critical F-score metrics while maintaining practical computational efficiency. Code: \url {https://github.com/halhamdan/CBANet} |
| 2026-05-22 | [IyàwóBench: A Benchmark for Evaluating Large Language Model Clinical Triage Accuracy on Undifferentiated Febrile Illness in Nigerian Primary Health Settings](http://arxiv.org/abs/2605.23465v1) | Anthonio Oladimeji Gabriel, Dimeji Abdulsobur Olawuyi et al. | Background. Undifferentiated febrile illness is the leading cause of primary care outpatient visits in Nigeria, yet no validated benchmark exists for evaluating large language model (LLM) clinical triage reasoning in West African primary health settings. Methods. We introduce IyàwóBench v1.0, a dataset of 200 synthetic clinical vignettes across eight febrile illness categories derived from statistical distributions of 1,200 real patient encounters at 19 primary health centres (PHCs) in Oyo State, Nigeria. Six LLMs were evaluated on structured triage classification across two metrics: triage accuracy and safety score. Results. All six models achieved 100% safety scores (95% CI: 96.4-100.0%), never downgrading a critical REFER NOW case to TREAT HERE. Triage accuracy varied substantially: Claude Sonnet (claude-sonnet-4-5) 67.5% (95% CI: 60.8-73.7%), Llama 4 Scout 59.5% (52.5-66.2%), Llama 3.3 70B 43.0% (36.2-50.0%), and Llama 3.1 8B 39.0% (32.4-45.9%). Two models demonstrated near-zero accuracy attributable to structured output non-compliance. Conclusions. Modern LLMs exhibit safe triage behaviour but vary substantially in structured clinical accuracy. Clinically engineered systems with embedded WHO guidelines outperform general-purpose models by up to 28.5 percentage points. IyàwóBench provides the first reproducible evaluation framework for LLM clinical decision support in West African primary care. |
| 2026-05-22 | [A Distributed Framework for Data-Driven Safe Coordination in Leader-Follower Networks](http://arxiv.org/abs/2605.23356v1) | Mirhan Urkmez, Maryam Sharifi et al. | This paper addresses connectivity preservation in leader-follower multi-agent systems with unknown control-affine dynamics and local state information. We introduce the distributed data-driven zeroing control barrier function (3D-ZCBF) framework, which ensures the controlled invariance of safety sets by identifying derivative bounds from input-state data without requiring explicit models of high-dimensional agent dynamics. In this work, we derive the explicit, decoupled safety conditions necessary to maintain connectivity for leader-leader, and follower-follower pairings. These individual constraints, along with the leader-follower conditions, are aggregated into explicit system-wide conditions that formally guarantee the preservation of the entire communication network. Furthermore, we provide a quantitative analysis demonstrating how the size of the collected data set and the accuracy of the learned Jacobian bounds impact the feasibility of the safety certificates. The proposed conditions are implemented via a projection-based controller, and simulations confirm that these explicit 3D-ZCBF requirements effectively maintain system-level connectivity using only local, two-hop information. |
| 2026-05-22 | [Prudent-Banker: No Extra Fees for Baseline Safety in Adversarial Bandits With and Without Delays](http://arxiv.org/abs/2605.23351v1) | Ting Hu, Luanda Cai et al. | We study adversarial multi-armed bandits with and without delayed feedback under a safety-aware goal: achieving minimax-optimal worst-case regret while keeping nearly constant regret relative to a designated "safe" baseline policy. Existing approaches can balance this trade-off with immediate feedback for smooth comparators, but arbitrary delays can mistime transitions between conservatism and exploration, endangering the safety guarantee. To bridge this gap, we propose Prudent-Banker, a novel algorithm that combines a delay-adapted variant of Online Mirror Descent with a modified phased-aggression mechanism. Its key technical contribution is a delay-calibrated restart threshold that rigorously accounts for the worst-case distortion induced by unobserved feedback and reliably detects comparator suboptimality. We also establish new lower bounds for safety-constrained adversarial delayed bandits, showing that the regret guarantees of Prudent-Banker are unimprovable, up to logarithmic factors, under the baseline-safety requirement. To the best of our knowledge, Prudent-Banker is the first algorithm to achieve the optimal safety--robustness trade-off: pseudo-regret $\widetilde{O}(\sqrt{T}+\sqrt{D})$ together with $\widetilde{O}(1)$ regret against the safe comparator, both with and without delays. Experiments across diverse delay distributions show that, unlike standard delay-robust baselines, Prudent-Banker effectively balances safety and learning. |
| 2026-05-22 | [From Visual to Digital: Coordination Scheduling and Its Effect on Safety and Efficiency in UAM Corridors](http://arxiv.org/abs/2605.23343v1) | Akihiro Fujita, Sasinee Pruekprasert et al. | This paper explores scalable coordination strategies for urban air mobility (UAM) corridors by comparing two representative approaches. The first, inspired by visual flight rules (VFR), is a local coordination strategy relying on spatial information available to each vehicle. The second, conceptually aligned with digital flight rules (DFR), is a global coordination strategy based on shared estimated times of arrival (ETAs) at constrained waypoints (CWPs). To support this comparison, we introduce a lightweight disturbance-avoidance mechanism that enables vehicles to adjust their ETAs in response to forecasted disruptions using shared information. We evaluate these approaches through numerical simulations under varying disturbance levels, comparing the locally reactive VFR-style scheme with the globally coordinated DFR-style scheme. Results show that VFR achieves high throughput in low-traffic scenarios but becomes increasingly prone to collisions at higher traffic densities unless conservative separation is enforced, which reduces traffic efficiency. In contrast, DFR maintains more consistent safety performance and traffic efficiency, even under moderate ETA update propagation delays. These findings highlight the advantages of DFR-style global coordination in managing high-density air traffic control (ATC) operations within UAM corridors. |
| 2026-05-22 | [Safety-Assured Arrival Scheduling in Sequential UAM Corridor Sections under Speed and Separation Constraints](http://arxiv.org/abs/2605.23333v1) | Sasinee Pruekprasert, Shinji Nakadai et al. | This paper presents a safety-assured arrival-scheduling framework for Urban Air Mobility (UAM) corridor operations. We propose an analytical method to compute a sufficient ETA gap at Constrained Waypoints (CWPs) that guarantees longitudinal separation along sequential corridor sections with heterogeneous speed limits. The resulting ETA-gap condition depends on section-specific speed bounds and the required separation distance, providing an efficiently computable rule suitable for integration into future digital ETA-scheduling and air traffic management systems. We show that the computed ETA gap ensures safe separation across all corridor sections under prescribed section travel times and speed limits. Numerical simulations for a decreasing-speed corridor confirm that vehicles coordinated with the proposed mechanism adjust their speeds to maintain the required spacing, avoid potential collisions, and support improved traffic flow compared with unscheduled operations. |
| 2026-05-22 | [Cultural Adaptation in Large Language Models for Political Discourse](http://arxiv.org/abs/2605.23332v1) | Wajdi Zaghouani | The integration of large language models into political discourse analysis creates new opportunities for comparative research, policy analysis, and civic technology, while introducing material risks for democratic accountability. This paper argues that cultural adaptation is a prerequisite for trustworthy deployment of large language models in political communication across diverse linguistic and institutional contexts. Current systems remain shaped by English dominant data, uneven multilingual coverage, and assumptions grounded in a narrow range of political institutions and discourse conventions, producing systematic errors when applied across cultures. We formalize cultural adaptation across translation, discourse, and ontology levels, identify recurring cultural failure modes in political NLP, and propose an operational evaluation matrix grounded in cultural fidelity, calibration, and democratic safety. Building on political text analysis, sociotechnical auditing, and cross cultural pragmatics, we outline methodological pathways including participatory dataset development, culturally aware transfer learning, and benchmark design that makes cultural adaptation empirically measurable. We conclude by clarifying governance constraints and scope conditions under which culturally adaptive political NLP can support democratic legitimacy. |
| 2026-05-22 | [Human-in-the-Loop Multi-Agent Ventilator Decision Support with Contextual Bandit Preference Learning](http://arxiv.org/abs/2605.23320v1) | Sijia Li, Xiaoyu Tan et al. | Ventilator decision support requires sequential decisions that track evolving physiology and disease trajectories while respecting safety boundaries and clinician specific tuning styles. Rule based approaches rarely generalize personalization, and end to end reinforcement learning or single large language model systems remain difficult to control and audit. We propose the Ventilator Decision Support System (VDSS), a human in the loop multi agent framework that coordinates modular decision components through contract driven structured interfaces and produces traceable evidence for review. VDSS performs online preference adaptation with a contextual bandit, updating clinician specific preferences from the final accepted decision at each adjustment cycle and using them to guide subsequent recommendations. Structured rejection feedback triggers targeted replanning to reduce unproductive iterations and improve interaction stability. Retrospective ICU trajectory replay with expert review indicates higher recommendation acceptability and fewer interaction rounds to reach an acceptable plan, supporting clinically deployable human AI collaboration. |
| 2026-05-22 | [DART: Semantic Recoverability for Structured Tool Agents](http://arxiv.org/abs/2605.23311v1) | Ke Yang, Panpan Li et al. | When a structured tool agent fails mid-execution, the runtime faces a dilemma: replaying the entire task is safe but wasteful, while restoring from a local checkpoint is efficient but can leave committed downstream work tied to an upstream history that no longer exists. This tension is acute in commitment-sensitive settings, where rollback targets a single failed instance yet downstream consumers have already acted on its output. Existing recovery approaches provide mechanical rollback but no criterion for whether a local restore remains semantically valid after downstream commitment. We formalize this gap as semantic recoverability and address it in DART, a modular runtime that localizes the failed instance, certifies semantically recoverable boundaries of that instance, aligns checkpoints to those boundaries, and selects an admissible restore point that preserves committed downstream work under dependency and effect constraints-or blocks otherwise. Across three LLM-driven domains and external validation on a LangGraph-based substrate, DART correctly recovers all evaluated commitment-sensitive cases where baseline local recovery fails, and a five-domain safety audit finds no unsafe admitted rollbacks. These results show that controller legality does not imply semantic validity, and that sound local recovery requires an explicit admissibility check. |
| 2026-05-22 | [General Hazard Detection](http://arxiv.org/abs/2605.23304v1) | Stephanie Ng, CP Lim et al. | Hazard, as an abstract concept, is typically defined through cognitive-level logical reasoning rather than concrete examples. In contrast, existing hazard detection systems rely on predefined hazard categories and require intensive collection of labelled examples within detection or classification architectures. This approach faces three fundamental challenges when addressing abstract safety concepts: (1) noisy and sparse training data, (2) dynamically evolving definitions that change across contexts and time, and (3) limited generalisation to unseen or novel scenarios. To address these limitations, we present the CompliVision dataset, the first general-purpose hazard dataset designed for rule-based compliance assessment, along with a baseline framework for hazard evaluation. Our key innovation is decoupling the hazard concept from image-based examples by expressing safety requirements through language-based rules. We ground our approach in authoritative domain regulations and ISO standards to define diverse hazard concepts across multiple domains. The CompliVision dataset comprises 3,006 images spanning traffic, construction, and warehouse environments, with each image annotated for compliance against specific safety rules, accompanied by natural language explanations highlighting the supporting visual evidence. To achieve robust generalisation, we develop an active learning framework to more effectively guide and refine vision-language models in assessing hazard compliance. While state-of-the-art VLMs demonstrate strong capabilities, they struggle with the fine-grained, context-dependent interpretation required for accurate safety assessment. We proposed a general hazard detection framework to address this limitation which combines LLaVA-based visual reasoning with with human-in-the-loop feedback. |

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



