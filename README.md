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
| 2025-11-25 | [Fighting AI with AI: Leveraging Foundation Models for Assuring AI-Enabled Safety-Critical Systems](http://arxiv.org/abs/2511.20627v1) | Anastasia Mavridou, Divya Gopinath et al. | The integration of AI components, particularly Deep Neural Networks (DNNs), into safety-critical systems such as aerospace and autonomous vehicles presents fundamental challenges for assurance. The opacity of AI systems, combined with the semantic gap between high-level requirements and low-level network representations, creates barriers to traditional verification approaches. These AI-specific challenges are amplified by longstanding issues in Requirements Engineering, including ambiguity in natural language specifications and scalability bottlenecks in formalization. We propose an approach that leverages AI itself to address these challenges through two complementary components. REACT (Requirements Engineering with AI for Consistency and Testing) employs Large Language Models (LLMs) to bridge the gap between informal natural language requirements and formal specifications, enabling early verification and validation. SemaLens (Semantic Analysis of Visual Perception using large Multi-modal models) utilizes Vision Language Models (VLMs) to reason about, test, and monitor DNN-based perception systems using human-understandable concepts. Together, these components provide a comprehensive pipeline from informal requirements to validated implementations. |
| 2025-11-25 | [Translating Large-Scale C Repositories to Idiomatic Rust](http://arxiv.org/abs/2511.20617v1) | Saman Dehghan, Tianran Sun et al. | Existing C to Rust translation techniques fail to balance quality and scalability: transpilation-based approaches scale to large projects but produce code with poor safety, idiomaticity, and readability. In contrast, LLM-based techniques are prohibitively expensive due to their reliance on frontier models (without which they cannot reliably generate compilable translations), thus limiting scalability. This paper proposes Rustine, a fully automated pipeline for effective and efficient repository-level C to idiomatic safe Rust translation. Evaluating on a diverse set of 23 C programs, ranging from 27 to 13,200 lines of code, Rustine can generate fully compilable Rust code for all and achieve 87% functional equivalence (passing 1,063,099 assertions out of 1,221,192 in test suites with average function and line coverage of 74.7% and 72.2%). Compared to six prior repository-level C to Rust translation techniques, the translations by Rustine are overall safer (fewer raw pointers, pointer arithmetic, and unsafe constructs), more idiomatic (fewer Rust linter violations), and more readable. When the translations cannot pass all tests to fulfill functional equivalence, human developers were able to complete the task in 4.5 hours, on average, using Rustine as debugging support. |
| 2025-11-25 | [Safe and Stable Neural Network Dynamical Systems for Robot Motion Planning](http://arxiv.org/abs/2511.20593v1) | Allen Emmanuel Binny, Mahathi Anand et al. | Learning safe and stable robot motions from demonstrations remains a challenge, especially in complex, nonlinear tasks involving dynamic, obstacle-rich environments. In this paper, we propose Safe and Stable Neural Network Dynamical Systems S$^2$-NNDS, a learning-from-demonstration framework that simultaneously learns expressive neural dynamical systems alongside neural Lyapunov stability and barrier safety certificates. Unlike traditional approaches with restrictive polynomial parameterizations, S$^2$-NNDS leverages neural networks to capture complex robot motions providing probabilistic guarantees through split conformal prediction in learned certificates. Experimental results on various 2D and 3D datasets -- including LASA handwriting and demonstrations recorded kinesthetically from the Franka Emika Panda robot -- validate S$^2$-NNDS effectiveness in learning robust, safe, and stable motions from potentially unsafe demonstrations. |
| 2025-11-25 | [PaTAS: A Parallel System for Trust Propagation in Neural Networks Using Subjective Logic](http://arxiv.org/abs/2511.20586v1) | Koffi Ismael Ouattara, Ioannis Krontiris et al. | Trustworthiness has become a key requirement for the deployment of artificial intelligence systems in safety-critical applications. Conventional evaluation metrics such as accuracy and precision fail to capture uncertainty or the reliability of model predictions, particularly under adversarial or degraded conditions. This paper introduces the \emph{Parallel Trust Assessment System (PaTAS)}, a framework for modeling and propagating trust in neural networks using Subjective Logic (SL). PaTAS operates in parallel with standard neural computation through \emph{Trust Nodes} and \emph{Trust Functions} that propagate input, parameter, and activation trust across the network. The framework defines a \emph{Parameter Trust Update} mechanism to refine parameter reliability during training and an \emph{Inference-Path Trust Assessment (IPTA)} method to compute instance-specific trust at inference. Experiments on real-world and adversarial datasets demonstrate that PaTAS produces interpretable, symmetric, and convergent trust estimates that complement accuracy and expose reliability gaps in poisoned, biased, or uncertain data scenarios. The results show that PaTAS effectively distinguishes between benign and adversarial inputs and identifies cases where model confidence diverges from actual reliability. By enabling transparent and quantifiable trust reasoning within neural architectures, PaTAS provides a principled foundation for evaluating model reliability across the AI lifecycle. |
| 2025-11-25 | [Gated Uncertainty-Aware Runtime Dual Invariants for Neural Signal-Controlled Robotics](http://arxiv.org/abs/2511.20570v1) | Tasha Kim, Oiwi Parker Jones | Safety-critical assistive systems that directly decode user intent from neural signals require rigorous guarantees of reliability and trust. We present GUARDIAN (Gated Uncertainty-Aware Runtime Dual Invariants), a framework for real-time neuro-symbolic verification for neural signal-controlled robotics. GUARDIAN enforces both logical safety and physiological trust by coupling confidence-calibrated brain signal decoding with symbolic goal grounding and dual-layer runtime monitoring. On the BNCI2014 motor imagery electroencephalogram (EEG) dataset with 9 subjects and 5,184 trials, the system performs at a high safety rate of 94-97% even with lightweight decoder architectures with low test accuracies (27-46%) and high ECE confidence miscalibration (0.22-0.41). We demonstrate 1.7x correct interventions in simulated noise testing versus at baseline. The monitor operates at 100Hz and sub-millisecond decision latency, making it practically viable for closed-loop neural signal-based systems. Across 21 ablation results, GUARDIAN exhibits a graduated response to signal degradation, and produces auditable traces from intent, plan to action, helping to link neural evidence to verifiable robot action. |
| 2025-11-25 | [Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward](http://arxiv.org/abs/2511.20561v1) | Yuwei Niu, Weiyang Jin et al. | Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at https://github.com/PKU-YuanGroup/UniSandBox |
| 2025-11-25 | [Assessing LLMs' Performance: Insights from the Chinese Pharmacist Exam](http://arxiv.org/abs/2511.20526v1) | Xinran Wang, Boran Zhu et al. | Background: As large language models (LLMs) become increasingly integrated into digital health education and assessment workflows, their capabilities in supporting high-stakes, domain-specific certification tasks remain underexplored.In China, the national pharmacist licensure exam serves as a standardized benchmark for evaluating pharmacists' clinical and theoretical competencies. Objective: This study aimed to compare the performance of two LLMs: ChatGPT-4o and DeepSeek-R1 on real questions from the Chinese Pharmacist Licensing Examination (2017-2021), and to discuss the implications of these performance differences for AI-enabled formative evaluation. Methods: A total of 2,306 multiple-choice (text-only) questions were compiled from official exams, training materials, and public databases. Questions containing tables or images were excluded. Each item was input in its original Chinese format, and model responses were evaluated for exact accuracy. Pearson's Chi-squared test was used to compare overall performance, and Fisher's exact test was applied to year-wise multiple-choice accuracy. Results: DeepSeek-R1 outperformed ChatGPT-4o with a significantly higher overall accuracy (90.0% vs. 76.1%, p < 0.001). Unit-level analyses revealed consistent advantages for DeepSeek-R1, particularly in foundational and clinical synthesis modules. While year-by-year multiple-choice performance also favored DeepSeek-R1, this performance gap did not reach statistical significance in any specific unit-year (all p > 0.05). Conclusion: DeepSeek-R1 demonstrated robust alignment with the structural and semantic demands of the pharmacist licensure exam. These findings suggest that domain-specific models warrant further investigation for this context, while also reinforcing the necessity of human oversight in legally and ethically sensitive contexts. |
| 2025-11-25 | [Adversarial Confusion Attack: Disrupting Multimodal Large Language Models](http://arxiv.org/abs/2511.20494v1) | Jakub Hoscilowicz, Artur Janicki | We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models. |
| 2025-11-25 | [Power-Efficient Autonomous Mobile Robots](http://arxiv.org/abs/2511.20467v1) | Liangkai Liu, Weisong Shi et al. | This paper presents pNav, a novel power-management system that significantly enhances the power/energy-efficiency of Autonomous Mobile Robots (AMRs) by jointly optimizing their physical/mechanical and cyber subsystems. By profiling AMRs' power consumption, we identify three challenges in achieving CPS (cyber-physical system) power-efficiency that involve both cyber (C) and physical (P) subsystems: (1) variabilities of system power consumption breakdown, (2) environment-aware navigation locality, and (3) coordination of C and P subsystems. pNav takes a multi-faceted approach to achieve power-efficiency of AMRs. First, it integrates millisecond-level power consumption prediction for both C and P subsystems. Second, it includes novel real-time modeling and monitoring of spatial and temporal navigation localities for AMRs. Third, it supports dynamic coordination of AMR software (navigation, detection) and hardware (motors, DVFS driver) configurations. pNav is prototyped using the Robot Operating System (ROS) Navigation Stack, 2D LiDAR, and camera. Our in-depth evaluation with a real robot and Gazebo environments demonstrates a >96% accuracy in predicting power consumption and a 38.1% reduction in power consumption without compromising navigation accuracy and safety. |
| 2025-11-25 | [Learning Control Barrier Functions with Deterministic Safety Guarantees](http://arxiv.org/abs/2511.20463v1) | Amy K. Strong, Ali Kashani et al. | Barrier functions (BFs) characterize safe sets of dynamical systems, where hard constraints are never violated as the system evolves over time. Computing a valid safe set and BF for a nonlinear (and potentially unmodeled), non-autonomous dynamical system is a difficult task. This work explores the design of BFs using data to obtain safe sets with deterministic assurances of control invariance. We leverage ReLU neural networks (NNs) to create continuous piecewise affine (CPA) BFs with deterministic safety guarantees for Lipschitz continuous, discrete-time dynamical system using sampled one-step trajectories. The CPA structure admits a novel classifier term to create a relaxed \ac{bf} condition and construction via a data driven constrained optimization. We use iterative convex overbounding (ICO) to solve this nonconvex optimization problem through a series of convex optimization steps. We then demonstrate our method's efficacy on two-dimensional autonomous and non-autonomous dynamical systems. |
| 2025-11-25 | [A Task-Oriented Evaluation Framework for Text Normalization in Modern NLP Pipelines](http://arxiv.org/abs/2511.20409v1) | Md Abdullah Al Kafi, Raka Moni et al. | Text normalization is an essential preprocessing step in many natural language processing (NLP) tasks, and stemming is one such normalization technique that reduces words to their base or root form. However, evaluating stemming methods is challenging because current evaluation approaches are limited and do not capture the potential harm caused by excessive stemming; therefore, it is essential to develop new approaches to evaluate stemming methods. To address this issue, this study propose a novel, task-oriented approach to evaluate stemming methods, which considers three aspects: (1) the utility of stemming using Stemming Effectiveness Score (SES), (2) the impact of stemming on downstream tasks using Model Performance Delta (MPD), and (3) the semantic similarity between stemmed and original words using Average Normalized Levenshtein Distance (ANLD), thus providing a comprehensive evaluation framework. We apply our evaluation framework to compare two stemmers for Bangla (BNLTK) and English (Snowball), and our results reveal a significant issue, prompting us to analyze their performance in detail. While the Bangla stemmer achieves the highest SES (1.67) due to effective word reduction (CR = 1.90), SES alone is insufficient because our proposed safety measure, ANLD, reveals that this high SES is due to harmful over-stemming (ANLD = 0.26), which correlates with the observed decrease in downstream performance.In contrast, the English stemmer achieves a moderate SES (1.31) with a safe meaning distance (ANLD = 0.14), allowing its word reduction to contribute positively to downstream performance; therefore, it is a more reliable stemmer. Our study provides a valuable tool for distinguishing between potential efficiency gains (high SES) and meaning preservation (low ANLD). |
| 2025-11-25 | [Optimization of the X-Arapuca Photon Collection Efficiency for the DUNE Horizontal Drift Far Detector](http://arxiv.org/abs/2511.20400v1) | E. Bertolini, C. Brizzolari et al. | The Deep Underground Neutrino Experiment (DUNE) Far Detector (FD) Photon Detection System (PDS) employs the X-Arapuca concept, a photon trapping system relying on reflective surfaces and dichroic filters. In this paper are reported measurements, performed at the University of Milano-Bicocca, aimed at increasing the FD Horizontal Drift (HD) PDS module efficiency. The baseline implementation of the X-Arapuca concept for the FD-HD PDS module is close to the DUNE requirements as demonstrated in the collaborations laboratory testing. However, an increased performance would provide a safety margin for a detector planned to be operated for 30 years, without possibility of performing maintenance. A higher detector performance would also benefit the DUNE low energy physics program. The already proven Milano-Bicocca setup has been utilized to test different PDS module configurations comparing them to the original baseline. Exploiting prior knowledge of the X-Arapuca components and Geant4 based optical simulations it has been possible to achieve up to an ~84% performance increase over the baseline design. In the following it is presented the testing procedure, the performed measurements and a brief discussion on the obtained results. |
| 2025-11-25 | [Identifying environmental factors associated with tetrodotoxin contamination in bivalve mollusks using eXplainable AI](http://arxiv.org/abs/2511.20395v1) | M. C. Schoppema, B. H. M. van der Velden et al. | Since 2012, tetrodotoxin (TTX) has been found in seafoods such as bivalve mollusks in temperate European waters. TTX contamination leads to food safety risks and economic losses, making early prediction of TTX contamination vital to the food industry and competent authorities. Recent studies have pointed to shallow habitats and water temperature as main drivers to TTX contamination in bivalve mollusks. However, the temporal relationships between abiotic factors, biotic factors, and TTX contamination remain unexplored.   We have developed an explainable, deep learning-based model to predict TTX contamination in the Dutch Zeeland estuary. Inputs for the model were meteorological and hydrological features; output was the presence or absence of TTX contamination.   Results showed that the time of sunrise, time of sunset, global radiation, water temperature, and chloride concentration contributed most to TTX contamination. Thus, the effective number of sun hours, represented by day length and global radiation, was an important driver for tetrodotoxin contamination in bivalve mollusks.   To conclude, our explainable deep learning model identified the aforementioned environmental factors (number of sun hours, global radiation, water temperature, and water chloride concentration) to be associated with tetrodotoxin contamination in bivalve mollusks; making our approach a valuable tool to mitigate marine toxin risks for food industry and competent authorities. |
| 2025-11-25 | [Accelerating Time-Optimal Trajectory Planning for Connected and Automated Vehicles with Graph Neural Networks](http://arxiv.org/abs/2511.20383v1) | Viet-Anh Le, Andreas A. Malikopoulos | In this paper, we present a learning-based framework that accelerates time- and energy-optimal trajectory planning for connected and automated vehicles (CAVs) using graph neural networks (GNNs). We formulate the multi-agent coordination problem encountered in traffic scenarios as a cooperative trajectory planning problem that minimizes travel time, subject to motion primitives derived from energy-optimal solutions. The effectiveness of this framework can be further improved through replanning at each time step, enabling the system to incorporate newly observed information. To achieve real-time execution of such a multi-agent replanning scheme, we employ a GNN architecture to learn the solutions of the time-optimal trajectory planning problem from offline-generated data. The trained model produces online predictions that serve as warm-start solutions for numerical optimization, thereby enabling rapid computation of minimal exit times and the associated feasible trajectories. This learning-augmented approach substantially reduces computation time while ensuring that all state, input, and safety constraints are satisfied. |
| 2025-11-25 | [Manganese-based macrocyclic chelates as novel MRI contrast agents: In vivo imaging in a porcine model](http://arxiv.org/abs/2511.20358v1) | Pål B. Marthinsen, Tuva R. Hope et al. | Objectives: Mn-based MRI contrast agents (MBCAs) have recently been proposed as alternatives to the currently used class of Gd-chelates. Unlike Gd, Mn is an endogenous paramagnetic metal with known biochemical pathways in the human body for excretion and metal regulation, which may alleviate the raised concerns about the safety of existing GBCAs. The aim of this study was to investigate the distribution, kinetics and image enhancement properties of a class of novel Mn-based macrocyclic chelates in a porcine model. Methods: Macrocyclic MBCAs, AH114608, GEH300017 and GEH200486, were tested and compared to gadoterate meglumine. Twelve female adult pigs were divided into four groups (n=3 for each CA). At 3 T MRI, T1 relaxometry analysis were measured longitudinally in multiple organs at five timepoints 30 minutes apart. CA kinetics was estimated from analysis of plasma CA concentrations by ICP-OES. Results: All four CAs exhibited T1-enhancing properties in the blood pool with GEH200486 having the largest increase in T1 relaxation rate (R1), GEH300017 and gadoterate meglumine having similar R1 increase and AH114608 a lower peak R1 change. A persistent increase in liver, kidney and myocardium R1 was observed with AH114608. To a lesser extent, a persistent increase of liver enhancement was also observed in T1-weighted images for GEH300017 and GEH200486 compared to gadoterate meglumine. All four CAs had similar bi-exponential plasma kinetics characterized by a rapid distribution phase and a slower elimination phase. Discussion: We have identified MBCA candidates with predominantly renal clearance and comparable efficacy in terms of vascular T1 relaxation, and comparable to the reference GBCA. The T1-enhancing properties of these novel Mn macrocyclic CAs can be used with routine clinical protocols and could be potentially utilised as an alternative to GBCAs for contrast-enhanced MRI procedures. |
| 2025-11-25 | [AD-R1: Closed-Loop Reinforcement Learning for End-to-End Autonomous Driving with Impartial World Models](http://arxiv.org/abs/2511.20325v1) | Tianyi Yan, Tao Tang et al. | End-to-end models for autonomous driving hold the promise of learning complex behaviors directly from sensor data, but face critical challenges in safety and handling long-tail events. Reinforcement Learning (RL) offers a promising path to overcome these limitations, yet its success in autonomous driving has been elusive. We identify a fundamental flaw hindering this progress: a deep seated optimistic bias in the world models used for RL. To address this, we introduce a framework for post-training policy refinement built around an Impartial World Model. Our primary contribution is to teach this model to be honest about danger. We achieve this with a novel data synthesis pipeline, Counterfactual Synthesis, which systematically generates a rich curriculum of plausible collisions and off-road events. This transforms the model from a passive scene completer into a veridical forecaster that remains faithful to the causal link between actions and outcomes. We then integrate this Impartial World Model into our closed-loop RL framework, where it serves as an internal critic. During refinement, the agent queries the critic to ``dream" of the outcomes for candidate actions. We demonstrate through extensive experiments, including on a new Risk Foreseeing Benchmark, that our model significantly outperforms baselines in predicting failures. Consequently, when used as a critic, it enables a substantial reduction in safety violations in challenging simulations, proving that teaching a model to dream of danger is a critical step towards building truly safe and intelligent autonomous agents. |
| 2025-11-25 | [Fast Matrix Multiplication via Ternary Meta Flip Graphs](http://arxiv.org/abs/2511.20317v1) | A. I. Perminov | Matrix multiplication optimization remains a fundamental challenge in computational mathematics. This work introduces a novel approach that discovers matrix multiplication schemes in the ternary field ($Z_T$), where coefficients are restricted to $\{-1, 0, 1\}$ to minimize naive additive complexity. The core of the method is a GPU-accelerated meta flip graph algorithm that maintains ternary safety through specialized arithmetic operations and sign symmetry breaking. Key results include new best ranks for the formats $4 \times 5 \times 12$, $5 \times 6 \times 10$, and $6 \times 7 \times 9$, the independent discovery of 32 schemes in $Z_T$ that match known optimal ranks (including 8 previously known only with rational coefficients), and 30 rank improvements in the binary field. The analysis of 164 known schemes shows that 92 can be implemented in $Z_T$, while 72 could not be found in the ternary field with current methods, defining the current boundaries of this approach. All software, results, and discovered schemes are provided as open-source. |
| 2025-11-25 | [Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs](http://arxiv.org/abs/2511.20158v1) | Ziqi Wang, Chang Che et al. | While continual visual instruction tuning (CVIT) has shown promise in adapting multimodal large language models (MLLMs), existing studies predominantly focus on models without safety alignment. This critical oversight ignores the fact that real-world MLLMs inherently require such mechanisms to mitigate potential risks. In this work, we shift our focus to CVIT for safety-aligned MLLMs and observe that during continual adaptation, the model not only suffers from task forgetting but also exhibits degradation in its safety. Achieving a harmonious balance between safety and task performance remains a crucial challenge. To address this, we propose Harmonious Parameter Adaptation (HPA), a post-training framework composed of focusing-based parameter partition, harmoniously balanced parameter selection, and orthogonal parameter adjustment. Specifically, HPA partitions parameters into two types based on their focus on safety or task performance, and selects the focused ones to preserve from a balanced perspective. In addition, HPA imposes orthogonality constraints on parameter updates to further alleviate catastrophic forgetting. Extensive experiments on the CVIT benchmark and safety evaluation datasets demonstrate that HPA better maintains high safety and mitigates forgetting than existing baselines. |
| 2025-11-25 | [The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights LLMs](http://arxiv.org/abs/2511.20104v1) | Craig Dickson | Prior work has shown that fine-tuning models on a narrow domain with misaligned data can lead to broad misalignment - a phenomenon termed "emergent misalignment" (Betley et al. 2025). While all tested models were susceptible to emergent misalignment, some models showed more resistance than others. Specifically the Qwen-2.5 family proved to be relatively resistant, while GPT-4o exhibited the strongest misalignment. In this paper we evaluate if current-generation open-weights models exhibit similar resistance to the Qwen-2.5 family and measure misalignment robustness over a range of model architectures and scales.   We replicate the effect across nine modern open-weights models (Gemma 3 and Qwen 3 families, 1B-32B parameters). Models fine-tuned on insecure code generation show a 0.68% misalignment rate (compared to 0.07% for base models), matching the lower end of prior open-model results but dramatically lower than GPT-4o's 20%.   We identify a critical format-dependent vulnerability: requiring JSON output doubles misalignment rates compared to natural language prompts (0.96% vs 0.42%). This suggests that structural constraints may bypass safety training by reducing the model's 'degrees of freedom' to refuse. These findings confirm emergent misalignment as a reproducible phenomenon in modern open-weights models, with rates substantially lower than observed in proprietary systems. |
| 2025-11-25 | [WPT: World-to-Policy Transfer via Online World Model Distillation](http://arxiv.org/abs/2511.20095v1) | Guangfeng Jiang, Yueru Luo et al. | Recent years have witnessed remarkable progress in world models, which primarily aim to capture the spatio-temporal correlations between an agent's actions and the evolving environment. However, existing approaches often suffer from tight runtime coupling or depend on offline reward signals, resulting in substantial inference overhead or hindering end-to-end optimization. To overcome these limitations, we introduce WPT, a World-to-Policy Transfer training paradigm that enables online distillation under the guidance of an end-to-end world model. Specifically, we develop a trainable reward model that infuses world knowledge into a teacher policy by aligning candidate trajectories with the future dynamics predicted by the world model. Subsequently, we propose policy distillation and world reward distillation to transfer the teacher's reasoning ability into a lightweight student policy, enhancing planning performance while preserving real-time deployability. Extensive experiments on both open-loop and closed-loop benchmarks show that our WPT achieves state-of-the-art performance with a simple policy architecture: it attains a 0.11 collision rate (open-loop) and achieves a 79.23 driving score (closed-loop) surpassing both world-model-based and imitation-learning methods in accuracy and safety. Moreover, the student sustains up to 4.9x faster inference, while retaining most of the gains. |

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



