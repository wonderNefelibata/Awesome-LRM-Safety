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
| 2025-08-29 | [Going over Fine Web with a Fine-Tooth Comb: Technical Report of Indexing Fine Web for Problematic Content Search and Retrieval](http://arxiv.org/abs/2508.21788v1) | Inés Altemir Marinas, Anastasiia Kucherenko et al. | Large language models (LLMs) rely heavily on web-scale datasets like Common Crawl, which provides over 80\% of training data for some modern models. However, the indiscriminate nature of web crawling raises challenges in data quality, safety, and ethics. Despite the critical importance of training data quality, prior research on harmful content has been limited to small samples due to computational constraints. This project presents a framework for indexing and analyzing LLM training datasets using an ElasticSearch-based pipeline. We apply it to SwissAI's FineWeb-2 corpus (1.5TB, four languages), achieving fast query performance--most searches in milliseconds, all under 2 seconds. Our work demonstrates real-time dataset analysis, offering practical tools for safer, more accountable AI systems. |
| 2025-08-29 | [Adaptive Dead-Zone Dual Sliding Mode Observer for Reliable Electrochemical Model-Based SOC Estimation](http://arxiv.org/abs/2508.21610v1) | Guangdi Hu, Keyi Liao et al. | Accurate state of charge (SOC) estimation is critical for ensuring the safety, reliability, and efficiency of lithium-ion batteries in electric vehicles and energy storage systems. Electrochemical models provide high fidelity for SOC estimation but introduce challenges due to parameter variations, nonlinearities, and computational complexity. To address these issues, this paper proposes an adaptive dead-zone dual sliding mode observer(SMO) based on an improved electrochemical single-particle model. The algorithm integrates a state observer for SOC estimation and a parameter observer for online parameter adaptation. A Lyapunov-derived adaptive dead-zone is introduced to ensure stability, activating parameter updates only when the terminal voltage error lies within a rigorously defined bound. The proposed method was validated under constant-current and UDDS dynamic conditions. Results demonstrate that the adaptive dead-zone dual SMO achieves superior accuracy compared with conventional dual SMO and equivalent circuit model-based EKF methods, maintaining SOC estimation errors within 0.2% under correct initialization and below 1% under a 30% initial SOC error, with rapid convergence. Computational efficiency analysis further shows that the adaptive dead-zone dual sliding mode observer reduces execution time compared with the conventional dual SMO by limiting unnecessary parameter updates, highlighting its suitability for real-time battery management applications. Moreover, robustness under battery aging was confirmed using a cycle-aging model, where the adaptive dead-zone dual SMO maintained stable SOC estimation despite parameter drift. These findings indicate that the proposed method offers a reliable, accurate, and computationally efficient solution for SOC estimation. |
| 2025-08-29 | [Adaptive Dead-Zone Dual Sliding Mode Observer for Reliable Electrochemical Model-Based SOC Estimation](http://arxiv.org/abs/2508.21610v2) | Guangdi Hu, Keyi Liao et al. | Accurate state of charge (SOC) estimation is critical for ensuring the safety, reliability, and efficiency of lithium-ion batteries in electric vehicles and energy storage systems. Electrochemical models provide high fidelity for SOC estimation but introduce challenges due to parameter variations, nonlinearities, and computational complexity. To address these issues, this paper proposes an adaptive dead-zone dual sliding mode observer(SMO) based on an improved electrochemical single-particle model. The algorithm integrates a state observer for SOC estimation and a parameter observer for online parameter adaptation. A Lyapunov-derived adaptive dead-zone is introduced to ensure stability, activating parameter updates only when the terminal voltage error lies within a rigorously defined bound. The proposed method was validated under constant-current and UDDS dynamic conditions. Results demonstrate that the adaptive dead-zone dual SMO achieves superior accuracy compared with conventional dual SMO and equivalent circuit model-based EKF methods, maintaining SOC estimation errors within 0.2% under correct initialization and below 1% under a 30% initial SOC error, with rapid convergence. Computational efficiency analysis further shows that the adaptive dead-zone dual sliding mode observer reduces execution time compared with the conventional dual SMO by limiting unnecessary parameter updates, highlighting its suitability for real-time battery management applications. Moreover, robustness under battery aging was confirmed using a cycle-aging model, where the adaptive dead-zone dual SMO maintained stable SOC estimation despite parameter drift. These findings indicate that the proposed method offers a reliable, accurate, and computationally efficient solution for SOC estimation. |
| 2025-08-29 | [Reduced-Order Modeling of Bolt Loosening: Application to a Pair of Oscillators Under Transverse Shock Excitation](http://arxiv.org/abs/2508.21585v1) | Qirui He, Rui Wang et al. | The safety and integrity of engineered structures are critically dependent on maintaining sufficient preload in their bolted joints. This preload can be dynamically lost due to sustained vibrations or sudden shock that are large enough to induce slip in the threads. While high-fidelity finite element simulations and analytical methods can accurately model the loss of preload for a single, their prohibitive computational expense and complexity render them unfeasible for analyzing large-scale structures with many bolts. This creates a critical need for reduced-order models that capture the essential physics of loosening while remaining computationally efficient. This paper introduces a reduced-order modeling methodology for predicting the loosening of bolted lap joints subjected to transverse shock excitation. The core idea is to treat the bolt tension as a dynamic degree-of-freedom that governs the effective properties of the joint through tension-dependent stiffness and damping that couple the components together. The methodology is applied to a pair of oscillators coupled by with a single lap joint with a strain-sensing bolt. Three different sets of experimental measurements are used to interrogate the dynamics of the system. Mathematical models are identified for the joint stiffness and damping and the instantaneous tension, which are combined with the equations of motion for the oscillators to simulate and reproduce the experimental measurements. Ultimately, the results validate the treatment of bolt tension as a dynamic degree-of-freedom, such that the methodology provides an effective framework for predicting loosening behavior in bolted joints. |
| 2025-08-29 | [Limitations of Physics-Informed Neural Networks: a Study on Smart Grid Surrogation](http://arxiv.org/abs/2508.21559v1) | Julen Cestero, Carmine Delle Femine et al. | Physics-Informed Neural Networks (PINNs) present a transformative approach for smart grid modeling by integrating physical laws directly into learning frameworks, addressing critical challenges of data scarcity and physical consistency in conventional data-driven methods. This paper evaluates PINNs' capabilities as surrogate models for smart grid dynamics, comparing their performance against XGBoost, Random Forest, and Linear Regression across three key experiments: interpolation, cross-validation, and episodic trajectory prediction. By training PINNs exclusively through physics-based loss functions (enforcing power balance, operational constraints, and grid stability) we demonstrate their superior generalization, outperforming data-driven models in error reduction. Notably, PINNs maintain comparatively lower MAE in dynamic grid operations, reliably capturing state transitions in both random and expert-driven control scenarios, while traditional models exhibit erratic performance. Despite slight degradation in extreme operational regimes, PINNs consistently enforce physical feasibility, proving vital for safety-critical applications. Our results contribute to establishing PINNs as a paradigm-shifting tool for smart grid surrogation, bridging data-driven flexibility with first-principles rigor. This work advances real-time grid control and scalable digital twins, emphasizing the necessity of physics-aware architectures in mission-critical energy systems. |
| 2025-08-29 | [Towards a Decentralized IoT Onboarding for Smart Homes Using Consortium Blockchain](http://arxiv.org/abs/2508.21480v1) | Narges Dadkhah, Khan Reaz et al. | The increasing adoption of smart home devices and IoT-based security systems presents significant opportunities to enhance convenience, safety, and risk management for homeowners and service providers. However, secure onboarding-provisioning credentials and establishing trust with cloud platforms-remains a considerable challenge. Traditional onboarding methods often rely on centralized Public Key Infrastructure (PKI) models and manufacturer-controlled keys, which introduce security risks and limit the user's digital sovereignty. These limitations hinder the widespread deployment of scalable IoT solutions. This paper presents a novel onboarding framework that builds upon existing network-layer onboarding techniques and extends them to the application layer to address these challenges. By integrating consortium blockchain technology, we propose a decentralized onboarding mechanism that enhances transparency, security, and monitoring for smart home architectures. The architecture supports device registration, key revocation, access control management, and risk detection through event-driven alerts across dedicated blockchain channels and smart contracts. To evaluate the framework, we formally model the protocol using the Tamarin Prover under the Dolev-Yao adversary model. The analysis focuses on authentication, token integrity, key confidentiality, and resilience over public channels. A prototype implementation demonstrates the system's viability in smart home settings, with verification completing in 0.34 seconds, highlighting its scalability and suitability for constrained devices and diverse stakeholders. Additionally, performance evaluation shows that the blockchain-based approach effectively handles varying workloads, maintains high throughput and low latency, and supports near real-time IoT data processing. |
| 2025-08-29 | [MMSearch-Plus: A Simple Yet Challenging Benchmark for Multimodal Browsing Agents](http://arxiv.org/abs/2508.21475v1) | Xijia Tao, Yihua Teng et al. | Large multimodal language models (MLLMs) are increasingly deployed as web agents, yet many multimodal browsing benchmarks can be solved by shallow, fixed workflows that lean on high-recall image search and nearby text-masking the genuinely multimodal challenges of fine-grained visual reasoning, provenance verification, and long-horizon tool use. We introduce MMSearch-Plus, a benchmark of 311 tasks that highly demand multimodal understanding while preserving the difficulty profile of strong text-only browsing suites. Each item is constructed to contain multiple weak, localized visual signals that must be extracted, propagated through iterative text-image search, and cross-validated under retrieval noise before answering. Our curation procedure, Spatial-Temporal Extrapolation, seeds questions whose answers require extrapolating from spatial cues (micro-text, part-level appearance, layouts, signage) and temporal traces (broadcast overlays, seasonal context) to out-of-image facts such as events, dates, and venues. We provide a model-agnostic agent framework with browsing tools and evaluate a range of closed and open MLLMs. The strongest agent (o3) attains 15.1% without search and 36.0% accuracy with rollout under our framework, while a strong open-source model (Qwen-2.5-VL-72B-Instruct) achieves 0.0% without search and 6.9% after 20 rounds of search. Beyond answer accuracy, we assess bounding-box production and cropped-image search, and conduct an error analysis that surfaces failures in source verification, part-based reasoning, and long-horizon planning. |
| 2025-08-29 | [Multi-Method Ensemble for Out-of-Distribution Detection](http://arxiv.org/abs/2508.21463v1) | Lucas Rakotoarivony | Detecting out-of-distribution (OOD) samples is essential for neural networks operating in open-world settings, particularly in safety-critical applications. Existing methods have improved OOD detection by leveraging two main techniques: feature truncation, which increases the separation between in-distribution (ID) and OOD samples, and scoring functions, which assign scores to distinguish between ID and OOD data. However, most approaches either focus on a single family of techniques or evaluate their effectiveness on a specific type of OOD dataset, overlooking the potential of combining multiple existing solutions. Motivated by this observation, we theoretically and empirically demonstrate that state-of-the-art feature truncation and scoring functions can be effectively combined. Moreover, we show that aggregating multiple scoring functions enhances robustness against various types of OOD samples. Based on these insights, we propose the Multi-Method Ensemble (MME) score, which unifies state-of-the-art OOD detectors into a single, more effective scoring function. Extensive experiments on both large-scale and small-scale benchmarks, covering near-OOD and far-OOD scenarios, show that MME significantly outperforms recent state-of-the-art methods across all benchmarks. Notably, using the BiT model, our method achieves an average FPR95 of 27.57% on the challenging ImageNet-1K benchmark, improving performance by 6% over the best existing baseline. |
| 2025-08-29 | [Challenges and Applications of Large Language Models: A Comparison of GPT and DeepSeek family of models](http://arxiv.org/abs/2508.21377v1) | Shubham Sharma, Sneha Tuli et al. | Large Language Models (LLMs) are transforming AI across industries, but their development and deployment remain complex. This survey reviews 16 key challenges in building and using LLMs and examines how these challenges are addressed by two state-of-the-art models with unique approaches: OpenAI's closed source GPT-4o (May 2024 update) and DeepSeek-V3-0324 (March 2025), a large open source Mixture-of-Experts model. Through this comparison, we showcase the trade-offs between closed source models (robust safety, fine-tuned reliability) and open source models (efficiency, adaptability). We also explore LLM applications across different domains (from chatbots and coding tools to healthcare and education), highlighting which model attributes are best suited for each use case. This article aims to guide AI researchers, developers, and decision-makers in understanding current LLM capabilities, limitations, and best practices. |
| 2025-08-29 | [AHELM: A Holistic Evaluation of Audio-Language Models](http://arxiv.org/abs/2508.21376v1) | Tony Lee, Haoqin Tu et al. | Evaluations of audio-language models (ALMs) -- multimodal models that take interleaved audio and text as input and output text -- are hindered by the lack of standardized benchmarks; most benchmarks measure only one or two capabilities and omit evaluative aspects such as fairness or safety. Furthermore, comparison across models is difficult as separate evaluations test a limited number of models and use different prompting methods and inference parameters. To address these shortfalls, we introduce AHELM, a benchmark that aggregates various datasets -- including 2 new synthetic audio-text datasets called PARADE, which evaluates the ALMs on avoiding stereotypes, and CoRe-Bench, which measures reasoning over conversational audio through inferential multi-turn question answering -- to holistically measure the performance of ALMs across 10 aspects we have identified as important to the development and usage of ALMs: audio perception, knowledge, reasoning, emotion detection, bias, fairness, multilinguality, robustness, toxicity, and safety. We also standardize the prompts, inference parameters, and evaluation metrics to ensure equitable comparisons across models. We test 14 open-weight and closed-API ALMs from 3 developers and 3 additional simple baseline systems each consisting of an automatic speech recognizer and a language model. Our results show that while Gemini 2.5 Pro ranks top in 5 out of 10 aspects, it exhibits group unfairness ($p=0.01$) on ASR tasks whereas most of the other models do not. We also find that the baseline systems perform reasonably well on AHELM, with one ranking 5th overall despite having only speech-to-text capabilities. For transparency, all raw prompts, model generations, and outputs are available on our website at https://crfm.stanford.edu/helm/audio/v1.0.0. AHELM is intended to be a living benchmark and new datasets and models will be added over time. |
| 2025-08-29 | [Robust Real-Time Coordination of CAVs: A Distributed Optimization Framework under Uncertainty](http://arxiv.org/abs/2508.21322v1) | Haojie Bai, Yang Wang et al. | Achieving both safety guarantees and real-time performance in cooperative vehicle coordination remains a fundamental challenge, particularly in dynamic and uncertain environments. This paper presents a novel coordination framework that resolves this challenge through three key innovations: 1) direct control of vehicles' trajectory distributions during coordination, formulated as a robust cooperative planning problem with adaptive enhanced safety constraints, ensuring a specified level of safety regarding the uncertainty of the interactive trajectory, 2) a fully parallel ADMM-based distributed trajectory negotiation (ADMM-DTN) algorithm that efficiently solves the optimization problem while allowing configurable negotiation rounds to balance solution quality and computational resources, and 3) an interactive attention mechanism that selectively focuses on critical interactive participants to further enhance computational efficiency. Both simulation results and practical experiments demonstrate that our framework achieves significant advantages in safety (reducing collision rates by up to 40.79\% in various scenarios) and real-time performance compared to state-of-the-art methods, while maintaining strong scalability with increasing vehicle numbers. The proposed interactive attention mechanism further reduces the computational demand by 14.1\%. The framework's effectiveness is further validated through real-world experiments with unexpected dynamic obstacles, demonstrating robust coordination in complex environments. The experiment demo could be found at https://youtu.be/4PZwBnCsb6Q. |
| 2025-08-29 | [One-Loop Nonlinear Matter Power Spectrum from Unified Lagrangian Perturbation Theory: Fast Computation and Comparison with Emulators](http://arxiv.org/abs/2508.21275v1) | Naonori Sugiyama | We present a fast and accurate formulation for computing the nonlinear matter power spectrum at one-loop order based on Unified Lagrangian Perturbation Theory (ULPT). ULPT decomposes the density field into the Jacobian deviation, capturing intrinsic nonlinear growth, and the displacement-mapping factor, accounting for large-scale distortions due to bulk flows. This structural separation leads to a natural division of the power spectrum into a source term and a displacement-mapping factor, ensuring infrared (IR) safety by construction. We implement an efficient numerical algorithm using FFTLog and FAST-PT, achieving approximately 2-second evaluations on a standard laptop. The results are validated against simulation-based emulators, including the Dark Emulator and Euclid Emulator 2. Across 100 sampled cosmologies, ULPT agrees with emulator predictions at the 2--3\% level up to \( k \simeq 0.4\,h\,\mathrm{Mpc}^{-1} \) for \( z \geq 0.5 \), without any nuisance parameters. Similar agreement is found in configuration space, where the two-point correlation function remains accurate down to \( r \simeq 10\,h^{-1}\mathrm{Mpc} \). Compared to standard perturbation theory, which fails at small scales due to series expansion of the displacement factor, ULPT maintains convergence by preserving its full exponential form. We also clarify the mechanism of BAO damping: exponential suppression by displacement and peak sharpening by nonlinear growth. The combination accurately reproduces BAO features seen in simulations. ULPT thus offers a robust, IR-safe, and computationally efficient framework for modeling large-scale structure in galaxy surveys. The numerical implementation developed in this work is publicly released as the open-source Python package \texttt{ulptkit} (https://github.com/naonori/ulptkit). |
| 2025-08-29 | [Mini Autonomous Car Driving based on 3D Convolutional Neural Networks](http://arxiv.org/abs/2508.21271v1) | Pablo Moraes, Monica Rodriguez et al. | Autonomous driving applications have become increasingly relevant in the automotive industry due to their potential to enhance vehicle safety, efficiency, and user experience, thereby meeting the growing demand for sophisticated driving assistance features. However, the development of reliable and trustworthy autonomous systems poses challenges such as high complexity, prolonged training periods, and intrinsic levels of uncertainty. Mini Autonomous Cars (MACs) are used as a practical testbed, enabling validation of autonomous control methodologies on small-scale setups. This simplified and cost-effective environment facilitates rapid evaluation and comparison of machine learning models, which is particularly useful for algorithms requiring online training. To address these challenges, this work presents a methodology based on RGB-D information and three-dimensional convolutional neural networks (3D CNNs) for MAC autonomous driving in simulated environments. We evaluate the proposed approach against recurrent neural networks (RNNs), with architectures trained and tested on two simulated tracks with distinct environmental features. Performance was assessed using task completion success, lap-time metrics, and driving consistency. Results highlight how architectural modifications and track complexity influence the models' generalization capability and vehicle control performance. The proposed 3D CNN demonstrated promising results when compared with RNNs. |
| 2025-08-28 | [Improving Aviation Safety Analysis: Automated HFACS Classification Using Reinforcement Learning with Group Relative Policy Optimization](http://arxiv.org/abs/2508.21201v1) | Arash Ahmadi, Sarah Sharif et al. | Analyzing the human factors behind aviation accidents is crucial for preventing future incidents, yet traditional methods using the Human Factors Analysis and Classification System (HFACS) are limited by scalability and consistency. To address this, we introduce an automated HFACS classification framework for aviation safety analysis that utilizes Reinforcement Learning with Group Relative Policy Optimization (GRPO) to fine-tune a Llama-3.1 8B language model. Our approach incorporates a multi-component reward system tailored for aviation safety analysis and integrates synthetic data generation to overcome class imbalance in accident datasets. The resulting GRPO-optimized model achieved noticeable performance gains, including a 350% increase in exact match accuracy (from 0.0400 to 0.1800) and an improved partial match accuracy of 0.8800. Significantly, our specialized model outperforms state-of-the-art LLMs (Large Language Models), including GPT-5-mini and Gemini-2.5-fiash, on key metrics. This research also proposes exact match accuracy in multi-label HFACS classification problem as a new benchmarking methodology to evaluate the advanced reasoning capabilities of language models. Ultimately, our work validates that smaller, domain-optimized models can provide a computationally efficient and better solution for critical safety analysis. This approach makes powerful, low-latency deployment on resource-constrained edge devices feasible. |
| 2025-08-28 | [Can Multimodal LLMs Solve the Basic Perception Problems of Percept-V?](http://arxiv.org/abs/2508.21143v1) | Samrajnee Ghosh, Naman Agarwal et al. | The reasoning abilities of Multimodal Large Language Models (MLLMs) have garnered a lot of attention in recent times, with advances made in frontiers like coding, mathematics, and science. However, very limited experiments have been done to assess their performance in simple perception tasks performed over uncontaminated, generated images containing basic shapes and structures. To address this issue, the paper introduces a dataset, Percept-V, containing a total of 7200 program-generated images equally divided into 30 categories, each testing a combination of visual perception skills. Unlike previously proposed datasets, Percept-V comprises very basic tasks of varying complexity that test the perception abilities of MLLMs. This dataset is then tested on state-of-the-art MLLMs like GPT-4o, Gemini, and Claude as well as Large Reasoning Models (LRMs) like OpenAI o4-mini and DeepSeek R1 to gauge their performance. Contrary to the evidence that MLLMs excel in many complex tasks, our experiments show a significant drop in the models' performance with increasing problem complexity across all categories. An analysis of the performances also reveals that the tested MLLMs exhibit a similar trend in accuracy across categories, testing a particular cognitive skill and find some skills to be more difficult than others. |
| 2025-08-28 | [Lithiation Analysis of Metal Components for Li-Ion Battery using Ion Beams](http://arxiv.org/abs/2508.21017v1) | Arturo Galindo, Neubi Xavier et al. | Metal components are extensively used as current collectors, anodes, and interlayers in lithium-ion batteries. Integrating these functions into one component enhances the cell energy density and simplifies its design. However, this multifunctional component must meet stringent requirements, including high and reversible Li storage capacity, rapid lithiation/delithiation kinetics, mechanical stability, and safety. Six single-atom metals (Mg, Zn, Al, Ag, Sn and Cu) are screened for lithiation behavior through their interaction with ion beams in electrochemically tested samples subjected to both weak and strong lithiation regimes. These different lithiation regimes allowed us to differentiate between the thermodynamics and kinetic aspects of the lithiation process. Three types of ions are used to determine Li depth profile: $H^+$ for nuclear reaction analysis (NRA), $He^+$ for Rutherford backscattering (RBS), and $Ga^+$ for focused ion beam (FIB) milling. The study reveals three lithiation behaviors: (i) Zn, Al, Sn form pure alloys with Li; (ii) Mg, Ag create intercalation solid solutions; (iii) Cu acts as a lithiation barrier. NRA and RBS offer direct and quantitative data, providing a more comprehensive understanding of the lithiation process in LIB components. These findings fit well with our ab-initio simulation results, establishing a direct correlation between electrochemical features and fundamental thermodynamic parameters. |
| 2025-08-28 | [Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees](http://arxiv.org/abs/2508.21001v1) | Yaniv Hassidof, Tom Jurgenson et al. | Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: https://sites.google.com/view/ditree. |
| 2025-08-28 | [ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents](http://arxiv.org/abs/2508.20973v1) | Tianjian Liu, Fanqi Wan et al. | Proactive dialogue has emerged as a critical and challenging research problem in advancing large language models (LLMs). Existing works predominantly focus on domain-specific or task-oriented scenarios, which leads to fragmented evaluations and limits the comprehensive exploration of models' proactive conversation abilities. In this work, we propose ProactiveEval, a unified framework designed for evaluating proactive dialogue capabilities of LLMs. This framework decomposes proactive dialogue into target planning and dialogue guidance, establishing evaluation metrics across various domains. Moreover, it also enables the automatic generation of diverse and challenging evaluation data. Based on the proposed framework, we develop 328 evaluation environments spanning 6 distinct domains. Through experiments with 22 different types of LLMs, we show that DeepSeek-R1 and Claude-3.7-Sonnet exhibit exceptional performance on target planning and dialogue guidance tasks, respectively. Finally, we investigate how reasoning capabilities influence proactive behaviors and discuss their implications for future model development. |
| 2025-08-28 | [AI Reasoning Models for Problem Solving in Physics](http://arxiv.org/abs/2508.20941v1) | Amir Bralin, N. Sanjay Rebello | Reasoning models are the new generation of Large Language Models (LLMs) capable of complex problem solving. Their reliability in solving introductory physics problems was tested by evaluating a sample of n = 5 solutions generated by one such model -- OpenAI's o3-mini -- per each problem from 20 chapters of a standard undergraduate textbook. In total, N = 408 problems were given to the model and N x n = 2,040 generated solutions examined. The model successfully solved 94% of the problems posed, excelling at the beginning topics in mechanics but struggling with the later ones such as waves and thermodynamics. |
| 2025-08-28 | [COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans](http://arxiv.org/abs/2508.20920v1) | Enrico Martini, Ho Jin Choi et al. | In the era of Industry 5.0, monitoring human activity is essential for ensuring both ergonomic safety and overall well-being. While multi-camera centralized setups improve pose estimation accuracy, they often suffer from high computational costs and bandwidth requirements, limiting scalability and real-time applicability. Distributing processing across edge devices can reduce network bandwidth and computational load. On the other hand, the constrained resources of edge devices lead to accuracy degradation, and the distribution of computation leads to temporal and spatial inconsistencies. We address this challenge by proposing COMETH (Convex Optimization for Multiview Estimation and Tracking of Humans), a lightweight algorithm for real-time multi-view human pose fusion that relies on three concepts: it integrates kinematic and biomechanical constraints to increase the joint positioning accuracy; it employs convex optimization-based inverse kinematics for spatial fusion; and it implements a state observer to improve temporal consistency. We evaluate COMETH on both public and industrial datasets, where it outperforms state-of-the-art methods in localization, detection, and tracking accuracy. The proposed fusion pipeline enables accurate and scalable human motion tracking, making it well-suited for industrial and safety-critical applications. The code is publicly available at https://github.com/PARCO-LAB/COMETH. |

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



