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
| 2025-11-10 | [TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research](http://arxiv.org/abs/2511.07412v1) | Han Zhang, Yiqing Shen et al. | Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit embodied agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create photorealistic and dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains unclear. We introduce TwinOR, a framework for constructing photorealistic, dynamic digital twins of ORs for embodied AI research. The system reconstructs static geometry from pre-scan videos and continuously models human and equipment motion through multi-view perception of OR activities. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter level accuracy while preserving dynamic interaction across surgical workflows, enabling realistic renderings and a virtual playground for embodied AI systems. In our experiments, TwinOR simulates stereo and monocular sensor streams for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 on TwinOR-synthesized data achieve performance within their reported accuracy on real indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for perception and localization challenges. By establishing a real-to-sim pipeline for constructing dynamic, photorealistic digital twins of OR environments, TwinOR enables the safe, scalable, and data-efficient development and benchmarking of embodied AI, ultimately accelerating the deployment of embodied AI from sim-to-real. |
| 2025-11-10 | [Unified Humanoid Fall-Safety Policy from a Few Demonstrations](http://arxiv.org/abs/2511.07407v1) | Zhengjie Xu, Ye Li et al. | Falling is an inherent risk of humanoid mobility. Maintaining stability is thus a primary safety focus in robot control and learning, yet no existing approach fully averts loss of balance. When instability does occur, prior work addresses only isolated aspects of falling: avoiding falls, choreographing a controlled descent, or standing up afterward. Consequently, humanoid robots lack integrated strategies for impact mitigation and prompt recovery when real falls defy these scripts. We aim to go beyond keeping balance to make the entire fall-and-recovery process safe and autonomous: prevent falls when possible, reduce impact when unavoidable, and stand up when fallen. By fusing sparse human demonstrations with reinforcement learning and an adaptive diffusion-based memory of safe reactions, we learn adaptive whole-body behaviors that unify fall prevention, impact mitigation, and rapid recovery in one policy. Experiments in simulation and on a Unitree G1 demonstrate robust sim-to-real transfer, lower impact forces, and consistently fast recovery across diverse disturbances, pointing towards safer, more resilient humanoids in real environments. Videos are available at https://firm2025.github.io/. |
| 2025-11-10 | [Coexistence of Ferroelectric and Relaxor-like Phases in a Multiferroic Solid Solution (1-x)Pb(Fe1/2Nb1/2)O3 - xPbMnO3](http://arxiv.org/abs/2511.07371v1) | Anna N. Morozovska, Victor N. Pavlikov et al. | Experimental and theoretical studies of unusual polar, dielectric and magnetic properties of room temperature multiferroics, such as perovskites Pb(Fe1/2Nb1/2)O3 (PFN) and Pb(Fe1/2Ta1/2)O3 (PFT), are very important. We study the phase composition, dielectric, ferroic properties of the solid solutions PFN and PFT substituted with 5, 10, 15, 20 and 30 % of Mn ions prepared by the solid-state synthesis. The XRD analysis confirmed the perovskite structure of sintered ceramics. Electric measurements revealed the ferroelectric-type hysteresis of electric charge in pure PFN ceramics and in PFN ceramics substituted with (10 - 30)% of Mn. At the same time, the PFN-5% Mn ceramics did not show any ferroelectric properties due to very high conductivity.Temperature dependences of the dielectric permittivity of PFN-10% Mn and PFN-15% Mn ceramics have two pronounced maxima, one of which is relatively sharp and has a weak frequency dispersion; another is diffuse and has a strong frequency dispersion. A further increase in the Mn content up to 20% leads to the right shift in the paraelectric-ferroelectric phase transition temperature, as well as to the strong suppression of the second wide maximum, which transforms into a small diffuse shoulder. An increase in the Mn substitution up to 30% leads to a significant decrease in the dielectric permittivity, left shift of its maximum, and induces a pronounced frequency dispersion of the paraelectric-ferroelectric transition temperature, which is inherent to relaxor-like ferroelectrics.Comparison of the model with experiments reveal the coexistence of the ordered ferroelectric-like and disordered relaxor-like phases in the multiferroic solid solutions PFN-Mn and PFT-Mn. |
| 2025-11-10 | [Robust Linear Design for Flight Control Systems with Operational Constraints](http://arxiv.org/abs/2511.07335v1) | Marcel Menner, Eugene Lavretsky | This paper presents a systematic approach for designing robust linear proportional-integral (PI) servo-controllers that effectively manage control input and output constraints in flight control systems. The control design leverages the Nagumo Theorem and the Comparison Lemma to prove constraint satisfaction, while employing min-norm optimal controllers in a manner akin to Control Barrier Functions. This results in a continuous piecewise-linear state feedback policy that maintains the analyzability of the closed-loop system through the principles of linear systems theory. Additionally, we derive multi-input multi-output (MIMO) robustness margins, demonstrating that our approach enables robust tracking of external commands even in the presence of operational constraints. Moreover, the proposed control design offers a systematic approach for anti-windup protection. Through flight control trade studies, we illustrate the applicability of the proposed framework to real-world safety-critical aircraft control scenarios. Notably, MIMO margin analysis with active constraints reveals that our method preserves gain and phase margins comparable to those of the unconstrained case, in contrast to controllers that rely on hard saturation heuristics, which suffer significant performance degradation under active constraints. Simulation results using a nonlinear six-degree-of-freedom rigid body aircraft model further validate the effectiveness of our method in achieving constraint satisfaction, robustness, and effective anti-windup protection. |
| 2025-11-10 | [Grounding Computer Use Agents on Human Demonstrations](http://arxiv.org/abs/2511.07332v1) | Aarash Feizi, Shravan Nayak et al. | Building reliable computer-use agents requires grounding: accurately connecting natural language instructions to the correct on-screen elements. While large datasets exist for web and mobile interactions, high-quality resources for desktop environments are limited. To address this gap, we introduce GroundCUA, a large-scale desktop grounding dataset built from expert human demonstrations. It covers 87 applications across 12 categories and includes 56K screenshots, with every on-screen element carefully annotated for a total of over 3.56M human-verified annotations. From these demonstrations, we generate diverse instructions that capture a wide range of real-world tasks, providing high-quality data for model training. Using GroundCUA, we develop the GroundNext family of models that map instructions to their target UI elements. At both 3B and 7B scales, GroundNext achieves state-of-the-art results across five benchmarks using supervised fine-tuning, while requiring less than one-tenth the training data of prior work. Reinforcement learning post-training further improves performance, and when evaluated in an agentic setting on the OSWorld benchmark using o3 as planner, GroundNext attains comparable or superior results to models trained with substantially more data,. These results demonstrate the critical role of high-quality, expert-driven datasets in advancing general-purpose computer-use agents. |
| 2025-11-10 | [Roundabout Constrained Convex Generators: A Unified Framework for Multiply-Connected Reachable Sets](http://arxiv.org/abs/2511.07330v1) | Peng Xie, Sabin Diaconescu et al. | This paper introduces Roundabout Constrained Convex Generators (RCGs), a set representation framework for modeling multiply connected regions in control and verification applications. The RCG representation extends the constrained convex generators framework by incorporating an inner exclusion zone, creating sets with topological holes that naturally arise in collision avoidance and safety-critical control problems. We present two equivalent formulations: a set difference representation that provides geometric intuition and a unified parametric representation that facilitates computational implementation. The paper establishes closure properties under fundamental operations, including linear transformations, Minkowski sums, and intersections with convex generator sets. We derive special cases, including roundabout zonotopes and roundabout ellipsotopes, which offer computational advantages for specific norm selections. The framework maintains compatibility with existing optimization solvers while enabling the representation of non-convex feasible regions that were previously challenging to model efficiently. |
| 2025-11-10 | [JPRO: Automated Multimodal Jailbreaking via Multi-Agent Collaboration Framework](http://arxiv.org/abs/2511.07315v1) | Yuxuan Zhou, Yang Bai et al. | The widespread application of large VLMs makes ensuring their secure deployment critical. While recent studies have demonstrated jailbreak attacks on VLMs, existing approaches are limited: they require either white-box access, restricting practicality, or rely on manually crafted patterns, leading to poor sample diversity and scalability. To address these gaps, we propose JPRO, a novel multi-agent collaborative framework designed for automated VLM jailbreaking. It effectively overcomes the shortcomings of prior methods in attack diversity and scalability. Through the coordinated action of four specialized agents and its two core modules: Tactic-Driven Seed Generation and Adaptive Optimization Loop, JPRO generates effective and diverse attack samples. Experimental results show that JPRO achieves over a 60\% attack success rate on multiple advanced VLMs, including GPT-4o, significantly outperforming existing methods. As a black-box attack approach, JPRO not only uncovers critical security vulnerabilities in multimodal models but also offers valuable insights for evaluating and enhancing VLM robustness. |
| 2025-11-10 | [LMM-IQA: Image Quality Assessment for Low-Dose CT Imaging](http://arxiv.org/abs/2511.07298v1) | Kagan Celik, Mehmet Ozan Unal et al. | Low-dose computed tomography (CT) represents a significant improvement in patient safety through lower radiation doses, but increased noise, blur, and contrast loss can diminish diagnostic quality. Therefore, consistency and robustness in image quality assessment become essential for clinical applications. In this study, we propose an LLM-based quality assessment system that generates both numerical scores and textual descriptions of degradations such as noise, blur, and contrast loss. Furthermore, various inference strategies - from the zero-shot approach to metadata integration and error feedback - are systematically examined, demonstrating the progressive contribution of each method to overall performance. The resultant assessments yield not only highly correlated scores but also interpretable output, thereby adding value to clinical workflows. The source codes of our study are available at https://github.com/itu-biai/lmms_ldct_iqa. |
| 2025-11-10 | [Verifying rich robustness properties for neural networks](http://arxiv.org/abs/2511.07293v1) | Mohammad Afzal, S. Akshay et al. | Robustness is a important problem in AI alignment and safety, with models such as neural networks being increasingly used in safety-critical systems. In the last decade, a large body of work has emerged on local robustness, i.e., checking if the decision of a neural network remains unchanged when the input is slightly perturbed. However, many of these approaches require specialized encoding and often ignore the confidence of a neural network on its output. In this paper, our goal is to build a generalized framework to specify and verify variants of robustness in neural network verification. We propose a specification framework using a simple grammar, which is flexible enough to capture most existing variants. This allows us to introduce new variants of robustness that take into account the confidence of the neural network in its outputs. Next, we develop a novel and powerful unified technique to verify all such variants in a homogeneous way, viz., by adding a few additional layers to the neural network. This enables us to use any state-of-the-art neural network verification tool, without having to tinker with the encoding within, while incurring an approximation error that we show is bounded. We perform an extensive experimental evaluation over a large suite of 8870 benchmarks having 138M parameters in a largest network, and show that we are able to capture a wide set of robustness variants and outperform direct encoding approaches by a significant margin. |
| 2025-11-10 | [Offset-Free Robust Nonlinear Control Using Data-Driven Model: A Nonlinear Multi-Model Computationally Efficient Approach](http://arxiv.org/abs/2511.07255v1) | Carine Menezes Rebello, Erbet Almeida Costa et al. | Robust model predictive control (MPC) aims to preserve performance under model-plant mismatch, yet robust formulations for nonlinear MPC (NMPC) with data-driven surrogates remain limited. This work proposes an offset-free robust NMPC scheme based on symbolic regression (SR). Using a compact NARX structure, we identify interpretable surrogate models that explicitly represent epistemic (structural) uncertainty at the operating-zone level, and we enforce robustness by embedding these models as hard constraints. gest margins. The single-zone variant (RNMPC$_{SZ}$) was investigated. Synthetic data were generated within one operating zone to identify SR models that are embedded as constraints, while the nominal predictor remains fixed, and the multi-zone variant (RNMPC$_{MZ}$), zone-specific SR models from multiple operating zones are jointly enforced in the constraint set; at each set-point change, the nominal predictor is re-scheduled to the SR model of the newly active zone. In both cases, robustness is induced by the intersection of the admissible sets defined by the enforced SR models, without modifying the nominal cost or introducing ancillary tube dynamics. The approach was validated on a simulated pilot-scale electric submersible pump (ESP) system with pronounced nonlinearities and dynamically varying safety envelopes (downthrust and upthrust). RNMPC$_{SZ}$ and RNMPC$_{MZ}$ maintained disturbance tracking and rejection, and by intersecting models in the constraints, they increased margins and eliminated violations (especially near the upthrust), with a slight increase in settling time. Including up to four models per zone did not increase the time per iteration, maintaining real-time viability; RNMPC$_{MZ}$ presented the lar |
| 2025-11-10 | [Leveraging Text-Driven Semantic Variation for Robust OOD Segmentation](http://arxiv.org/abs/2511.07238v1) | Seungheon Song, Jaekoo Lee | In autonomous driving and robotics, ensuring road safety and reliable decision-making critically depends on out-of-distribution (OOD) segmentation. While numerous methods have been proposed to detect anomalous objects on the road, leveraging the vision-language space-which provides rich linguistic knowledge-remains an underexplored field. We hypothesize that incorporating these linguistic cues can be especially beneficial in the complex contexts found in real-world autonomous driving scenarios.   To this end, we present a novel approach that trains a Text-Driven OOD Segmentation model to learn a semantically diverse set of objects in the vision-language space. Concretely, our approach combines a vision-language model's encoder with a transformer decoder, employs Distance-Based OOD prompts located at varying semantic distances from in-distribution (ID) classes, and utilizes OOD Semantic Augmentation for OOD representations. By aligning visual and textual information, our approach effectively generalizes to unseen objects and provides robust OOD segmentation in diverse driving environments.   We conduct extensive experiments on publicly available OOD segmentation datasets such as Fishyscapes, Segment-Me-If-You-Can, and Road Anomaly datasets, demonstrating that our approach achieves state-of-the-art performance across both pixel-level and object-level evaluations. This result underscores the potential of vision-language-based OOD segmentation to bolster the safety and reliability of future autonomous driving systems. |
| 2025-11-10 | [Mapping Reduced Accessibility to WASH Facilities in Rohingya Refugee Camps with Sub-Meter Imagery](http://arxiv.org/abs/2511.07231v1) | Kyeongjin Ahn, YongHun Suh et al. | Access to Water, Sanitation, and Hygiene (WASH) services remains a major public health concern in refugee camps. This study introduces a remote sensing-driven framework to quantify WASH accessibility-specifically to water pumps, latrines, and bathing cubicles-in the Rohingya camps of Cox's Bazar, one of the world's most densely populated displacement settings. Detecting refugee shelters in such emergent camps presents substantial challenges, primarily due to their dense spatial configuration and irregular geometric patterns. Using sub-meter satellite images, we develop a semi-supervised segmentation framework that achieves an F1-score of 76.4% in detecting individual refugee shelters. Applying the framework across multi-year data reveals declining WASH accessibility, driven by rapid refugee population growth and reduced facility availability, rising from 25 people per facility in 2022 to 29.4 in 2025. Gender-disaggregated analysis further shows that women and girls experience reduced accessibility, in scenarios with inadequate safety-related segregation in WASH facilities. These findings suggest the importance of demand-responsive allocation strategies that can identify areas with under-served populations-such as women and girls-and ensure that limited infrastructure serves the greatest number of people in settings with fixed or shrinking budgets. We also discuss the value of high-resolution remote sensing and machine learning to detect inequality and inform equitable resource planning in complex humanitarian environments. |
| 2025-11-10 | [MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks](http://arxiv.org/abs/2511.07107v1) | Liang Shan, Kaicheng Shen et al. | Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment. |
| 2025-11-10 | [Conservative Software Reliability Assessments Using Collections of Bayesian Inference Problems](http://arxiv.org/abs/2511.07038v1) | Kizito Salako, Rabiu Tsoho Muhammad | When using Bayesian inference to support conservative software reliability assessments, it is useful to consider a collection of Bayesian inference problems, with the aim of determining the worst-case value (from this collection) for a posterior predictive probability that characterizes how reliable the software is. Using a Bernoulli process to model the occurrence of software failures, we explicitly determine (from collections of Bayesian inference problems) worst-case posterior predictive probabilities of the software operating without failure in the future. We deduce asymptotic properties of these conservative posterior probabilities and their priors, and illustrate how to use these results in assessments of safety-critical software. This work extends robust Bayesian inference results and so-called conservative Bayesian inference methods. |
| 2025-11-10 | [Certified L2-Norm Robustness of 3D Point Cloud Recognition in the Frequency Domain](http://arxiv.org/abs/2511.07029v1) | Liang Zhou, Qiming Wang et al. | 3D point cloud classification is a fundamental task in safety-critical applications such as autonomous driving, robotics, and augmented reality. However, recent studies reveal that point cloud classifiers are vulnerable to structured adversarial perturbations and geometric corruptions, posing risks to their deployment in safety-critical scenarios. Existing certified defenses limit point-wise perturbations but overlook subtle geometric distortions that preserve individual points yet alter the overall structure, potentially leading to misclassification. In this work, we propose FreqCert, a novel certification framework that departs from conventional spatial domain defenses by shifting robustness analysis to the frequency domain, enabling structured certification against global L2-bounded perturbations. FreqCert first transforms the input point cloud via the graph Fourier transform (GFT), then applies structured frequency-aware subsampling to generate multiple sub-point clouds. Each sub-cloud is independently classified by a standard model, and the final prediction is obtained through majority voting, where sub-clouds are constructed based on spectral similarity rather than spatial proximity, making the partitioning more stable under L2 perturbations and better aligned with the object's intrinsic structure. We derive a closed-form lower bound on the certified L2 robustness radius and prove its tightness under minimal and interpretable assumptions, establishing a theoretical foundation for frequency domain certification. Extensive experiments on the ModelNet40 and ScanObjectNN datasets demonstrate that FreqCert consistently achieves higher certified accuracy and empirical accuracy under strong perturbations. Our results suggest that spectral representations provide an effective pathway toward certifiable robustness in 3D point cloud recognition. |
| 2025-11-10 | [EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers](http://arxiv.org/abs/2511.06890v1) | Yilin Jiang, Mingzi Zhang et al. | Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See https://github.com/YL1N/EduGuardBench for Materials. |
| 2025-11-10 | [Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment](http://arxiv.org/abs/2511.06852v1) | Peng Zhang, peijie sun | Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment. |
| 2025-11-10 | [Vision-Aided Online A* Path Planning for Efficient and Safe Navigation of Service Robots](http://arxiv.org/abs/2511.06801v1) | Praveen Kumar, Tushar Sandhan | The deployment of autonomous service robots in human-centric environments is hindered by a critical gap in perception and planning. Traditional navigation systems rely on expensive LiDARs that, while geometrically precise, are seman- tically unaware, they cannot distinguish a important document on an office floor from a harmless piece of litter, treating both as physically traversable. While advanced semantic segmentation exists, no prior work has successfully integrated this visual intelligence into a real-time path planner that is efficient enough for low-cost, embedded hardware. This paper presents a frame- work to bridge this gap, delivering context-aware navigation on an affordable robotic platform. Our approach centers on a novel, tight integration of a lightweight perception module with an online A* planner. The perception system employs a semantic segmentation model to identify user-defined visual constraints, enabling the robot to navigate based on contextual importance rather than physical size alone. This adaptability allows an operator to define what is critical for a given task, be it sensitive papers in an office or safety lines in a factory, thus resolving the ambiguity of what to avoid. This semantic perception is seamlessly fused with geometric data. The identified visual constraints are projected as non-geometric obstacles onto a global map that is continuously updated from sensor data, enabling robust navigation through both partially known and unknown environments. We validate our framework through extensive experiments in high-fidelity simulations and on a real-world robotic platform. The results demonstrate robust, real-time performance, proving that a cost- effective robot can safely navigate complex environments while respecting critical visual cues invisible to traditional planners. |
| 2025-11-10 | [SAFENLIDB: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces](http://arxiv.org/abs/2511.06778v1) | Ruiheng Liu, XiaoBing Chen et al. | The rapid advancement of Large Language Models (LLMs) has driven significant progress in Natural Language Interface to Database (NLIDB). However, the widespread adoption of LLMs has raised critical privacy and security concerns. During interactions, LLMs may unintentionally expose confidential database contents or be manipulated by attackers to exfiltrate data through seemingly benign queries. While current efforts typically rely on rule-based heuristics or LLM agents to mitigate this leakage risk, these methods still struggle with complex inference-based attacks, suffer from high false positive rates, and often compromise the reliability of SQL queries. To address these challenges, we propose \textsc{SafeNlidb}, a novel privacy-security alignment framework for LLM-based NLIDB. The framework features an automated pipeline that generates hybrid chain-of-thought interaction data from scratch, seamlessly combining implicit security reasoning with SQL generation. Additionally, we introduce reasoning warm-up and alternating preference optimization to overcome the multi-preference oscillations of Direct Preference Optimization (DPO), enabling LLMs to produce security-aware SQL through fine-grained reasoning without the need for human-annotated preference data. Extensive experiments demonstrate that our method outperforms both larger-scale LLMs and ideal-setting baselines, achieving significant security improvements while preserving high utility.WARNING: This work may contain content that is offensive and harmful! |
| 2025-11-10 | [Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View](http://arxiv.org/abs/2511.06722v1) | Jianyu Qi, Ding Zou et al. | Recent advances in Multimodal Large Language Models (MLLMs) have spurred significant progress in Chain-of-Thought (CoT) reasoning. Building on the success of Deepseek-R1, researchers extended multimodal reasoning to post-training paradigms based on reinforcement learning (RL), focusing predominantly on mathematical datasets. However, existing post-training paradigms tend to neglect two critical aspects: (1) The lack of quantifiable difficulty metrics capable of strategically screening samples for post-training optimization. (2) Suboptimal post-training paradigms that fail to jointly optimize perception and reasoning capabilities. To address this gap, we propose two novel difficulty-aware sampling strategies: Progressive Image Semantic Masking (PISM) quantifies sample hardness through systematic image degradation, while Cross-Modality Attention Balance (CMAB) assesses cross-modal interaction complexity via attention distribution analysis. Leveraging these metrics, we design a hierarchical training framework that incorporates both GRPO-only and SFT+GRPO hybrid training paradigms, and evaluate them across six benchmark datasets. Experiments demonstrate consistent superiority of GRPO applied to difficulty-stratified samples compared to conventional SFT+GRPO pipelines, indicating that strategic data sampling can obviate the need for supervised fine-tuning while improving model accuracy. Our code will be released at https://github.com/qijianyu277/DifficultySampling. |

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



