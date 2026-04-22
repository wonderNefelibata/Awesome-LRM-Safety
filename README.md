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
| 2026-04-21 | [Safe Continual Reinforcement Learning in Non-stationary Environments](http://arxiv.org/abs/2604.19737v1) | Austin Coursey, Abel Diaz-Gonzalez et al. | Reinforcement learning (RL) offers a compelling data-driven paradigm for synthesizing controllers for complex systems when accurate physical models are unavailable; however, most existing control-oriented RL methods assume stationarity and, therefore, struggle in real-world non-stationary deployments where system dynamics and operating conditions can change unexpectedly. Moreover, RL controllers acting in physical environments must satisfy safety constraints throughout their learning and execution phases, rendering transient violations during adaptation unacceptable. Although continual RL and safe RL have each addressed non-stationarity and safety, respectively, their intersection remains comparatively unexplored, motivating the study of safe continual RL algorithms that can adapt over the system's lifetime while preserving safety. In this work, we systematically investigate safe continual reinforcement learning by introducing three benchmark environments that capture safety-critical continual adaptation and by evaluating representative approaches from safe RL, continual RL, and their combinations. Our empirical results reveal a fundamental tension between maintaining safety constraints and preventing catastrophic forgetting under non-stationary dynamics, with existing methods generally failing to achieve both objectives simultaneously. To address this shortcoming, we examine regularization-based strategies that partially mitigate this trade-off and characterize their benefits and limitations. Finally, we outline key open challenges and research directions toward developing safe, resilient learning-based controllers capable of sustained autonomous operation in changing environments. |
| 2026-04-21 | [HardNet++: Nonlinear Constraint Enforcement in Neural Networks](http://arxiv.org/abs/2604.19669v1) | Andrea Goertzen, Kaveh Alim et al. | Enforcing constraint satisfaction in neural network outputs is critical for safety, reliability, and physical fidelity in many control and decision-making applications. While soft-constrained methods penalize constraint violations during training, they do not guarantee constraint adherence during inference. Other approaches guarantee constraint satisfaction via specific parameterizations or a projection layer, but are tailored to specific forms (e.g., linear constraints), limiting their utility in other general problem settings. Many real-world problems of interest are nonlinear, motivating the development of methods that can enforce general nonlinear constraints. To this end, we introduce HardNet++, a constraint-enforcement method that simultaneously satisfies linear and nonlinear equality and inequality constraints. Our approach iteratively adjusts the network output via damped local linearizations. Each iteration is differentiable, admitting an end-to-end training framework, where the constraint satisfaction layer is active during training. We show that under certain regularity conditions, this procedure can enforce nonlinear constraint satisfaction to arbitrary tolerance. Finally, we demonstrate tight constraint adherence without loss of optimality in a learning-for-optimization context, where we apply this method to a model predictive control problem with nonlinear state constraints. |
| 2026-04-21 | [Multiscale Assessment of Tritium Behavior in Preliminary Fusion Pilot Plant Design Using Surrogate Models in TMAP8](http://arxiv.org/abs/2604.19647v1) | Lin Yang, Pierre-Clément A. Simon et al. | The complexity and significance of multiscale phenomena in fusion energy systems make advanced modeling necessary for designing, optimizing, and safely deploying fusion plants. Tritium accountancy is one of those challenges for deuterium-tritium fusion systems. Its availability is constrained by its short half-life (12.33 years) and limited natural abundance, which require fusion plants to breed tritium onsite. Therefore, accurate tritium accountancy is essential for effective resource management, safety, and economics in fusion plants. Through the U.S. Department of Energy milestone program, Tokamak Energy Ltd. is developing a fusion pilot plant design and evaluating tritium retention and loss in key components and their effect on the fuel cycle. To rapidly explore design trade-offs and quantify design decisions on tritium management, this study presents a multiscale analysis to investigate tritium diffusion, trapping, and recovery in key plasma-facing components. To enhance computational efficiency, we integrate surrogate models at the component-level within a fuel cycle model at the system-level, enabling rapid evaluation of tritium recycling dynamics and inventory under various operational scenarios. The goal of this study is twofold: (1) demonstrate the feasibility of utilizing surrogate models to increase the accuracy of fuel cycle modeling, and (2) rapidly evaluate the performance of fusion technologies to accelerate design iterations. This multiscale model provides the tritium transport and retention behavior and supports the plasma-facing components design optimization in normal and bake-out operations. The work is implemented using the Tritium Migration Analysis Program, Version 8 (TMAP8), an open-source application for tritium transport analysis in fusion systems. |
| 2026-04-21 | [Safety-Critical Contextual Control via Online Riemannian Optimization with World Models](http://arxiv.org/abs/2604.19639v1) | Tongxin Li | Modern world models are becoming too complex to admit explicit dynamical descriptions. We study safety-critical contextual control, where a Planner must optimize a task objective using only feasibility samples from a black-box Simulator, conditioned on a context signal $ξ_t$. We develop a sample-based Penalized Predictive Control (PPC) framework grounded in online Riemannian optimization, in which the Simulator compresses the feasibility manifold into a score-based density $\hat{p}(u \mid ξ_t)$ that endows the action space with a Riemannian geometry guiding the Planner's gradient descent. The barrier curvature $κ(ξ_t)$, the minimum curvature of the conditional log-density $-\ln\hat{p}(\cdot\midξ_t)$, governs both convergence rate and safety margin, replacing the Lipschitz constant of the unknown dynamics. Our main result is a contextual safety bound showing that the distance from the true feasibility manifold is controlled by the score estimation error and a ratio that depends on $κ(ξ_t)$, both of which improve with richer context. Simulations on a dynamic navigation task confirm that contextual PPC substantially outperforms marginal and frozen density models, with the advantage growing after environment shifts. |
| 2026-04-21 | [SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models](http://arxiv.org/abs/2604.19638v1) | Josue Torres-Fonseca, Naihao Deng et al. | Multimodal Large Language Models are increasingly adopted as autonomous agents in interactive environments, yet their ability to proactively address safety hazards remains insufficient. We introduce SafetyALFRED, built upon the embodied agent benchmark ALFRED, augmented with six categories of real-world kitchen hazards. While existing safety evaluations focus on hazard recognition through disembodied question answering (QA) settings, we evaluate eleven state-of-the-art models from the Qwen, Gemma, and Gemini families on not only hazard recognition, but also active risk mitigation through embodied planning. Our experimental results reveal a significant alignment gap: while models can accurately recognize hazards in QA settings, average mitigation success rates for these hazards are low in comparison. Our findings demonstrate that static evaluations through QA are insufficient for physical safety, thus we advocate for a paradigm shift toward benchmarks that prioritize corrective actions in embodied contexts. We open-source our code and dataset under https://github.com/sled-group/SafetyALFRED.git |
| 2026-04-21 | [Adding Compilation Metadata To Binaries To Make Disassembly Decidable](http://arxiv.org/abs/2604.19628v1) | Daniel Engel, Freek Verbeek et al. | The binary executable format is the standard method for distributing and executing software. Yet, it is also as opaque a representation of software as can be. If the binary format were augmented with metadata that provides security-relevant information, such as which data is intended by the compiler to be executable instructions, or how memory regions are expected to be bounded, that would dramatically improve the safety and maintainability of software. In this paper, we propose a binary format that is a middle ground between a stripped black-box binary and open source. We provide a tool that generates metadata capturing the compiler's intent and inserts it into the binary. This metadata enables lifting to a correct and recompilable higher-level representation and makes analysis and instrumentation more reliable. Our evaluation shows that adding metadata does not affect runtime behavior or performance. Compared to DWARF, our metadata is roughly 17% of its size. We validate correctness by compiling a comprehensive set of real-world C and C++ binaries and demonstrating that they can be lifted, instrumented, and recompiled without altering their behavior. |
| 2026-04-21 | [Goal-Oriented Semantic Communication for Logical Decision Making](http://arxiv.org/abs/2604.19614v1) | Ahmet Faruk Saz, Faramarz Fekri | This paper develops a principled foundation for goal-oriented semantic communication for logical decision-making. Consider a setting where autonomous agents engage in collaborative perception. In such settings, the volume of sensory data and limited bandwidth often make transmission of raw observations infeasible, requiring intelligent selection of task-relevant information. Because these scenarios are safety-critical, the selection and decision processes must also be transparent and verifiable. To address this, we propose an explainable semantic communication framework grounded in a First-Order Logic (FOL) hierarchical representation of the world. We define semantic information, entropy, conditional entropy, and mutual information by assigning an inductive logical probability measure over semantic structures in the language. Based on these definitions, we formulate a goal-oriented semantic communication objective through semantic rate-distortion theory and, equivalently, through the semantic information bottleneck principle. In this framework, task rules are represented as goal-oriented states, defined as a layer over the world states to capture decision-relevant abstractions. The resulting principle selects evidence that is most informative about these states, aiming to transmit only those FOL clauses most critical for decision-making while preserving logical verifiability. We demonstrate the effectiveness of the approach in a deduction-based safe path-following task within an FOL-based urban environment simulator with multiple dynamic agents. |
| 2026-04-21 | [Cross-Model Consistency of AI-Generated Exercise Prescriptions: A Repeated Generation Study Across Three Large Language Models](http://arxiv.org/abs/2604.19598v1) | Kihyuk Lee | This study compared repeated generation consistency of exercise prescription outputs across three large language models (LLMs), specifically GPT-4.1, Claude Sonnet 4.6, and Gemini 2.5 Flash, under temperature=0 conditions. Each model generated prescriptions for six clinical scenarios 20 times, yielding 360 total outputs analyzed across four dimensions: semantic similarity, output reproducibility, FITT classification, and safety expression. Mean semantic similarity was highest for GPT-4.1 (0.955), followed by Gemini 2.5 Flash (0.950) and Claude Sonnet 4.6 (0.903), with significant inter-model differences confirmed (H = 458.41, p < .001). Critically, these scores reflected fundamentally different generative behaviors: GPT-4.1 produced entirely unique outputs (100%) with stable semantic content, while Gemini 2.5 Flash showed pronounced output repetition (27.5% unique outputs), indicating that its high similarity score derived from text duplication rather than consistent reasoning. Identical decoding settings thus yielded fundamentally different consistency profiles, a distinction that single-output evaluations cannot capture. Safety expression reached ceiling levels across all models, confirming its limited utility as a differentiating metric. These results indicate that model selection constitutes a clinical rather than merely technical decision, and that output behavior under repeated generation conditions should be treated as a core criterion for reliable deployment of LLM-based exercise prescription systems. |
| 2026-04-21 | [Enhancing Construction Worker Safety in Extreme Heat: A Machine Learning Approach Utilizing Wearable Technology for Predictive Health Analytics](http://arxiv.org/abs/2604.19559v1) | Syed Sajid Ullah, Amir Khan | Construction workers are highly vulnerable to heat stress, yet tools that translate real-time physiological data into actionable safety intelligence remain scarce. This study addresses this gap by developing and evaluating deep learning models, specifically a baseline Long Short-Term Memory (LSTM) network and an attention-based LSTM, to predict heat stress among 19 workers in Saudi Arabia. Using Garmin Vivosmart 5 smartwatches to monitor metrics such as heart rate, HRV, and oxygen saturation, the attention-based model outperformed the baseline, achieving 95.40% testing accuracy and significantly reducing false positives and negatives. With precision, recall, and F1 scores of 0.982, this approach not only improves predictive performance but also offers interpretable results suitable for integration into IoT-enabled safety systems and BIM dashboards, advancing proactive, informatics-driven safety management in the construction industry. |
| 2026-04-21 | [Integrating Anomaly Detection into Agentic AI for Proactive Risk Management in Human Activity](http://arxiv.org/abs/2604.19538v1) | Farbod Zorriassatine, Ahmad Lotfi | Agentic AI, with goal-directed, proactive, and autonomous decision-making capabilities, offers a compelling opportunity to address movement-related risks in human activity, including the persistent hazard of falls among elderly populations. Despite numerous approaches to fall mitigation through fall prediction and detection, existing systems have not yet functioned as universal solutions across care pathways and safety-critical environments. This is largely due to limitations in consistently handling real-world complexity, particularly poor context awareness, high false alarm rates, environmental noise, and data scarcity. We argue that fall detection and fall prediction can usefully be formulated as anomaly detection problems and more effectively addressed through an agentic AI system. More broadly, this perspective enables the early identification of subtle deviations in movement patterns associated with increased risk, whether arising from age-related decline, fatigue, or environmental factors. While technical requirements for immediate deployment are beyond the scope of this paper, we propose a conceptual framework that highlights potential value. This framework promotes a well-orchestrated approach to risk management by dynamically selecting relevant tools and integrating them into adaptive decision-making workflows, rather than relying on static configurations tailored to narrowly defined scenarios. |
| 2026-04-21 | [GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance for Bimanual Mobile Manipulation](http://arxiv.org/abs/2604.19522v1) | Marcelino Julio Fernando, Miguel Altamirano Cabrera et al. | Bimanual mobile manipulation requires a seamless integration between high-level semantic reasoning and safe, compliant physical interaction - a challenge that end-to-end models approach opaquely and classical controllers lack the context to address. This paper presents GenerativeMPC, a hierarchical cyber-physical framework that explicitly bridges semantic scene understanding with physical control parameters for bimanual mobile manipulators. The system utilizes a Vision-Language Model with Retrieval-Augmented Generation (VLM-RAG) to translate visual and linguistic context into grounded control constraints, specifically outputting dynamic velocity limits and safety margins for a Whole-Body Model Predictive Controller (MPC). Simultaneously, the VLM-RAG module modulates virtual stiffness and damping gains for a unified impedance-admittance controller, enabling context-aware compliance during human-robot interaction. Our framework leverages an experience-driven vector database to ensure consistent parameter grounding without retraining. Experimental results in MuJoCo, IsaacSim, and on a physical bimanual platform confirm a 60% speed reduction near humans and safe, socially-aware navigation and manipulation through semantic-to-physical parameter grounding. This work advances the field of human-centric cybernetics by grounding large-scale cognitive models into predictable, high-frequency physical control loops. |
| 2026-04-21 | [Operando Characterization of Volume Changes in Lithium-Ion Battery Electrodes during Cycling using Isotope Multilayers](http://arxiv.org/abs/2604.19519v1) | Erwin Hueger, Daniel Uxa et al. | This study reports on advancements in operando characterization of volume changes in lithium-ion battery (LIB) electrode materials during electrochemical cycling. Volume changes are crucial for LIB operation because they are related to the amount of stored energy as well as LIB integrity, performance, and safety. The study introduces a method based on isotope multilayers as active material to track the intrinsic modification of electrode volume in real time under operating conditions with operando neutron reflectometry. A natGe-73Ge multilayer film is used as a model system to measure the volume change of amorphous germanium electrodes during charging and discharging. Isotope modulation produces a Bragg peak in the neutron reflectivity pattern, sensitive only to the modification of volume within the active material of the electrode. Battery side reactions, such as the growth and reduction of the solid-electrolyte interphase is excluded. Using this method, the volume modification as a function of Li content x in LixGe can easily be derived from the scattering vector position of the Bragg peak without fitting numerous complex reflectivity patterns. The experiments show a reversible volume change of amorphous germanium of up to 250 percent for x = 3, which appears to be largely independent of current density, cycle number, and the thickness of the individual Ge layers. Also, there are tentative indications that the crystallization and re-amorphization of LixGe are not influencing the volume change. |
| 2026-04-21 | [Assessing VLM-Driven Semantic-Affordance Inference for Non-Humanoid Robot Morphologies](http://arxiv.org/abs/2604.19509v1) | Jess Jones, Raul Santos-Rodriguez et al. | Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding human-object interactions, but their application to robotic systems with non-humanoid morphologies remains largely unexplored. This work investigates whether VLMs can effectively infer affordances for robots with fundamentally different embodiments than humans, addressing a critical gap in the deployment of these models for diverse robotic applications. We introduce a novel hybrid dataset that combines annotated real-world robotic affordance-object relations with VLM-generated synthetic scenarios, and perform an empirical analysis of VLM performance across multiple object categories and robot morphologies, revealing significant variations in affordance inference capabilities. Our experiments demonstrate that while VLMs show promising generalisation to non-humanoid robot forms, their performance is notably inconsistent across different object domains. Critically, we identify a consistent pattern of low false positive rates but high false negative rates across all morphologies and object categories, indicating that VLMs tend toward conservative affordance predictions. Our analysis reveals that this pattern is particularly pronounced for novel tool use scenarios and unconventional object manipulations, suggesting that effective integration of VLMs in robotic systems requires complementary approaches to mitigate over-conservative behaviour while preserving the inherent safety benefits of low false positive rates. |
| 2026-04-21 | [Involuntary In-Context Learning: Exploiting Few-Shot Pattern Completion to Bypass Safety Alignment in GPT-5.4](http://arxiv.org/abs/2604.19461v1) | Alex Polyakov, Daniel Kuznetsov | Safety alignment in large language models relies on behavioral training that can be overridden when sufficiently strong in-context patterns compete with learned refusal behaviors. We introduce Involuntary In-Context Learning (IICL), an attack class that uses abstract operator framing with few-shot examples to force pattern completion that overrides safety training. Through 3479 probes across 10 OpenAI models, we identify the attack's effective components through a seven-experiment ablation study. Key findings: (1)~semantic operator naming achieves 100\,\% bypass rate (50/50, $p < 0.001$); (2)~the attack requires abstract framing, since identical examples in direct question-and-answer format yield 0\,\%; (3)~example ordering matters strongly (interleaved: 76\,\%, harmful-first: 6\,\%); (4)~temperature has no meaningful effect (46--56\,\% across 0.0--1.0). On the HarmBench benchmark, IICL achieves 24.0\,\% bypass $[18.6\%, 30.4\%]$ against GPT-5.4 with detailed 619-word responses, compared to 0.0\,\% for direct queries. |
| 2026-04-21 | [Do LLMs Game Formalization? Evaluating Faithfulness in Logical Reasoning](http://arxiv.org/abs/2604.19459v1) | Kyuhee Kim, Auguste Poiroux et al. | Formal verification guarantees proof validity but not formalization faithfulness. For natural-language logical reasoning, where models construct axiom systems from scratch without library constraints, this gap between valid proofs and faithful translations is especially acute. We investigate whether frontier models exploit this gap when generating Lean 4 proofs, a behavior we term formalization gaming.   We evaluate GPT-5 and DeepSeek-R1 on 303 first-order logic problems (203 from FOLIO, 100 from Multi-LogiEval), comparing unified generation against a two-stage pipeline that separates formalization from proving. Despite compilation rates of 87-99%, we find no evidence of systematic gaming in unified generation: models prefer reporting failure over forcing proofs, even under prompting designed to encourage it. However, unfaithfulness that evades our detection signals may still occur. The two-stage pipeline reveals two distinct modes of unfaithfulness: GPT-5 fabricates axioms during proof generation, a reactive fallback detectable via cross-stage comparison, while DeepSeek-R1 mistranslates premises during formalization, producing internally consistent outputs that evade detection entirely. These findings show that high compilation rates or accuracies should not be equated with faithful reasoning. Code and data are available at https://github.com/koreankiwi99/formalization-gaming. |
| 2026-04-21 | [Mind2Drive: Predicting Driver Intentions from EEG in Real-world On-Road Driving](http://arxiv.org/abs/2604.19368v1) | Ghadah Alosaimi, Hanadi Alhamdan et al. | Predicting driver intention from neurophysiological signals offers a promising pathway for enhancing proactive safety in advanced driver assistance systems, yet remains challenging in real-world driving due to EEG signal non-stationarity and the complexity of cognitive-motor preparation. This study proposes and evaluates an EEG-based driver intention prediction framework using a synchronised multi-sensor platform integrated into a real electric vehicle. A real-world on-road dataset was collected across 32 driving sessions, and 12 deep learning architectures were evaluated under consistent experimental conditions. Among the evaluated architectures, TSCeption achieved the highest average accuracy (0.907) and Macro-F1 score (0.901). The proposed framework demonstrates strong temporal stability, maintaining robust decoding performance up to 1000 ms before manoeuvre execution with minimal degradation. Furthermore, additional analyses reveal that minimal EEG preprocessing outperforms artefact-handling pipelines, and prediction performance peaks within a 400-600 ms interval, corresponding to a critical neural preparatory phase preceding driving manoeuvres. Overall, these findings support the feasibility of early and stable EEG-based driver intention decoding under real-world on-road conditions. Code: https://github.com/galosaimi/Mind2Drive. |
| 2026-04-21 | [Are Large Language Models Economically Viable for Industry Deployment?](http://arxiv.org/abs/2604.19342v1) | Abdullah Mohammad, Sushant Kumar Ray et al. | Generative AI-powered by Large Language Models (LLMs)-is increasingly deployed in industry across healthcare decision support, financial analytics, enterprise retrieval, and conversational automation, where reliability, efficiency, and cost control are critical. In such settings, models must satisfy strict constraints on energy, latency, and hardware utilization-not accuracy alone. Yet prevailing evaluation pipelines remain accuracy-centric, creating a Deployment-Evaluation Gap-the absence of operational and economic criteria in model assessment. To address this gap, we present EDGE-EVAL-a industry-oriented benchmarking framework that evaluates LLMs across their full lifecycle on legacy NVIDIA Tesla T4 GPUs. Benchmarking LLaMA and Qwen variants across three industrial tasks, we introduce five deployment metrics-Economic Break-Even (Nbreak), Intelligence-Per-Watt (IPW ), System Density (\r{ho}sys), Cold-Start Tax (Ctax), and Quantization Fidelity (Qret)-capturing profitability, energy efficiency, hardware scaling, serverless feasibility, and compression safety. Our results reveal a clear efficiency frontier-models in the <2B parameter class dominate larger baselines across economic and ecological dimensions. LLaMA-3.2-1B (INT4) achieves ROI break-even in 14 requests (median), delivers 3x higher energy-normalized intelligence than 7B models, and exceeds 6,900 tokens/s/GB under 4-bit quantization. We further uncover an efficiency anomaly-while QLoRA reduces memory footprint, it increases adaptation energy by up to 7x for small models-challenging prevailing assumptions about quantization-aware training in edge deployment. |
| 2026-04-21 | [Beyond Semantic Similarity: A Component-Wise Evaluation Framework for Medical Question Answering Systems with Health Equity Implications](http://arxiv.org/abs/2604.19281v1) | Abu Noman Md Sakib, Md. Main Oddin Chisty et al. | The use of Large Language Models (LLMs) to support patients in addressing medical questions is becoming increasingly prevalent. However, most of the measures currently used to evaluate the performance of these models in this context only measure how closely a model's answers match semantically, and therefore do not provide a true indication of the model's medical accuracy or of the health equity risks associated with it. To address these shortcomings, we present a new evaluation framework for medical question answering called VB-Score (Verification-Based Score) that provides a separate evaluation of the four components of entity recognition, semantic similarity, factual consistency, and structured information completeness for medical question-answering models. We perform rigorous reviews of the performance of three well-known and widely used LLMs on 48 public health-related topics taken from high-quality, authoritative information sources. Based on our analyses, we discover a major discrepancy between the models' semantic and entity accuracy. Our assessments of the performance of all three models show that each of them has almost uniformly severe performance failures when evaluated against our criteria. Our findings indicate alarming performance disparities across various public health topics, with most of the models exhibiting 13.8% lower performance (compared to an overall average) for all the public health topics that relate to chronic conditions that occur in older and minority populations, which indicates the existence of what's known as condition-based algorithmic discrimination. Our findings also demonstrate that prompt engineering alone does not compensate for basic architectural limitations on how these models perform in extracting medical entities and raise the question of whether semantic evaluation alone is a sufficient measure of medical AI safety. |
| 2026-04-21 | [HarDBench: A Benchmark for Draft-Based Co-Authoring Jailbreak Attacks for Safe Human-LLM Collaborative Writing](http://arxiv.org/abs/2604.19274v1) | Euntae Kim, Soomin Han et al. | Large language models (LLMs) are increasingly used as co-authors in collaborative writing, where users begin with rough drafts and rely on LLMs to complete, revise, and refine their content. However, this capability poses a serious safety risk: malicious users could jailbreak the models-filling incomplete drafts with dangerous content-to force them into generating harmful outputs. In this paper, we identify the vulnerability of current LLMs to such draft-based co-authoring jailbreak attacks and introduce HarDBench, a systematic benchmark designed to evaluate the robustness of LLMs against this emerging threat. HarDBench spans a range of high-risk domains-including Explosives, Drugs, Weapons, and Cyberattacks-and features prompts with realistic structure and domain-specific cues to assess the model susceptibility to harmful completions. To mitigate this risk, we introduce a safety-utility balanced alignment approach based on preference optimization, training models to refuse harmful completions while remaining helpful on benign drafts. Experimental results show that existing LLMs are highly vulnerable in co-authoring contexts and our alignment method significantly reduces harmful outputs without degrading performance on co-authoring capabilities. This presents a new paradigm for evaluating and aligning LLMs in human-LLM collaborative writing settings. Our new benchmark and dataset are available on our project page at https://github.com/untae0122/HarDBench |
| 2026-04-21 | [When Can We Trust Deep Neural Networks? Towards Reliable Industrial Deployment with an Interpretability Guide](http://arxiv.org/abs/2604.19206v1) | Hang-Cheng Dong, Yuhao Jiang et al. | The deployment of AI systems in safety-critical domains, such as industrial defect inspection, autonomous driving, and medical diagnosis, is severely hampered by their lack of reliability. A single undetected erroneous prediction can lead to catastrophic outcomes. Unfortunately, there is often no alternative but to place trust in the outputs of a trained AI system, which operates without an internal safeguard to flag unreliable predictions, even in cases of high accuracy. We propose a post-hoc explanation-based indicator to detect false negatives in binary defect detection networks. To our knowledge, this is the first method to proactively identify potentially erroneous network outputs. Our core idea leverages the difference between class-specific discriminative heatmaps and class-agnostic ones. We compute the difference in their intersection over union (IoU) as a reliability score. An adversarial enhancement method is further introduced to amplify this disparity. Evaluations on two industrial defect detection benchmarks show our method effectively identifies false negatives. With adversarial enhancement, it achieves 100\% recall, albeit with a trade-off for true negatives. Our work thus advocates for a new and trustworthy deployment paradigm: data-model-explanation-output, moving beyond conventional end-to-end systems to provide critical support for reliable AI in real-world applications. |

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



