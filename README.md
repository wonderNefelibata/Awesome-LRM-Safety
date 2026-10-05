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
| 2026-10-02 | [Risk-Aware Input-Constrained Safe Intercept Guidance Against Multiple Moving Defenders](http://arxiv.org/abs/2610.03681v1) | Praveen Kumar Ranjan, Abhinav Sinha et al. | This paper develops a risk-aware guidance law for an attacker to intercept a stationary target in the presence of multiple moving defenders while respecting the control input limits. We represent defender threats via engagement zones (EZs) (regions of state-space from which attacker interception is feasible) and employ the dynamic maneuvering cue (DMC) (minimum heading correction to exit EZs) as a deterministic measure of geometric risk. Safety is enforced through a prescribed bound on the DMC to enable controlled EZ penetration subject to a specified risk tolerance. For multiple defenders, the corresponding safety constraints are combined through a smooth-minimum aggregation. The proposed guidance strategy drives the attacker toward the target when the engagement remains safe and transitions to an avoidance maneuver as the prescribed safety boundary is approached. The resulting online optimization-free guidance law directly bounds the applied lateral acceleration and initiates a bounded preemptive maneuver before corrective action is required. Closed-loop analysis establishes preservation of the prescribed risk constraints under pointwise bounded-input feasibility and finite-time target interception under a sustained target-accessibility condition. Numerical studies validate the proposed strategy and illustrate the tradeoff between allowable geometric risk and interception performance. |
| 2026-10-02 | [CORNAV: Construction-Aware Reasoning for Robot Navigation on Active Worksites](http://arxiv.org/abs/2610.03622v1) | Parastoo Ali Pour, Deepak Prakash Kumar et al. | The construction industry faces persistent labor shortages, low productivity that costs the global economy over $1.6 trillion annually, and one of the highest injury rates among major industries. These factors motivate the use of autonomous robots to improve efficiency and worker safety. Existing language-grounded navigation systems, however, rely on semantic scene understanding alone and lack access to construction-specific context such as architectural plans, evolving work schedules, and safety constraints. As a result, they localize permanent building features unreliably and cannot safely navigate active jobsites. We present CORNAV, a blueprint-grounded, schedule-aware navigation framework that operates from 2D CAD drawings and project schedules without requiring a Building Information Model. CORNAV aligns architectural blueprints against hierarchical open-vocabulary 3D scene graphs to ground object queries, converts project schedules into time-varying navigation constraints, and validates requests through an LLM-based safety module that escalates hazardous zones before planning. An A* planner then enforces mandatory exclusion zones while preferentially avoiding higher-risk areas. Across an indoor office and a real construction site, blueprint grounding raises task success from 13.0% to 72.2% over semantic retrieval alone, schedule awareness eliminates all hard-zone violations, and the safety module correctly rejects hazardous requests arising from mislabeled project schedules. |
| 2026-10-02 | [Writerslogic at PAN 2026: Process over Content for Robust Detection under Domain Shift](http://arxiv.org/abs/2610.03565v1) | David L. Condrey | We describe the Writerslogic systems for three PAN at CLEF 2026 shared tasks (Reasoning Trajectory Detection, Voight-Kampff Generative AI Detection, and Multi-Author Writing Style Analysis), unified by a shared analytical framework: feature robustness under distribution shift is governed by support overlap between training and test distributions, not by training-set effect size. This yields a taxonomy (domain-anchored, domain-portable, domain-invariant) that explains why generator-specific features die under domain shift while vocabulary fingerprints (hapax ratio, Yule's K, Heaps' exponent), compression measures, and character n-grams survive. On Reasoning Trajectory Detection, where training was entirely mathematics and 84 percent of test was unseen domains, the framework guided system design to 1st place in source detection (0.85 macro F1 via Opus-Sonnet agreement) and 3rd place in safety classification (0.66 macro F1 via query-refusal decomposition). For Voight-Kampff, we built a calibrated ensemble of DeBERTa-v2 (ONNX), multi-seed LightGBM with 44 domain-portable stylometric features, and SVM on n-gram TF-IDF, combined via learned stacking with isotonic calibration; the best configuration achieved 0.891 on the PAN 2026 test set with balanced sub-metrics (0.853 to 0.902 across all evaluation dimensions). For Multi-Author Writing Style Analysis, we describe a system fusing spectral clustering over character n-gram similarity graphs, normalized compression distance for local boundary detection, and SmolLM-135M perplexity for neural change-point detection; a platform mix-up meant our run never reached the official evaluation, so we report the design and its a priori predictions. Across all three tasks, features measuring generation process properties are designed to outperform features measuring generated content properties under domain shift. |
| 2026-10-02 | [CorrectGuard: Eyes-Off Correctness Estimation for Black-Box Security Guardrails](http://arxiv.org/abs/2610.03470v1) | Adam Faulkner, Nil-Jana Akpinar et al. | AI services increasingly rely on black-box security guardrails, yet privacy-preserving model auditing regimes often cannot measure how well these systems perform in both a human eyes-off production setting, which disallows human inspection of user input, and a machine eyes-off setting, which disallows model inspection of such input. We introduce CorrectGuard, an eyes-off correctness estimation framework for both settings, which involves an independent model-based evaluator predicting whether guardrail decisions on human- and machine-inaccessible inputs are correct using only labeled eyes-on data and without access to the guardrail's internals. We evaluate in-context learning, embedding, and finetuning-based correctness models under leave-one-dataset-out evaluation across 13 safety and security datasets spanning harmful content, jailbreaks, prompt injection, and extraction, and across open-weight guardrails treated uniformly as black boxes. Across both human and machine eyes-off settings (the latter implemented using privacy-preserving fingerprinting of inputs), in-context-learning-based correctness classifiers substantially improve error identification across guardrails, achieving up to a 25 percentage-point increase in macro accuracy, as do finetuning-based approaches which provide a nearly 15-point boost, although performance varies sharply across guardrails and held-out datasets. Correctness scores also support guardrail decision ranking and abstention: across 3 guardrails, the best correctness rankings reduce AURC from unranked baselines of 0.33-0.44 to 0.17-0.22, while the best operating points retain 37.5-52.0% of guardrail decisions at 15% observed risk. These results show that external correctness models can expose systematic failures and support guardrail decision abstention without privileged access to the guardrail. |
| 2026-10-02 | [Forward Bayesian Inference for the Binary Black Hole Populations](http://arxiv.org/abs/2610.03459v1) | Poulami Dutta Roy, Tanmaya Mishra et al. | We present PopCWB, a Bayesian framework for the inference of the binary black hole (BBH) population based on the unmodeled search pipeline coherent WaveBurst (cWB). The standard population analysis takes an inverse approach by inferring the underlying population model parameters for a given set of detected events using the individual posterior samples. In PopCWB, we take a forward approach to population inference by varying population models to identify the most optimal one which describes the observed events. Rather than using event-level posterior samples, PopCWB uses a large set of simulated events reconstructed by cWB and their total mass, estimated with a machine-learning regression, to construct marginal probability distributions. These marginal probabilities are then used in the construction of the likelihood. We apply PopCWB to BBH events detected by cWB during the third observing run (O3) of the LIGO--Virgo--KAGRA detectors. We constrain an astrophysically motivated BBH population model that incorporates the effects of pulsational pair-instability supernovae and dynamical mergers. The analysis predicts a maximum black hole mass of $\mathbf{45.5^{+5.3}_{-6.5} M_{\odot}}$ from stellar evolution and an inferred local BBH merger rate of $\mathbf{23.1^{+11.3}_{-8.4} \, Gpc^{-3} yr^{-1}}$. We compare these results with the existing population models from the literature. |
| 2026-10-02 | [Defense-in-Depth at the Perception-Reasoning Interface of LLM-Centric Agentic UAV Swarms](http://arxiv.org/abs/2610.03319v1) | Mohammadhossein Homaei, Yousef Emami et al. | Large Language Models (LLMs) increasingly support Uncrewed Aerial Vehicle (UAV) swarm operations such as data collection scheduling, where the model reads structured sensor reports and decides which sensors to visit. An adversary who quietly manipulates those reports can redirect the swarm without modifying the model weights or the UAV. Defenses for this interface have been proposed architecturally but rarely implemented or evaluated. We implement and evaluate defense-in-depth at the perception-reasoning interface of LLM-Centric Agentic UAV Swarms. Five layers check the provenance of a report, whether its values are physically admissible, whether they agree with what swarm geometry and service history predict, whether the resulting schedule starves any sensor, and, when these fail, hand control to a deterministic scheduler that ignores the suspect input. We test each layer against an adversary strong enough to defeat the layer before it. For each of the three input-side layers, we derive in closed form how far a report can be distorted before that layer reacts, fixing each boundary from deployment parameters before any attack data is collected; across thirty matched simulation runs, predicted and measured boundaries agree. Separating attack detection from response is a well-established principle, and we quantify the cost of neglecting this distinction at the perception-reasoning interface. When the system rejects a report, it replaces it with the most recent accepted report. This prevents the adversary from controlling the UAV schedule, but it also increases cumulative cost by 79% and 74% for the two detectors, respectively, compared with the undefended system. The safety check does not detect any attacks, but it nevertheless reduces the attack-induced cost by 37.5%. |
| 2026-10-02 | [Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case](http://arxiv.org/abs/2610.03190v1) | Tingzhu Bi, Ping Wang et al. | Accident, defect and outage investigations end with a decision that ordinary question answering never faces: whether the evidence gathered so far is enough to close the case. We study this decision for LLM investigators, which request evidence from a case file, revise their hypotheses, and either close the case with a conclusion grounded in what they read or leave it open and name what is missing. This judgment does not come with capability: an untrained 9B model overstates its evidence in 97% of its answers, and a frontier model that identifies the right cause in 84% of cases still overstates in 91% and closes 17 of the 41 cases whose official finding is "cause undetermined". Measuring it is also non-trivial: the source of a case largely predicts its label, and a rule that reads only the source reaches 83.0 balanced accuracy on our test cases. We therefore evaluate closure with three tests: closure accuracy, reported against this rule and within each source; evidence dependence, which removes the grounds of a conclusion and checks whether the model stops closing; and conclusion and gap quality, a judged checklist of what the model asserts and what it says is missing. We build Nautil, 731 audited cases from aviation, rail, maritime, chemical-safety and vehicle-defect reports and production server incidents, with teacher trajectories, an out-of-distribution test set and counterfactual evidence versions. Fine-tuning a 9B model on these trajectories makes its closures follow the evidence: removing the grounds lowers its closure rate by 26 points relative to a matched control, overstatement falls from 97% to 35%, and correct, non-overstated conclusions rise from 3% to 43%. Reinforcement learning that rewards only the closure decision then raises balanced accuracy from 69.2 to 83.3, on par with the teacher, and within-source accuracy from 60.4 to 74.1, at some cost in evidence dependence. |
| 2026-10-02 | [EvoRiskBench: An Evolving Benchmark for Runtime Security Risks in Workspace Agents](http://arxiv.org/abs/2610.03153v1) | Shiyi Kuang, Xuemei Luo et al. | Workspace agents combine large language models with execution harnesses to perform stateful, multi-step tasks that access or modify external resources. Existing benchmarks leave gaps in executable coverage of their runtime security risks, while evolving model capabilities, harnesses, tools, and threats motivate benchmark evolution. We introduce EvoRiskBench, an evolving benchmark organized around the EP-Path-EF framework, which links an initial risk entry point to a one-hop technical effect through an agent-mediated risk path. The framework defines nine entry-point categories and five effect categories; a 20-participant study supports their interpretability and classification consistency on representative cases. Guided by this framework, an automated end-to-end workflow constructs and executes risk cases in isolated environments and independently verifies outcomes using runtime traces and environment states. The benchmark provides a reproducible dataset of 450 adversarial tasks across six scenarios. We evaluate nine model-harness configurations spanning three models (GPT-5.6 Sol, DeepSeek-V4-Pro-0813, and Claude Opus 5) and three harnesses (Claude Code, Codex, and OpenClaw). Our results reveal substantial vulnerabilities across systems. The most vulnerable configuration, Codex with DeepSeek-V4-Pro-0813, reaches a 68.44% attack success rate (ASR), indicating that configuration of workspace agent is insufficient to ensure secure autonomous execution. ASR varies more across models than harnesses, and harness differences depend on the model. The benchmark cases and evaluation platform will be released after completion of artifact safety and reproducibility checks. |
| 2026-10-02 | [Safe Streaming Flow Planning by Aligning Sampling Dynamics with Execution Dynamics](http://arxiv.org/abs/2610.03132v1) | Seunghwan Jang, Jeongyong Yang et al. | Generative planners based on diffusion/flow matching can learn to synthesize long-horizon trajectories from demonstrations. However, real-world deployment requires (i) enforcing safety constraints during execution and (ii) tight online replanning at fast execution rates. Prior safe diffusion/flow planners generate the agent's full trajectory at once, while repeatedly perturbing intermediate states to satisfy safety constraints. This approach is not only computationally intensive, but also introduces distribution shift since the learned sampling dynamics is distinct from the system's execution dynamics. We propose SafeStreamingFlow, a goal-conditioned planner that aligns flow sampling dynamics with execution dynamics by sequentially integrating a learned state vector field with hierarchical state prediction. Importantly, we need to enforce safety constraints only for the executed step via high order control barrier functions. Across navigation, racing, and locomotion benchmarks, SafeStreamingFlow reduces planning latency and improves safety compared to existing methods, while maintaining competitive goal-reaching success. |
| 2026-10-02 | [Smart Sensing for Safer Bridges: From Sensor Signals to AI-Driven Anomaly Detection](http://arxiv.org/abs/2610.03082v1) | Rahul Jaiswal, Joakim Hellum et al. | Bridges contribute significantly to transportation connectivity and urban development. Therefore, reliable bridge monitoring is crucial for protecting public safety and detecting anomalous behavior in bridge sensor data that may provide early indications of abnormal structural conditions. This paper investigates anomaly detection in real-world bridge sensor data using two different complementary approaches, namely signal processing and the data-driven machine learning model Isolation Forest. The real-time bridge sensor data is collected from an iBridge sensor device installed on a bridge in Norway. The methods are evaluated using anomaly counts, anomaly detection time, processing rate, anomaly rates, visualization, and temporal agreement. Moreover, a controlled anomaly-injection analysis is performed to evaluate the sensitivity of each method. Numerical results demonstrate distinct detection characteristics and computational requirements, highlighting the potential of machine learning, particularly the data-driven Isolation Forest, alongside signal processing for identifying anomalies in bridge sensor measurements. |
| 2026-10-02 | [PEEK: Heterogeneous Parallelism for Privileged Error Detection in Safety-Critical Processors](http://arxiv.org/abs/2610.03045v1) | Tinglue Wang, Zhenghui Guo et al. | Heterogeneous parallel error detection architecture has been widely studied for safeguarding OoO superscalar processors in safetycritical systems, as it achieves significantly lower hardware overhead compared to traditional LockStep, by exploiting the parallelism that exists in a secondary execution. However, previous works do not cover the protection of privileged-mode execution, impeding their effectiveness in real-world deployment. Moreover, naive extension to privileged-mode can cause a litany of issues, from abysmal performance due to high synchronization costs, to full deadlocks. Here, we present PEEK, the first privileged parallel error detection architecture. Based on a deep analysis of privileged execution, we redesign the verification pipeline, addressing all the bottlenecks and bugs identified in privileged-mode protection. Evaluated using various metrics on an RTL-level full system running Linux, PEEK achieves full-privilege protection on Linux with negligible performance slowdown and affordable hardware overhead. PEEK has been taped out using a 28nm process, and its source is available at https://anonymous.4open.science/r/PEEK-3000. |
| 2026-10-02 | [Mobility Enhancement of Patients Body Monitoring based on WBAN with Multipath Routing](http://arxiv.org/abs/2610.03042v1) | Yasna Ghanbari Birgani, Nastooh Taheri Javan et al. | One of the promising applications of wireless sensor networks (WSNs) is monitoring of the human body for health concerns. For this purpose, a large number of small sensors are implanted in the human body. These sensors altogether provide a network of wireless sensors (WBANs) and monitor the vital signs and signals of the human body; these sensors will then send this information to the doctor. The most important application of the WBAN is the implementation of the monitoring network for patient safety in the hospital environment. In this case, supporting patients' mobility is one of the basic needs, which has been underestimated in recent studies. The problem that involves providing the required energy for the units used in this type of network is challenging; for this reason, sent/received units with very low power consumption and with a very small radius are used in order to save energy. The resulting small sending range leads to the lack of support for patients' mobility. In this paper, the ad hoc mode is suggested for use to establish a network and a multipath routing algorithm for the purpose of supporting patients' mobility in a hospital setting. The results of the simulation show that, in addition to supporting patients' mobility, the use of the proposed idea instead of previously presented protocols reduces delays in data transmission and energy consumption; it also increases the delivery rate depending on the destination and the lifetime of the network, while increasing routing overhead. |
| 2026-10-02 | [Tailoring the Quantization Space for 1-Bit KV Cache Compression](http://arxiv.org/abs/2610.03027v1) | Minsoo Cheong, Donghyun Son et al. | The key-value (KV) cache becomes a major memory bottleneck in long-context LLM inference, placing substantial pressure on memory capacity and bandwidth. To mitigate this bottleneck, vector quantization (VQ) has emerged as a promising approach for aggressive KV cache compression. However, existing VQ methods degrade substantially in the 1-bit regime. At such extreme compression, each codebook must represent a larger group of channels with a limited set of centroids, making effective use of its capacity increasingly challenging. To address this, we introduce $\textbf{TaSQ}$, which tailors the VQ target space by combining query-guided channel weighting, cross-head normalization, and covariance-aware channel grouping to better reflect the error sensitivity and statistical structure of cached activations. Since these transforms are RoPE-compatible and can be easily merged into projection weights and codebooks, TaSQ preserves the conventional VQ lookup structure and adds negligible serving overhead. Across general, long-chain-of-thought reasoning, and long-context retrieval benchmarks, TaSQ consistently outperforms existing low-bit KV cache VQ baselines while preserving reasoning stability. On a single RTX 6000 Ada GPU, its SGLang implementation supports up to $14\times$ larger batch sizes and achieves $1.87\times$ higher peak throughput compared to the BF16 baseline. |
| 2026-10-02 | [Evolutionary Computation for Trustworthy AI: From Attacks and Defenses to Self-Evolving Era](http://arxiv.org/abs/2610.02996v1) | Junhao Dong, Chenkai Wang et al. | As Artificial Intelligence (AI) has evolved from task-specific models to foundation models and agents, the scope of trustworthy AI has expanded from model-level robustness to the reliability and safety of broader AI systems. This evolution has also expanded the attack surface from individual models to broader system-level interactions, including tool use, context, and interaction trajectories with dynamic environments. As a result, maintaining reliable and safe behavior under changing or deliberately manipulated conditions has become increasingly challenging. The search for effective attacks and defenses often relies on black-box feedback to navigate discrete choices among words, actions, system components, or their combinations. Multiple objectives and expensive candidate evaluations further limit what can be explored. Evolutionary Computation (EC), with its population-based, gradient-free search and flexible variation and selection mechanisms, is well suited to these settings. This survey reviews how EC has been applied to trustworthy AI across three directions: evolutionary attacks, evolutionary defenses, and trustworthy self-evolving AI systems. Unlike prior reviews that treat trustworthy AI, EC, and self-evolving systems largely separately, we connect these lines through a common evolutionary perspective. For self-evolving AI, we examine how trustworthiness governs the generation and retention of updates that shape subsequent adaptation. We further synthesize evaluation methods and benchmark resources from both trustworthiness and evolutionary-search perspectives. Finally, we discuss key challenges and future research directions toward more effective and reliable use of EC in trustworthy AI. |
| 2026-10-02 | [PLCWorld: Benchmarking LLM-Generated PLC Programs in Closed-Loop Plant Simulation](http://arxiv.org/abs/2610.02982v1) | Yunji Kim, Yunseok Lee et al. | Programmable logic controllers (PLCs) coordinate industrial equipment by reading sensor inputs and issuing control commands. Evaluating whether large language model (LLM)-generated PLC programs satisfy task requirements and safety constraints requires observing how their commands affect device and workpiece states. We introduce PLCWorld, a common closed-loop execution environment and benchmark that couples Structured Text (ST) execution with simulated plant responses and sensor feedback. Grounded in control relations identified in industrial PLC programs and engineering documentation, PLCWorld contains 100 synthetic tasks and 473 registered task-condition pairs across Motion Control and Material Handling, with difficulty defined by control-dependency scope. A common protocol reports Task Success and Safety Violation separately. Validation combines practitioner review, reference and alternative programs, targeted counterexamples, specification-evaluator alignment checks, and comparisons with independent ST runtimes. Reference and alternative programs satisfy their applicable cases, while all 542 targeted counterexamples activate their designated evaluator rules under at least one registered condition. Execution Gap relates submission-profile acceptance to subsequent task failure or observed Safety Violation. Across the constructed task groups, direct GPT-5.5 achieves 82.70% Task Success on Easy cases but 25.10% on Hard cases. Evaluations of six LLMs and four adapted generation-and-verification workflows further expose differences between completion, safety, and generation cost. Our code, simulation environment, benchmark tasks, and baseline implementations are publicly available at https://yunji0516.github.io/PLCWorld/. |
| 2026-10-02 | [Profile-Aware Trustworthy Recipe Generation with Planner-Critic Agentic Remediation](http://arxiv.org/abs/2610.02969v1) | Shanhong Liu, Pai Chet Ng et al. | Recipe generation from food images has practical value for intelligent cooking assistance, but traditional one-pass generation often overlooks user-specific safety requirements such as allergies, dietary restrictions, and preparation constraints. We propose PCAR, a Planner-Critic Agentic Remediation framework for trustworthy recipe generation. PCAR separates recipe planning from safety verification: a Planner Agent extracts ingredients and generates recipe drafts conditioned on the user profile, while a Safety Critic Agent audits each draft and provides structured feedback for remediation when violations are detected. This remediation loop enables unsafe recipes to be revised rather than directly returned or discarded. We evaluate PCAR on real food images with 100 benchmark user profiles across four backbone models, including proprietary and locally served open-source models. Results show that PCAR achieves strong safety and generation performance with capable backbone models, while preserving practical recipe quality. |
| 2026-10-02 | [Positive-Unlabeled Learning for Agent Safety False Alarm Auditing](http://arxiv.org/abs/2610.02925v1) | Xichen Yan, Chongyang Gao et al. | Safety monitors help safeguard language-model agents interacting with external tools and environments, but conservative monitoring can generate many false alarms, consuming extensive review resources and weakening trust in alerts. Because false and genuine alarms often remain interleaved in native monitor scores, obtaining a reliable cutoff still requires substantial manual verification. In practice, a small set of verified-safe non-alarmed trajectories may be available while alarms remain unlabeled, naturally casting false-alarm auditing as a positive-unlabeled (PU) ranking problem. The key challenge is monitor-induced selection, since observed safe references are accepted by the monitor, while the hidden safe alarms of interest are precisely those it incorrectly flags, making the observed positives poorly representative of the positives to be recovered. To address this challenge, we propose a two-stage framework in which Trust-aware PU Supervision adapts safe references toward the alarm domain and protects plausible false alarms from excessive negative pressure, while Reliability-gated Rank Distillation consolidates consistent ordering preferences from multiple PU reference models into a single student. Consensus-guided Structural Refinement then improves the student ranking using hierarchical safe-reference support, alarm relations, and predicted reference consensus. The framework requires no alarm safety labels for fitting and leaves the underlying monitor unchanged. Across mainstream safety monitors, our method achieves a macro AUPRC of $0.6444$, outperforming eight evaluated PU baselines by 5.27--16.98 absolute percentage points; compared with PULDA, the strongest evaluated PU baseline, it recovers 33.3% more false alarms at a 5% review budget. |
| 2026-10-02 | [HASTE: Evolving Agent Harnesses Against Emerging Attacks Using Sparse Evidence](http://arxiv.org/abs/2610.02920v1) | Xiqiao Xiong, Moxin Li et al. | Agent harnesses play a critical role in defenses by enforcing safety constraints to prevent unsafe actions. However, rapidly emerging attacks outpace manual harness adaptation, motivating automated harness evolution. Yet the signals available for harness evolution are often sparse, such as brief descriptions or a few attack examples in threat reports and preprints. To address this limitation, we introduce HASTE, a multi-agent framework that evolves agent harnesses from sparse threat evidence through an adversarial interplay between safety-specification generation and attack-case generation. Safety specifications guide harness updates toward addressing identified safety vulnerabilities, while attack cases probe for remaining safety vulnerabilities after each update. By feeding evaluation outcomes back into both processes, HASTE enables harness evolution against emerging attacks beyond the initially observed evidence. Experimental results across multiple backbone models, attack types, and evidence forms show that HASTE consistently reduces attack success rates while preserving benign-task utility. The code is available at https://github.com/xxiqiao/HASTE. |
| 2026-10-02 | [Frequency Is Not Sensitivity Identifying Safety-Sensitive Experts in Sparse MoE LLM](http://arxiv.org/abs/2610.02910v1) | Md Nurul Absar Siddiky, Liuwan Zhu et al. | Suppressing a small set of routed experts can weaken the safety behavior of a sparse Mixture-of-Experts (MoE) language model without retraining. Which experts to suppress is therefore a security question, and the usual answer is activation frequency, but frequency measures use, not influence. We test an alternative: router-gradient sensitivity, the sensitivity of the sequence loss to the gate weights that select an expert. Across five MoE architectures, we rank experts by each signal on 500 benign and 500 malicious prompts and measure refusal on 100 held-out malicious prompts under two budgets: equal expert counts and equal nominal malicious routing traffic (1%-5%). Under each of the two budgets, router-gradient selection reduces refusals more than activation in 24 of 25 conditions, and more than a ten-trial random mean in all 25. The largest effect is in OLMoE, where refusals fall from 34 to 9 of 100 prompts (73.53% relative) with no degraded outputs, indicating substantive compliance rather than broken generation. After matching expert counts in every layer, gradient selection still produces greater refusal reduction than activation in 23 of 25 conditions, with two ties. An exploratory cross-model analysis links larger malicious-versus-benign concentration gaps to greater peak gradient effects (rho = 0.90; exact two-sided p = 0.083, n = 5). Together, the results support gradient selection under the tested budgets. |
| 2026-10-02 | [PaxosLease in Relativistic Inertial Frames](http://arxiv.org/abs/2610.02879v1) | Márton Trencséni | A lease grants one holder exclusive authority over a resource for a fixed term, and its safety property guarantees that at most one holder has the lease at any time. Under special relativity ``at any time'' has no frame-independent meaning for participants in relative motion. This note first restates the classical safety property with light cones, requiring a holder's expiry to lie in the causal past of any later acquisition by a different holder, an order every observer agrees about. Second, the paper modifies classical PaxosLease: the core change is that the containment and quarantine rules each gain a factor of $k = \sqrt{(1+β)/(1-β)}$, the relativistic Doppler factor, where $β$ is relative speed as a fraction of the speed of light. |

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



