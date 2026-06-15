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
| 2026-06-12 | [Safe Reinforcement Learning of Autonomous Highway Driving: A Unified Framework for Safety and Efficiency](http://arxiv.org/abs/2606.14609v1) | Chufei Yan, Zhihao Cui et al. | Deep reinforcement learning (DRL) offers a compelling route to decision-making for advanced autonomous vehicles (AVs), yet its trial-and-error nature makes it difficult to guarantee safety during training and to achieve both safety and efficiency at deployment. We propose a unified safe reinforcement learning (SRL) framework that integrates safe distance (SD), reward machines (RM), and mixture-of-experts (MoE), termed MoE-RM-SRL. For deployment, SD and RM jointly shape a rule-aware reward that encodes highway traffic regulations and stage-wise objectives, enabling safe and reliable behavior without sacrificing efficiency. For training, we introduce a sparsely gated MoE layer comprising up to 11 deep Q-networks (DQNs); an SD-based gating rule activates a minimal set of experts for lane-keeping and lane-changing, mitigating the instability, discontinuities, and impulsive transients commonly induced by switching between heterogeneous controllers (e.g., MPC/rule-based modules and learned policies). We implement the proposed architecture in CARLA and integrate it with a 6-DoF driver-in-the-loop virtual-reality (DiL-VR) platform. Experiments in stochastic two-lane traffic show that MoE-RM-SRL substantially improves safety and efficiency over state-of-the-art baselines, and the framework naturally extends to multi-lane driving as well as on-ramp merging and exiting scenarios. |
| 2026-06-12 | [A Temporal Planning Framework for Disruption Aware Dynamic Route Optimization in Heterogeneous Railway Systems](http://arxiv.org/abs/2606.14582v1) | Pollob Chandra Ray, Sabah Binte Noor et al. | Efficient route optimization play a vital role in ensuring both safety and punctuality in railway operations. It is very crucial particularly in heterogeneous multi-gauge railway networks with varying train speed, stopping pattern, infrastructure compatibility constraints increase coordination complexity. In single-track systems these challenges are further intensify due to all trains to share the same track and requires frequent track switching.Stochastic disruptions events including blocked tracks, blocked trains, engine failure and speed slowdowns introduces additional unpredictability in operations and deviate the timetable. However, existing studies predominantly focuses on high-level timetabling, omitting operational details such as track switching coordination. As a result leaving decision to human operators, increasing safety risks into railway operations. This study proposes a framework based on temporal planning for dynamic route optimization and disruption management in heterogeneous railway systems. The framework formulates railway operations as a temporal planning problem using PDDL 2.1 with explicitly modeling gauge compatibility constraints and diverse disruption scenarios. It generates conflict-free timestamped operational plans specifying both optimized schedules and executable action sequences. To evaluate the proposed framework, we developed a benchmark problem set with 200 instances using up to 1,000 track points and 120 trains. Two state-of-the-art temporal planners and a plan validator were employed to assessed the framework. The experimental results demonstrate that the framework effectively generates temporal operational plans for heterogeneous railway systems and handles multi-gauge constraints, disruptions, and reduces dependence on manual decision making. |
| 2026-06-12 | [Persuasion Index: A Theory-Guided Framework for Persuasion Analysis](http://arxiv.org/abs/2606.14580v1) | Liancheng Gong, Zhiyang Wang et al. | Identifying persuasive rhetorical cues is critical across domains, from detecting information manipulation and improving AI safety to advancing public health communication. We propose Persuasion Index (PI), a taxonomy of 15 dimensions grounded in persuasion theories from psychology and communication, and one transparent implementation using 55 sub-features built from lexicons and rule-based detectors. The taxonomy is modular: individual detectors can be replaced while preserving the theoretical structure. By evaluating PI on four public datasets varying in domain, style, and outcome measures, we show that PI provides a shared feature space for interpreting rhetorical patterns associated with persuasion-related outcomes. Linear models show that PI features carry meaningful predictive signal while remaining computationally lightweight. Dimension-level analyses reveal recurring associations between PI dimensions and persuasion outcomes across datasets, while also highlighting topic- and stance-specific variation. We release PI as an open-source package and web interface for principled and auditable analysis of human and AI-mediated communication. |
| 2026-06-12 | [Provably Safe, Yet Scalable Reinforcement Learning](http://arxiv.org/abs/2606.14536v1) | Kai S. Yun, Zeyang Li et al. | Safe reinforcement learning (RL) aims to learn policies that optimize rewards while satisfying constraints. Predominant approaches rely on soft-constrained policy optimization, which has achieved empirical success but does not provide formal safety guarantees for the learned policy. In contrast, methods with strict guarantees typically rely on explicit certificate functions, whose construction requires the direct synthesis and verification of control-invariant sets, a process that scales poorly with state dimension and often yields overly conservative behavior. In this paper, we present the Provably Safe, yet Scalable RL (PS2-RL) framework, a novel two-phase architecture for learning provably safe policies in a scalable manner, designed to overcome the key bottlenecks of prior methods. Rather than explicitly computing invariant sets, PS2-RL leverages a learned backup policy to forward-integrate the system dynamics, generating an implicit control-invariant set online. In the first phase, the backup policy is trained with our proposed safe-arrival value function, which characterizes the optimal backup policy for invariant-set construction. In the second phase, an RL policy is trained end-to-end through a differentiable projection layer that strictly enforces the safety guarantees induced by the learned backup policy. By maximizing the volume of the implicit control-invariant set in the first phase, the resulting PS2 policy from the second phase is performant and scalable, while maintaining provable safety. Crucially, PS2-RL imposes no restrictions on the underlying RL algorithm and can be plugged into any existing training pipeline. We establish theoretical guarantees for the proposed framework and evaluate it on robotic control tasks with state dimensions up to 10, a regime in which prior provably safe RL methods struggle or become impractical. |
| 2026-06-12 | [From Shield to Target: Denial-of-Service Attacks on LLM-Based Agent Guardrails](http://arxiv.org/abs/2606.14517v1) | Yuguang Zhou, Xunguang Wang et al. | LLM-based guardrails have emerged as a highly effective defense against prompt injection and jailbreak attacks in autonomous agents. However, we reveal that the very reasoning and task-following capabilities enabling this protection introduce a novel vulnerability: attackers can inject crafted data to trap the guardrail in extended reasoning loops, effectuating a systematic denial-of-service (DoS) attack. To systematically expose this threat, we design a beam-search optimization framework that crafts natural-language payloads to maximize guardrail reasoning length, utilizing an LLM proposer guided by a strategy bank. Based on the observation of guardrail's schema-following nature, we also provide another attack framework driven by mechanism-aware structural mutations with less computational load. The attack efficacy is systematically evaluated in two parts. First, in standalone evaluations, the attack generalizes across diverse guardrail architectures, safety templates, and agent benchmarks. Payloads optimized on a single open-source surrogate successfully transfer to eight leading model backbones (e.g., Claude, GPT, Gemini, DeepSeek, and Qwen), achieving a 13--63$\times$ token amplification. Second, in end-to-end real-world agent deployments (web, desktop, code, and multi-agent systems), the attack reveals up to a 148$\times$ latency amplification. We show that a single poisoned document can saturate shared guardrail infrastructures, effectively starving co-located agents and paralyzing the entire system. By uncovering this availability flaw, our work underscores the urgent need to develop cost-bounded, reasoning-robust guardrails. |
| 2026-06-12 | [Demographic Patterns in Cybersecurity Culture: Insights from a Global Organisation Supporting Safety-Critical and Critical Infrastructure Sectors](http://arxiv.org/abs/2606.14462v1) | Tita Alissa Bach, Amandine Kaiser | This study investigates demographic differences in cybersecurity culture in a large global organisation supporting safety critical and critical infrastructure sectors to target CSC improvement. A global survey was administered to all internal and external employees of a total of 21148 employees, with 6502 responses. The questionnaire evaluates nine CSC dimensions such as Password Management, Governance, Email Use. Anonymous survey responses were analysed using Kruskal-Wallis tests and Dunns post hoc comparisons to identify differences across demographic variables including employment, recruitment paths, managerial role, gender, age, tenure, and work base. CSC was broadly consistent across the organisation, with statistically significant but small to moderate demographic effects. CSC variations were observed across employment, age, recruitment paths, and line managerial role. In general, fulltime, internal, permanent, older employees, Merge and Acquisition recruits, and line managers consistently scored higher across multiple CSC dimensions. Parttime, younger, external employees, and those with 6 to 20 years of tenure in general scored lower. These patterns highlight higher-scoring groups that may act as CSC carriers and lower-scoring groups that may benefit from tailored improvement measures, enabling organisational learning. Our study offers a practical, scalable way to assess CSC, generating meaningful insights despite industrial constraints. It enables organisations to benchmark maturity, identify gaps, and prioritise targeted improvements using workforce diversity as a guide. |
| 2026-06-12 | [Operational forecasting of solar energetic particle event and proton flux using multi-source solar observations and multi-task deep learning](http://arxiv.org/abs/2606.14440v1) | Yian Yu, Yang Chen et al. | Solar energetic particle (SEP) events, defined by proton flux exceeding 10 pfu in the $>$ 10 MeV channel, pose major risks to spacecraft operations, astronaut safety, and high-latitude aviation. Due to the complexity and rarity of SEP events, reliable operational SEP forecasting remains an important challenge in space weather. Here we present a novel 24-hour-ahead operational forecasting framework based on a multi-task learning structure with a thoroughly constructed list of features from multiple sources that span over multiple solar cycles. It extends an earlier-introduced \texttt{SEPNET}-based models by integrating a broader range of solar observations, including active region magnetic parameters from SHARP and SMARP, solar flares information, coronal mass ejections, soft X-ray flux, and historical $>$ 10 MeV proton flux. The inclusion of SMARP data expands the temporal coverage of magnetic-field predictors to earlier solar cycles, while flux-based inputs provide additional precursor information. Building on the multi-task architecture, we develop \texttt{SEPNET-Ov2}, which jointly predicts SEP event occurrence together with future proton flux and soft X-ray flux. Evaluation on the CLEAR SEP benchmark dataset shows improved classification performance over the earlier \texttt{SEPNET-O} (operational version of \texttt{SEPNET}) on the newly aligned dataset. The best operational model is obtained when magnetic, radiative, and proton-flux predictors are combined, highlighting the value of expanded historical coverage and complementary precursor information for improving operational SEP forecasting. |
| 2026-06-12 | [CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning](http://arxiv.org/abs/2606.14415v1) | Ayoub Belouadah, Sylvain Kubler et al. | Safe reinforcement learning (Safe RL) aims to maximize expected return while satisfying safety constraints, typically modeled as Constrained Markov Decision Processes (CMDPs). While primal-dual methods scale well to deep RL, they often suffer from delayed constraint correction, leading to oscillatory behavior and prolonged safety violations. In this paper, we propose Constraint-Sensitive Policy Optimization (CSPO), a first-order primal-dual method that incorporates local constraint sensitivity into policy updates. CSPO augments the primal objective with a constraint-sensitive correction derived from the shortest signed distance to the safety boundary, enabling smarter recovery steps back to safety, compensating for delayed Lagrange multiplier updates, reducing oscillations near the boundary, and preserving the KKT solutions of the original constrained problem. Experiments on navigation and locomotion benchmarks demonstrate that CSPO achieves faster safety recovery and high reward preservation, resulting in higher constrained returns compared to state-of-the-art primal-dual and penalty-based methods |
| 2026-06-12 | [REPOSE: Quantifying the Price of Security in Weakly-Hard Real-Time Cyber-Physical Systems](http://arxiv.org/abs/2606.14395v1) | Vijay Banerjee, Monowar Hasan | In contemporary IoT edge devices with real-time requirements, security is primarily enforced through design-time parameters associated with security tasks, leading to mechanisms that operate in an \emph{opportunistic} manner. As a result, security checks are often performed as secondary operations. This approach can result in systems where no security tasks are executed due to high utilization by other tasks. An alternative approach taken in prior work is to add security mechanisms to every task in the system, resulting in substantially lower performance than that of a system with no security. These approaches have resulted in an \emph{all-or-nothing} scenario for edge device security, motivating numerous studies on the safety-security trade-off in real-time cyber-physical systems (RT-CPS). This study introduces an analytical framework -- REPOSE -- for evaluating the security feasibility of real-time control systems at runtime. REPOSE is developed for \textit{weakly-hard} real-time control systems that facilitate a ``bounded trade-off'' between safety and security. In contrast to imposing additional (pessimistic) design-time overhead as considered in some real-time security literature, REPOSE performs security operations in both \textit{proactive} and \textit{reactive} manners based on the task's current behavior. Our evaluations show that REPOSE can effectively add security operations to RT-CPS with a feasibility overhead of $0.06\%$ at $80\%$ utilization, compared to a $ 29\%$ overhead observed in systems with hard constraints. Through a case study of a classic control system, we also demonstrate that REPOSE provides a robust framework to \textit{analyze and calculate} the safety-security tradeoff. |
| 2026-06-12 | [A Low-Rank Subspace Analysis of LLM Interventions](http://arxiv.org/abs/2606.14388v1) | Angira Sharma, Christian Schroeder de Witt et al. | Interventions designed to modify a particular behavior in LLMs, such as refusal or sycophancy, often produce unintended changes in other behaviors. This lack of targeted control makes it difficult to design and implement reliable safety controls. To understand these side-effects, we introduce a diagnostic framework for analyzing interacting behaviors in LLMs. We model behaviors as low-rank subspaces in activation space, and study how interventions influence across behaviors. Across multiple instruction-tuned models (7B-70B) and across refusal, jailbreak, and sycophancy settings, we find that different behaviors share internal representations, and intervening on one behavior alters others in asymmetric ways. Some behaviors act as upstream control points whose interventions propagate broadly across other behaviors, while others remain more isolated. We relate these effects to two geometric quantities: (i) the overlap between behavior subspaces, measured as the average squared cosine of principal angles, and (ii) the angle between each behavior subspace and the decision subspace (capturing the model's final decision e.g., refuse vs. comply). Empirically, intervention effects on other behaviors tend to be larger for behavior pairs with higher subspace overlap, and for source behaviors whose subspaces lie closer (smaller angle) to the decision subspace. These findings highlight a challenge for targeted behavior control: behaviors are difficult to modify independently, as interventions can propagate through shared representations and asymmetric interactions. |
| 2026-06-12 | [IndustryBench-MIPU: Benchmarking Multi-Image Attribute Value Extraction for Industrial Products](http://arxiv.org/abs/2606.14383v1) | Haonan Qi, Jin Cao et al. | Industrial products such as valves and circuit breakers are defined by dense technical specifications that govern procurement, compatibility, and safety across supply chains. These specifications are scattered across multiple heterogeneous product images, including specification tables, nameplates, and technical drawings, yet whether Multimodal Large Language Models (MLLMs) can reliably recover them remains underexplored. To fill this gap, we introduce IndustryBench-MIPU, the first large-scale benchmark for multi-image industrial product understanding, built around structured attribute extraction -- recovering property-value pairs from product images. This task jointly probes text recognition on specification tables and nameplates, visual reasoning over technical drawings, domain knowledge to decode industrial terminology, and cross-image evidence integration to assemble scattered specifications. Concretely, the benchmark comprises 4,559 products across 27,652 images with 103,703 annotations spanning 18 industrial categories, constructed through multi-model consensus and three-tier quality assurance. Evaluating nine MLLMs under both single-image and product-level multi-image settings reveals a stark completeness gap: models achieve high precision (86--94%) but the best recovers only 49.9% of product-level attributes; moving from single-image to multi-image extraction costs 15--34 percentage points of recall. Multi-image completeness, not single-image accuracy, is the core bottleneck. Dataset and code are publicly available. |
| 2026-06-12 | [ForceForget: Reinforcement Concept Removal for Enhancing Safety in Text-to-Image Models](http://arxiv.org/abs/2606.14351v1) | Dong Han, Yong Li | With the advance of generative AI, the text-to-image (T2I) model has the ability to generate various contents. However, T2I models still can generate unsafe contents. To alleviate this issue, various concept erasing methods are proposed. However, existing methods tend to excessively erase unsafe concepts and suppress benign concepts contained in harmful prompts, which can negatively affect model utility. In this paper, we focus on eliminating unsafe content while maintaining model capability in safe semantic meaning interpretation by optimizing the concept erasing reward (CER) with reinforcement learning. To avoid overly content erasure, we introduce the Safe Adapter to project partial text embedding for efficient concept regulation in cross-attention layers. Extensive experiments conducted on different datasets demonstrate the effectiveness of the proposed method in alleviating unsafe content generation while preserving the high fidelity of benign images compared with existing state-of-the-art (SOTA) concept erasing methods. In terms of robustness, our method outperforms counterparts against red-teaming tools. Moreover, we showcase the proposed approach is more effective in emerging image-to-image (I2I) scenarios compared with others. Lastly, we extend our method to erase general concepts, such as artistic styles and objects. Disclaimer: This paper includes discussions of sexually explicit content that may be offensive to certain readers. All images used in this work are synthesized or from public datasets. |
| 2026-06-12 | [I'm Sorry Driver, I'm Afraid I Can't Do That: Appraising the Safety of LLMs within Automotive Contexts](http://arxiv.org/abs/2606.14327v1) | Shaun Feakins, Ibrahim Habli et al. | This paper appraises recent frameworks within AI development to integrate LLMs into control tasks in automotive contexts from the perspective of safety assurance. This work has built upon the rapid integration of LLMs across automotive settings. However, we find that at present, these frameworks face significant challenges, limiting their efficacy in real-time safety-critical contexts. Firstly, we consider conceptual challenges, including the fact that deployers are faced with a dual challenge, wherein they must assure a model which has been developed upstream, i.e. as general-purpose tools by the large AI labs, in a downstream context, i.e. into specific vehicle architectures. Secondly, we consider concrete challenges from across existing standards. We show that there are currently both fundamental engineering constraints covered in ISO21448, such as latency, and novel LLM-specific issues, such as alignment-related issues covered in ISO/PAS8800. We ground both examples in a concrete introductory, experimental case study exploring an existing open-source repository, Talk2Drive. We present a safety argument in order to make explicit the limitations of existing solutions. Nonetheless, given that the use of LLMs in automotive contexts is being explored at a technical level and operationalised, we propose potential assurance mechanisms for LLM-related hazardous events going forward. |
| 2026-06-12 | [When and How Severely: Scenario-Specific Safety Envelopes for Driving VLAs](http://arxiv.org/abs/2606.14238v1) | Abhinaw Priyadershi, Jelena Frtunikj | Safety certification of Vision-Language-Action (VLA) driving planners under ISO 21448 (SOTIF) rests on an Operational Design Domain (ODD) specification that answers two complementary questions: when does the planner start to fail, and how severely does it fail once it does? We evaluate Alpamayo R1, a 10B-parameter open-weight driving VLA, on 15,968 (clip, attack) pairs. We find a conservative-aggregate gap: an aggregate safe threshold of $σ\leq 50$ under a 15% average displacement error (ADE) budget masks well-sampled scenarios that tolerate the top of the tested grid ($σ= 70$). A Gaussian Mixture Model (GMM) on the changed-explanation subset identifies six discrete severity bands (BIC-optimal $k{=}6$), so two perturbation conditions with the same mean error can differ materially in their share of high-severity (C4/C5) failures. Joining the two analyses on the same corpus surfaces a finding neither yields in isolation: the scenarios with the loosest noise thresholds are not those with the lowest high-severity rate: STOP_SIGNAL concentrates roughly $4\times$ the C4/C5 share of LANE_KEEPING despite tolerating a larger $σ$. A deployable SOTIF ODD specification for driving VLAs therefore requires a two-dimensional safety envelope, not a single aggregate value per hazard. |
| 2026-06-12 | [Selective Agentic Recovery for UAV Autonomy with a Persistent Mission Runtime](http://arxiv.org/abs/2606.14219v1) | Taewoo Park, Kyeonghyun Yoo et al. | Agentic AI can support unmanned aerial vehicle (UAV) autonomy by providing high-level recovery reasoning when local waypoint- or setpoint-based execution encounters blocked passages, repeated no-progress behavior, or mission-level ambiguity. On physical UAVs, however, remote reasoning is most useful when it is invoked selectively, since each call introduces latency, resource cost, backend uncertainty, and a need to validate the returned decision. This paper presents Persistent Mission Runtime (PMR), a UAV recovery framework that keeps the mission loop and safety-critical execution local while using an external agentic reasoner only as an on-demand recovery module. The reasoner selects from predefined recovery skills, and each returned decision is parsed, verified, safety-filtered, and mapped to local executor actions before it can affect flight. PMR introduces learned Cognitive Value of Invocation (learned-CVI), a compact admission gate that estimates when remote agentic reasoning is likely to improve near-term mission progress enough to justify its operational cost. Across a fixed 400-run Gazebo/PX4 benchmark with eight scenarios, learned-CVI raises hard/ambiguous-regime success from 5.0% under local-only autonomy to 95.0%, outperforms one-shot and periodic reasoning baselines by 20.0 and 32.5 percentage points, and reduces remote-agent calls by 16.7% and logged tokens by 29.2% relative to a manually tuned rule-based invocation baseline. |
| 2026-06-12 | [Short-Horizon Position Accuracy of Single-Track Models: Implications for Motion Planning of Autonomous Vehicles](http://arxiv.org/abs/2606.14216v1) | Aron J. Aertssen, Lars A. T. H. van Alen et al. | Accurate and computationally efficient vehicle models are essential for motion planning of autonomous vehicles, where positional accuracy directly affects trajectory feasibility and safety. However, the positional accuracy has not been systematically evaluated against real measurements. Therefore, this paper compares the short-horizon positional accuracy of three single-track vehicle models against vehicle measurements across various driving maneuvers. Model parameters are identified through dedicated experiments with the instrumented test vehicle. Rather than identifying a single best model, this work aims to provide insight into the trade-offs between model complexity, parameterization quality, and positional accuracy for informed model selection in Model Predictive Control applications. |
| 2026-06-12 | [Robustness without Wrinkles: Parallel Simulation and Robust MPC for Certified Deformable Manipulation](http://arxiv.org/abs/2606.14188v1) | Wei-Chen Li, Jeffrey Fang et al. | We present CORD-SLS, a real-time control method for safe deformable object manipulation, with a focus on ropes and cloth. At its core is a GPU-parallel differentiable simulator with contact smoothing which enables efficient gradient-based planning through intermittent contact. To robustly satisfy constraints under model and sensing uncertainty, we develop a real-time, GPU-parallel output-feedback robust model predictive control (MPC) algorithm that plans with this simulator. We further show that the simulator accelerates model-based RL for training neural manipulation policies. To improve real-world robustness, we use conformal prediction to calibrate visual-feedback and perception-error bounds for MPC, producing reachable tubes that enable high-probability safe control. We evaluate CORD-SLS on high-dimensional, contact-rich rope and cloth manipulation tasks in simulation and hardware, including obstacle avoidance, routing, folding, and smoothing. Across settings, CORD-SLS achieves millisecond-speed planning, exceeding baselines in safety, speed, and task success. |
| 2026-06-12 | [SkillMutator: Benchmarking and Defending Language-and-Code Cross-modal Attacks on LLM Agent Skills](http://arxiv.org/abs/2606.14154v1) | Youngduk Kim, Minkyoo Song et al. | Large language model (LLM) agents increasingly extend their capabilities at runtime by loading Agent Skills, which pair natural-language specifications (SKILL.md) with executable scripts and resources. Because a skill's behavior relies on both natural-language instructions and executable code, assessing its safety requires cross-modal reasoning, creating a new language-and-code attack surface. Attackers can present a benign workflow in SKILL.md while embedding implicit directives that steer the agent to exfiltrate sensitive files, even if the scripts appear harmless. This attack surface remains understudied; prior work treats skills merely as prompt-injection vectors or static code artifacts, leaving attacks emerging from cross-modal interactions largely unmeasured. In our evaluation, open-source and commercial skill scanners detect only 2%-8% and 9%-17% of such attacks, respectively. To address this gap, we introduce SkillMutator, the first benchmark for install-time detection of language-and-code cross-modal attacks on Agent Skills. It emulates an adversarial mutation process across 13 attack categories, iteratively refining malicious skills using scanner feedback to make injected behaviors indistinguishable from legitimate workflows. We further propose a four-phase reasoning-trajectory distillation framework to distill frontier-teacher traces into smaller open-weight models. This produces a locally deployable scanner avoiding third-party data exposure and excessive API costs. On the strongest SkillMutator subset (n=76), our distilled model (Qwen2.5-Coder-7B-Instruct) improves detection from 17.1% to 88.2%, surpassing GPT-4o-mini (23.7%) and GPT-5.4-mini (79.0%), and reaching frontier-level GPT-5.4 (86.8%). These results show practical defense against cross-modal attacks is feasible without relying on costly frontier models. |
| 2026-06-12 | [Trust but Verify: Mitigating Medical Hallucinations via Post-Hoc Adversarial Auditing and Multi-Agent Feedback Loops](http://arxiv.org/abs/2606.14149v1) | Muhammad Osama, Maheera Amjad et al. | Large Language Models (LLMs) are increasingly deployed in healthcare settings, yet their tendency to hallucinate poses risks when clinical decisions are involved. This study examine whether LLMs recommend recently banned or withdrawn pharmaceuticals when answering clinical questions and tests an agent-based method for reducing such errors. We developed a five-agent "Trust but Verify" system using a single LLM backbone. To measure regulatory knowledge obsolescence, we created an adversarial dataset of 103 clinical MCQs where historically correct answers now refer to banned substances. This scale ensures statistical significance across various therapeutic classes. We evaluated three open-access model families (GPT-OSS, Llama-3, Falcon-3) under vanilla and agentic conditions. Performance was measured via pointwise score, label accuracy, Hallucination Error Rate (HER), and Component Fidelity (CF) score. We also observed clinical safety regression in proprietary models. In default configurations, all models showed high hallucination rates, consistently selecting banned drugs that matched training data patterns. Our proposed agentic architecture reduced HER by approximately 53% across models. Pointwise scores shifted from -0.25 (unsafe recommendation) toward 0.0 (appropriate refusal). The safety audit intercepted dangerous outputs even when models' parametric knowledge favored the banned substance. The proposed multi-agent framework offers a model-agnostic method for enforcing regulatory compliance that prioritizes patient safety over fluent text generation. Our work demonstrates a practical approach for deploying autonomous AI systems in safety-critical healthcare settings. It shows how real-time regulatory data can be integrated into LLM pipelines to support clinical decision-making. |
| 2026-06-12 | [Personal Care Utility: Health as Everyday Infrastructure](http://arxiv.org/abs/2606.14145v1) | Mahyar Abbasian, Elahe Khatibi et al. | Healthcare is essential, expert, and episodic by design - built around the roughly one hour per year a person spends with a clinician. The 8,759 hours outside clinical settings, where eating, sleeping, movement, medication, and stress actually shape long-term health, have no comparable infrastructure. The bottleneck for personalized health is not raw data or reasoning capability; it is the absence of that infrastructure layer. This paper introduces the Personal Care Utility (PCU): a layered, event-driven architecture proposed as the missing utility for everyday health, in the way that payments, networks, and power are utilities for their domains. PCU organizes continuous personal signals into semantically meaningful life events through a Personicle, estimates dynamic health state against personal baselines, reasons about cause and context, and routes guidance through an orchestrator that separates clinical decision logic, behavioral strategy selection, and natural-language expression. This separation lets large language models support reasoning and communication while keeping safety-critical clinical decisions grounded in validated evidence. We instantiate PCU for Type 2 Diabetes - turning CGM, meal, activity, medication, sleep, stress, and clinical data into glycemic events, individualized state estimates, causal explanations, and knowledge-grounded interventions. A day-in-the-life scenario shows the same infrastructure producing real-time nudges, weekly summaries, medication check-ins, silence, or deterministic safety alerts depending on context and risk. We close with how PCU generalizes to other chronic conditions and the governance questions any always-on personal health utility must address. The result is a blueprint that treats personalization not as a final messaging layer, but as an architectural property of everyday health guidance. |

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



