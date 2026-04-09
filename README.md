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
| 2026-04-08 | [Intertemporal Demand Allocation for Inventory Control in Online Marketplaces](http://arxiv.org/abs/2604.07312v1) | Rene Caldentey, Tong Xie | Online marketplaces increasingly do more than simply match buyers and sellers: they route orders across competing sellers and, in many categories, offer ancillary fulfillment services that make seller inventory a source of platform revenue. We investigate how a platform can use intertemporal demand allocation to influence sellers' inventory choices without directly controlling stock. We develop a model in which the platform observes aggregate demand, allocates orders across sellers over time, and sellers choose between two fulfillment options, fulfill-by-merchant (FBM) and fulfill-by-platform (FBP), while replenishing inventory under state-dependent base-stock policies. The key mechanism we study is informational: by changing the predictability of each seller's sales stream, the platform changes sellers' safety-stock needs even when average demand shares remain unchanged. We focus on nondiscriminatory allocation policies that give sellers the same demand share and forecast risk. Within this class, uniform splitting minimizes forecast uncertainty, whereas any higher level of uncertainty can be implemented using simple low-memory allocation rules. Moreover, increasing uncertainty above the uniform benchmark requires routing rules that prevent sellers from inferring aggregate demand from their own sales histories. These results reduce the platform's problem to choosing a level of forecast uncertainty that trades off adoption of platform fulfillment against the inventory held by adopters. Our analysis identifies demand allocation as a powerful operational and informational design lever in digital marketplaces. |
| 2026-04-08 | [Validated Intent Compilation for Constrained Routing in LEO Mega-Constellations](http://arxiv.org/abs/2604.07264v1) | Yuanhang Li | Operating LEO mega-constellations requires translating high-level operator intents ("reroute financial traffic away from polar links under 80 ms") into low-level routing constraints -- a task that demands both natural language understanding and network-domain expertise. We present an end-to-end system comprising three components: (1) a GNN cost-to-go router that distills Dijkstra-quality routing into a 152K-parameter graph attention network achieving 99.8% packet delivery ratio with 17x inference speedup; (2) an LLM intent compiler that converts natural language to a typed constraint intermediate representation using few-shot prompting with a verifier-feedback repair loop, achieving 98.4% compilation rate and 87.6% full semantic match on feasible intents in a 240-intent benchmark (193 feasible, 47 infeasible); and (3) an 8-pass deterministic validator with constructive feasibility certification that achieves 0% unsafe acceptance on all 47 infeasible intents (30 labeled + 17 discovered by Pass 8), with 100% corruption detection across 240 structural corruption tests and 100% on 15 targeted adversarial attacks. End-to-end evaluation across four constrained routing scenarios confirms zero constraint violations with both routers. We further demonstrate that apparent performance gaps in polar-avoidance scenarios are largely explained by topological reachability ceilings rather than routing quality, and that the LLM compiler outperforms a rule-based baseline by 46.2 percentage points on compositional intents. Our system bridges the semantic gap between operator intent and network configuration while maintaining the safety guarantees required for operational deployment. |
| 2026-04-08 | [BATON: A Multimodal Benchmark for Bidirectional Automation Transition Observation in Naturalistic Driving](http://arxiv.org/abs/2604.07263v1) | Yuhang Wang, Yiyao Xu et al. | Existing driving automation (DA) systems on production vehicles rely on human drivers to decide when to engage DA while requiring them to remain continuously attentive and ready to intervene. This design demands substantial situational judgment and imposes significant cognitive load, leading to steep learning curves, suboptimal user experience, and safety risks from both over-reliance and delayed takeover. Predicting when drivers hand over control to DA and when they take it back is therefore critical for designing proactive, context-aware HMI, yet existing datasets rarely capture the multimodal context, including road scene, driver state, vehicle dynamics, and route environment. To fill this gap, we introduce BATON, a large-scale naturalistic dataset capturing real-world DA usage across 127 drivers, and 136.6 hours of driving. The dataset synchronizes front-view video, in-cabin video, decoded CAN bus signals, radar-based lead-vehicle interaction, and GPS-derived route context, forming a closed-loop multimodal record around each control transition. We define three benchmark tasks: driving action understanding, handover prediction, and takeover prediction, and evaluate baselines spanning sequence models, classical classifiers, and zero-shot VLMs. Results show that visual input alone is insufficient for reliable transition prediction: front-view video captures road context but not driver state, while in-cabin video reflects driver readiness but not the external scene. Incorporating CAN and route-context signals substantially improves performance over video-only settings, indicating strong complementarity across modalities. We further find takeover events develop more gradually and benefit from longer prediction horizons, whereas handover events depend more on immediate contextual cues, revealing an asymmetry with direct implications for HMI design in assisted driving systems. |
| 2026-04-08 | [Designing Safe and Accountable GenAI as a Learning Companion with Women Banned from Formal Education](http://arxiv.org/abs/2604.07253v1) | Hamayoon Behmanush, Freshta Akhtari et al. | In gender-restrictive and surveilled contexts, where access to formal education may be restricted for women, pursuing education involves safety and privacy risks. When women are excluded from schools and universities, they often turn to online self-learning and generative AI (GenAI) to pursue their educational and career aspirations. However, we know little about what safe and accountable GenAI support is required in the context of surveillance, household responsibilities, and the absence of learning communities. We present a remote participatory design study with 20 women in Afghanistan, informed by a recruitment survey (n = 140), examining how participants envision GenAI for learning and employability. Participants describe using GenAI less as an information source and more as an always-available peer, mentor, and source of career guidance that helps compensate for the absence of learning communities. At the same time, they emphasize that this companionship is constrained by privacy and surveillance risks, contextually unrealistic and culturally unsafe support, and direct-answer interactions that can undermine learning by creating an illusion of progress. Beyond eliciting requirements, envisioning the future with GenAI through participatory design was positively associated with significant increases in participants' aspirations (p=.01), perceived agency (p=.01), and perceived avenues (p=.03). These outcomes show that accountable and safe GenAI is not only about harm reduction but can also actively enable women to imagine and pursue viable learning and employment futures. Building on this, we translate participants' proposals into accountability-focused design directions that center on safety-first interaction and user control, context-grounded support under constrained resources, and offer pedagogically aligned assistance that supports genuine learning rather than quick answers. |
| 2026-04-08 | [TraceSafe: A Systematic Assessment of LLM Guardrails on Multi-Step Tool-Calling Trajectories](http://arxiv.org/abs/2604.07223v1) | Yen-Shan Chen, Sian-Yao Huang et al. | As large language models (LLMs) evolve from static chatbots into autonomous agents, the primary vulnerability surface shifts from final outputs to intermediate execution traces. While safety guardrails are well-benchmarked for natural language responses, their efficacy remains largely unexplored within multi-step tool-use trajectories. To address this gap, we introduce TraceSafe-Bench, the first comprehensive benchmark specifically designed to assess mid-trajectory safety. It encompasses 12 risk categories, ranging from security threats (e.g., prompt injection, privacy leaks) to operational failures (e.g., hallucinations, interface inconsistencies), featuring over 1,000 unique execution instances. Our evaluation of 13 LLM-as-a-guard models and 7 specialized guardrails yields three critical findings: 1) Structural Bottleneck: Guardrail efficacy is driven more by structural data competence (e.g., JSON parsing) than semantic safety alignment. Performance correlates strongly with structured-to-text benchmarks ($ρ=0.79$) but shows near-zero correlation with standard jailbreak robustness. 2) Architecture over Scale: Model architecture influences risk detection performance more significantly than model size, with general-purpose LLMs consistently outperforming specialized safety guardrails in trajectory analysis. 3) Temporal Stability: Accuracy remains resilient across extended trajectories. Increased execution steps allow models to pivot from static tool definitions to dynamic execution behaviors, actually improving risk detection performance in later stages. Our findings suggest that securing agentic workflows requires jointly optimizing for structural reasoning and safety alignment to effectively mitigate mid-trajectory risks. |
| 2026-04-08 | [CSA-Graphs: A Privacy-Preserving Structural Dataset for Child Sexual Abuse Research](http://arxiv.org/abs/2604.07132v1) | Carlos Caetano, Camila Laranjeira et al. | Child Sexual Abuse Imagery (CSAI) classification is an important yet challenging problem for computer vision research due to the strict legal and ethical restrictions that prevent the public sharing of CSAI datasets. This limitation hinders reproducibility and slows progress in developing automated methods. In this work, we introduce CSA-Graphs, a privacy-preserving structural dataset. Instead of releasing the original images, we provide structural representations that remove explicit visual content while preserving contextual information. CSA-Graphs includes two complementary graph-based modalities: scene graphs describing object relationships and skeleton graphs encoding human pose. Experiments show that both representations retain useful information for classifying CSAI, and that combining them further improves performance. This dataset enables broader research on computer vision methods for child safety while respecting legal and ethical constraints. |
| 2026-04-08 | [Yale-DM-Lab at ArchEHR-QA 2026: Deterministic Grounding and Multi-Pass Evidence Alignment for EHR Question Answering](http://arxiv.org/abs/2604.07116v1) | Elyas Irankhah, Samah Fodeh | We describe the Yale-DM-Lab system for the ArchEHR-QA 2026 shared task. The task studies patient-authored questions about hospitalization records and contains four subtasks (ST): clinician-interpreted question reformulation, evidence sentence identification, answer generation, and evidence-answer alignment. ST1 uses a dual-model pipeline with Claude Sonnet 4 and GPT-4o to reformulate patient questions into clinician-interpreted questions. ST2-ST4 rely on Azure-hosted model ensembles (o3, GPT-5.2, GPT-5.1, and DeepSeek-R1) combined with few-shot prompting and voting strategies. Our experiments show three main findings. First, model diversity and ensemble voting consistently improve performance compared to single-model baselines. Second, the full clinician answer paragraph is provided as additional prompt context for evidence alignment. Third, results on the development set show that alignment accuracy is mainly limited by reasoning. The best scores on the development set reach 88.81 micro F1 on ST4, 65.72 macro F1 on ST2, 34.01 on ST3, and 33.05 on ST1. |
| 2026-04-08 | [Decision-focused Conservation Voltage Reduction to Consider the Cascading Impact of Forecast Errors](http://arxiv.org/abs/2604.07106v1) | Qintao Du, Ran Li et al. | Conservation Voltage Reduction (CVR) relies on the effective coordination of slow-acting devices, such as OLTCs and CBs, and fast-acting devices, such as SVGs and PV inverters, typically implemented through a hierarchical multi-stage Volt-Var Control (VVC) spanning day-ahead scheduling, intra-day dispatch, and real-time control. However, existing sequential methods fail to account for the cas-cading impact of forecast errors on multi-stage decision-making. This oversight results in suboptimal day-ahead schedules for OLTCs and CBs that hinder the ef-fective coordination with fast-acting SVGs and inverters, inevitably driving a trade-off between real-time voltage security and CVR efficiency. To improve the Pareto front of this trade-off, this paper proposes a novel bi-level multi-timescale forecasting (Bi-MTF) framework for multi-stage VVC optimization. By integrating the downstream multi-stage VVC optimization into the upstream forecasting mod-els training, the decision-focused forecasting models are able to learn the trade-offs across temporal horizons. To solve the computationally challenging bi-level for-mulation, a modified sensitivity-driven integer L-shaped method is developed. It utilizes a hybrid gradient feedback mechanism that integrates numerical sensitivity analysis for discrete variables with analytical dual information for continuous fore-cast parameters to ensure tractability. Numerical results on a modified IEEE 33-bus system demonstrate that the proposed approach yields superior energy savings and operational safety compared to conventional MSE-based sequential paradigms. Specifically, as the capacity of fast-acting devices increases, the energy savings of the proposed method rise from 2.74% to 3.41%, which is far superior to the 1.50% to 1.76% achieved by conventional MSE-based sequential paradigms. |
| 2026-04-08 | [AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules](http://arxiv.org/abs/2604.07039v1) | Xue Qin, Simin Luan et al. | Robotic systems lack a principled abstraction for organizing intelligence, capabilities, and execution in a unified manner. Existing approaches either couple skills within monolithic architectures or decompose functionality into loosely coordinated modules or multiple agents, often without a coherent model of identity and control authority. We argue that a robot should be modeled as a single persistent intelligent subject whose capabilities are extended through installable packages. We formalize this view as AEROS (Agent Execution Runtime Operating System), in which each robot corresponds to one persistent agent and capabilities are provided through Embodied Capability Modules (ECMs). Each ECM encapsulates executable skills, models, and tools, while execution constraints and safety guarantees are enforced by a policy-separated runtime. This separation enables modular extensibility, composable capability execution, and consistent system-level safety. We evaluate a reference implementation in PyBullet simulation with a Franka Panda 7-DOF manipulator across eight experiments covering re-planning, failure recovery, policy enforcement, baseline comparison, cross-task generality, ECM hot-swapping, ablation, and failure boundary analysis. Over 100 randomized trials per condition, AEROS achieves 100% task success across three tasks versus baselines (BehaviorTree.CPP-style and ProgPrompt-style at 92--93%, flat pipeline at 67--73%), the policy layer blocks all invalid actions with zero false acceptances, runtime benefits generalize across tasks without task-specific tuning, and ECMs load at runtime with 100% post-swap success. |
| 2026-04-08 | [Strategic Persuasion with Trait-Conditioned Multi-Agent Systems for Iterative Legal Argumentation](http://arxiv.org/abs/2604.07028v1) | Philipp D. Siedler | Strategic interaction in adversarial domains such as law, diplomacy, and negotiation is mediated by language, yet most game-theoretic models abstract away the mechanisms of persuasion that operate through discourse. We present the Strategic Courtroom Framework, a multi-agent simulation environment in which prosecution and defense teams composed of trait-conditioned Large Language Model (LLM) agents engage in iterative, round-based legal argumentation. Agents are instantiated using nine interpretable traits organized into four archetypes, enabling systematic control over rhetorical style and strategic orientation.   We evaluate the framework across 10 synthetic legal cases and 84 three-trait team configurations, totaling over 7{,}000 simulated trials using DeepSeek-R1 and Gemini~2.5~Pro. Our results show that heterogeneous teams with complementary traits consistently outperform homogeneous configurations, that moderate interaction depth yields more stable verdicts, and that certain traits (notably quantitative and charismatic) contribute disproportionately to persuasive success. We further introduce a reinforcement-learning-based Trait Orchestrator that dynamically generates defense traits conditioned on the case and opposing team, discovering strategies that outperform static, human-designed trait combinations.   Together, these findings demonstrate how language can be treated as a first-class strategic action space and provide a foundation for building autonomous agents capable of adaptive persuasion in multi-agent environments. |
| 2026-04-08 | [Differentiable Environment-Trajectory Co-Optimization for Safe Multi-Agent Navigation](http://arxiv.org/abs/2604.06972v1) | Zhan Gao, Gabriele Fadini et al. | The environment plays a critical role in multi-agent navigation by imposing spatial constraints, rules, and limitations that agents must navigate around. Traditional approaches treat the environment as fixed, without exploring its impact on agents' performance. This work considers environment configurations as decision variables, alongside agent actions, to jointly achieve safe navigation. We formulate a bi-level problem, where the lower-level sub-problem optimizes agent trajectories that minimize navigation cost and the upper-level sub-problem optimizes environment configurations that maximize navigation safety. We develop a differentiable optimization method that iteratively solves the lower-level sub-problem with interior point methods and the upper-level sub-problem with gradient ascent. A key challenge lies in analytically coupling these two levels. We address this by leveraging KKT conditions and the Implicit Function Theorem to compute gradients of agent trajectories w.r.t. environment parameters, enabling differentiation throughout the bi-level structure. Moreover, we propose a novel metric that quantifies navigation safety as a criterion for the upper-level environment optimization, and prove its validity through measure theory. Our experiments validate the effectiveness of the proposed framework in a variety of safety-critical navigation scenarios, inspired from warehouse logistics to urban transportation. The results demonstrate that optimized environments provide navigation guidance, improving both agents' safety and efficiency. |
| 2026-04-08 | [TRAPTI: Time-Resolved Analysis for SRAM Banking and Power Gating Optimization in Embedded Transformer Inference](http://arxiv.org/abs/2604.06955v1) | Jan Klhufek, Alberto Marchisio et al. | Transformer neural networks achieve state-of-the-art accuracy across language and vision tasks, but their deployment on embedded hardware is hindered by stringent area, latency, and energy constraints. During inference, performance and efficiency are increasingly dominated by the Key--Value (KV) cache, whose memory footprint grows with sequence length, straining on-chip memory utilization. Although existing mechanisms such as Grouped-Query Attention (GQA) reduce KV cache requirements compared to Multi-Head Attention (MHA), effectively exploiting this reduction requires understanding how on-chip memory demand evolves over time. This work presents TRAPTI, a two-stage methodology that combines cycle-level inference simulation with time-resolved analysis of on-chip memory occupancy to guide design decisions. In the first stage, the framework obtains memory occupancy traces and memory access statistics from simulation. In the second stage, the framework leverages the traces to explore banked memory organizations and power-gating configurations in an offline optimization flow. We apply this methodology to GPT-2 XL and DeepSeek-R1-Distill-Qwen-1.5B under the same accelerator configuration, enabling a direct comparison of MHA and GQA memory profiles. The analysis shows that DeepSeek-R1-Distill-Qwen-1.5B exhibits a 2.72x reduction in peak on-chip memory utilization in this setting compared to GPT-2 XL, unlocking further opportunities for power-gating optimization. |
| 2026-04-08 | [Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](http://arxiv.org/abs/2604.06950v1) | Zhiheng Li, Zongyang Ma et al. | Multimodal Large Language Models (MLLMs) are increasingly being deployed as automated content moderators. Within this landscape, we uncover a critical threat: Adversarial Smuggling Attacks. Unlike adversarial perturbations (for misclassification) and adversarial jailbreaks (for harmful output generation), adversarial smuggling exploits the Human-AI capability gap. It encodes harmful content into human-readable visual formats that remain AI-unreadable, thereby evading automated detection and enabling the dissemination of harmful content. We classify smuggling attacks into two pathways: (1) Perceptual Blindness, disrupting text recognition; and (2) Reasoning Blockade, inhibiting semantic understanding despite successful text recognition. To evaluate this threat, we constructed SmuggleBench, the first comprehensive benchmark comprising 1,700 adversarial smuggling attack instances. Evaluations on SmuggleBench reveal that both proprietary (e.g., GPT-5) and open-source (e.g., Qwen3-VL) state-of-the-art models are vulnerable to this threat, producing Attack Success Rates (ASR) exceeding 90%. By analyzing the vulnerability through the lenses of perception and reasoning, we identify three root causes: the limited capabilities of vision encoders, the robustness gap in OCR, and the scarcity of domain-specific adversarial examples. We conduct a preliminary exploration of mitigation strategies, investigating the potential of test-time scaling (via CoT) and adversarial training (via SFT) to mitigate this threat. Our code is publicly available at https://github.com/zhihengli-casia/smugglebench. |
| 2026-04-08 | [Learning-Based Strategy for Composite Robot Assembly Skill Adaptation](http://arxiv.org/abs/2604.06949v1) | Khalil Abuibaid, Aleksandr Sidorenko et al. | Contact-rich robotic skills remain challenging for industrial robots due to tight geometric tolerances, frictional variability, and uncertain contact dynamics, particularly when using position-controlled manipulators. This paper presents a reusable and encapsulated skill-based strategy for peg-in-hole assembly, in which adaptation is achieved through Residual Reinforcement Learning (RRL). The assembly process is represented using composite skills with explicit pre-, post-, and invariant conditions, enabling modularity, reusability, and well-defined execution semantics across task variations. Safety and sample efficiency are promoted through RRL by restricting adaptation to residual refinements within each skill during contact-rich interactions, while the overall skill structure and execution flow remain invariant. The proposed approach is evaluated in MuJoCo simulation on a UR5e robot equipped with a Robotiq gripper and trained using SAC and JAX. Results demonstrate that the proposed formulation enables robust execution of assembly skills, highlighting its suitability for industrial automation. |
| 2026-04-08 | [Physics-driven Sonification for Improving Multisensory Needle Guidance in Percutaneous Epicardial Access](http://arxiv.org/abs/2604.06911v1) | Veronica Ruozzi, Sasan Matinfar et al. | Percutaneous epicardial access (PEA), performed on a beating heart under fluoroscopy, enables arrhythmia treatment. However, advancing a needle toward the thin and moving pericardium remains highly challenging and risky. To address this problem, we present a physics-driven sonification method for Extended Reality (XR)-based multisensory navigation to enhance user perception during the critical needle landing phase in PEA.   Dynamic cardiac anatomy from 4D CTA was reconstructed and registered to a real-world coordinate system. Real-time needle tracking provided the position of the needle tip relative to moving cardiac structures and drove an audio-visual feedback module. The visual display presented navigational cues and dynamic anatomy, while the auditory display encoded physiological cardiac states using a multilayer physical membrane model.   A phantom study was conducted with twelve cardiologists performing needle insertions under visual-only and multisensory feedback. The multisensory method significantly improved navigation safety ($χ^2 = 11.30$, $p < 0.01$), reducing myocardial contact (3.64% vs. 7.27%) and increasing correct access (90.91% vs. 52.73%). Needle placement accuracy improved, with closer membrane proximity (Cliff delta = 0.19) and reduced variability ($p < 0.05$). Execution time was comparable, while time-accuracy correlations differed significantly between modalities ($p < 0.01$). NASA-TLX indicated lower cognitive load with multisensory guidance ($p < 0.01$).   These results demonstrate the feasibility of physics-driven sonification for improving spatiotemporal awareness and supporting user-centered surgical navigation. |
| 2026-04-08 | [Data Leakage in Automotive Perception: Practitioners' Insights](http://arxiv.org/abs/2604.06899v1) | Md Abu Ahammed Babu, Sushant Kumar Pandey et al. | Data leakage is the inadvertent transfer of information between training and evaluation datasets that poses a subtle, yet critical, risk to the reliability of machine learning (ML) models in safety-critical systems such as automotive perception. While leakage is widely recognized in research, little is known about how industrial practitioners actually perceive and manage it in practice. This study investigates practitioners' knowledge, experiences, and mitigation strategies around data leakage through ten semi-structured interviews with system design, development, and verification engineers working on automotive perception functions development. Using reflexive thematic analysis, we identify that knowledge of data leakage is widespread and fragmented along role boundaries: ML engineers conceptualize it as a data-splitting or validation issue, whereas design and verification roles interpret it in terms of representativeness and scenario coverage. Detection commonly arises through generic considerations and observed performance anomalies rather than implying specific tools. However, data leakage prevention is more commonly practiced, which depends mostly on experience and knowledge sharing. These findings suggest that leakage control is a socio-technical coordination problem distributed across roles and workflows. We discuss implications for ML reliability engineering, highlighting the need for shared definitions, traceable data practices, and continuous cross-role communication to institutionalize data leakage awareness within automotive ML development. |
| 2026-04-08 | [Are LLMs Ready for Computer Science Education? A Cross-Domain, Cross-Lingual and Cognitive-Level Evaluation Using Professional Certification Exams](http://arxiv.org/abs/2604.06898v1) | Chen Gao, Chi Liu et al. | Large language models (LLMs) are increasingly applied in computer science education for tasks such as tutoring, content generation, and code assessment. However, systematic evaluations aligned with formal curricula and certification standards remain limited. This study benchmarked four recent models, including GPT-5, DeepSeek-R1, Qwen-Plus, and Llama-3.3-70B-Instruct, using a dataset of 1,068 questions derived from six certification exams covering networking, office applications, and Java programming.   We evaluated performance across language (Chinese vs. English), cognitive levels based on Bloom's Taxonomy, domain knowledge, confidence-accuracy alignment, and robustness to input masking. Results showed that GPT-5 performed best on English-language certifications, while Qwen-Plus performed better in Chinese contexts. DeepSeek-R1 achieved the most balanced cross-lingual performance, whereas Llama-3.3 showed clear limitations in higher-order reasoning and robustness. All models performed worse on more complex tasks.   These findings provide empirical support for the integration of LLMs into computer science education and offer practical implications for curriculum design and assessment. |
| 2026-04-08 | [Towards Multiparty Session Types for Highly-Concurrent and Fault-Tolerant Web Applications](http://arxiv.org/abs/2604.06878v1) | Richard Casetta, Nils Gesbert et al. | Modern web applications combine persistent state updates, concurrent interactions, and unreliable communication with external services. Failures such as timeouts can occur after partial state changes, producing temporary inconsistencies whose resolution depends on liveness properties that are often not verified in practice. Although formal methods offer rigorous guarantees for reasoning about complex software, they remain rarely adopted in enterprise settings due to their perceived complexity and lack of practical automation. Multiparty Session Types (MPST) offer strong guarantees for communication safety, yet they do not account for the interplay between state evolution, dynamic workflow structure, and failure behaviour that are essential for reasoning about the correctness of real web applications.   This paper introduces a global-type framework that equips MPST with explicit failure semantics and dynamic participation. We define the syntax and operational semantics of these enriched global types and establish core properties, including coherence preservation. This foundation enables formal reasoning about communications in web applications where failures may occur, and lays the groundwork for future stateful extensions and automated verification of liveness properties. |
| 2026-04-08 | [Generating Local Shields for Decentralised Partially Observable Markov Decision Processes](http://arxiv.org/abs/2604.06873v1) | Haoran Yang, Nobuko Yoshida | Multi-agent systems under partial observation often struggle to maintain safety because each agent's locally chosen action does not, in general, determine the resulting joint action. Shielding addresses this by filtering actions based on the current state, but most existing techniques either assume access to a shared centralised global state or employ memoryless local filters that cannot consider interaction history.   We introduce a shield process algebra with guarded choice and recursion for specifying safe global behaviour in communication-free Dec-POMDP settings. From a shield process, we compile a process automaton, then a global Mealy machine as a safe joint-action filter, and finally project it to local Mealy machines whose states are belief-style subsets of the global Mealy machine states consistent with each agent's observations, and which output per-agent safe action sets.   We implement the pipeline in Rust and integrate PRISM, the Probabilistic Symbolic Model Checker, to compute best- and worst-case safety probabilities independently of the agents' policies. A multi-agent path-finding case study demonstrates how different shield processes substantially reduce collisions compared to the unshielded baseline while exhibiting varying levels of expressiveness and conservatism. |
| 2026-04-08 | [Compressing Correct-by-Design Synthesis for Stochastic Homogeneous Multi-Agent Systems with Counting LTL](http://arxiv.org/abs/2604.06868v1) | Xinyuan Qiu, Ruohan Wang et al. | Correct-by-design synthesis provides a principled framework for establishing formal safety guarantees for stochastic multi-agent systems (MAS). However, conventional approaches based on finite abstractions often incur prohibitive computational costs as the number of agents and the complexity of temporal logic specifications increase. In this work, we study homogeneous stochastic MAS under counting linear temporal logic (cLTL) specifications, and show that the corresponding satisfaction probability admits a structured tensor decomposition via leveraging deterministic finite automata (DFA). Building on this structure, we develop a dual-tree-based value iteration framework that reduces redundant computation in the process of dynamic programming. Numerical results demonstrate the proposed approach's effectiveness and scalability for complex specifications and large-scale MAS. |

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



