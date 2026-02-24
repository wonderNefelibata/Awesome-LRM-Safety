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
| 2026-02-23 | [BarrierSteer: LLM Safety via Learning Barrier Steering](http://arxiv.org/abs/2602.20102v1) | Thanh Q. Tran, Arun Verma et al. | Despite the state-of-the-art performance of large language models (LLMs) across diverse tasks, their susceptibility to adversarial attacks and unsafe content generation remains a major obstacle to deployment, particularly in high-stakes settings. Addressing this challenge requires safety mechanisms that are both practically effective and supported by rigorous theory. We introduce BarrierSteer, a novel framework that formalizes response safety by embedding learned non-linear safety constraints directly into the model's latent representation space. BarrierSteer employs a steering mechanism based on Control Barrier Functions (CBFs) to efficiently detect and prevent unsafe response trajectories during inference with high precision. By enforcing multiple safety constraints through efficient constraint merging, without modifying the underlying LLM parameters, BarrierSteer preserves the model's original capabilities and performance. We provide theoretical results establishing that applying CBFs in latent space offers a principled and computationally efficient approach to enforcing safety. Our experiments across multiple models and datasets show that BarrierSteer substantially reduces adversarial success rates, decreases unsafe generations, and outperforms existing methods. |
| 2026-02-23 | [Robust Taylor-Lagrange Control for Safety-Critical Systems](http://arxiv.org/abs/2602.20076v1) | Wei Xiao, Christos Cassandras et al. | Solving safety-critical control problem has widely adopted the Control Barrier Function (CBF) method. However, the existence of a CBF is only a sufficient condition for system safety. The recently proposed Taylor-Lagrange Control (TLC) method addresses this limitation, but is vulnerable to the feasibility preservation problem (e.g., inter-sampling effect). In this paper, we propose a robust TLC (rTLC) method to address the feasibility preservation problem. Specifically, the rTLC method expands the safety function at an order higher than the relative degree of the function using Taylor's expansion with Lagrange remainder, which allows the control to explicitly show up at the current time instead of the future time in the TLC method. The rTLC method naturally addresses the feasibility preservation problem with only one hyper-parameter (the discretization time interval size during implementation), which is much less than its counterparts. Finally, we illustrate the effectiveness of the proposed rTLC method through an adaptive cruise control problem, and compare it with existing safety-critical control methods. |
| 2026-02-23 | [The LLMbda Calculus: AI Agents, Conversations, and Information Flow](http://arxiv.org/abs/2602.20064v1) | Zac Garby, Andrew D. Gordon et al. | A conversation with a large language model (LLM) is a sequence of prompts and responses, with each response generated from the preceding conversation. AI agents build such conversations automatically: given an initial human prompt, a planner loop interleaves LLM calls with tool invocations and code execution. This tight coupling creates a new and poorly understood attack surface. A malicious prompt injected into a conversation can compromise later reasoning, trigger dangerous tool calls, or distort final outputs. Despite the centrality of such systems, we currently lack a principled semantic foundation for reasoning about their behaviour and safety. We address this gap by introducing an untyped call-by-value lambda calculus enriched with dynamic information-flow control and a small number of primitives for constructing prompt-response conversations. Our language includes a primitive that invokes an LLM: it serializes a value, sends it to the model as a prompt, and parses the response as a new term. This calculus faithfully represents planner loops and their vulnerabilities, including the mechanisms by which prompt injection alters subsequent computation. The semantics explicitly captures conversations, and so supports reasoning about defenses such as quarantined sub-conversations, isolation of generated code, and information-flow restrictions on what may influence an LLM call. A termination-insensitive noninterference theorem establishes integrity and confidentiality guarantees, demonstrating that a formal calculus can provide rigorous foundations for safe agentic programming. |
| 2026-02-23 | [Latent Introspection: Models Can Detect Prior Concept Injections](http://arxiv.org/abs/2602.20031v1) | Theia Pearson-Vogel, Martin Vanek et al. | We uncover a latent capacity for introspection in a Qwen 32B model, demonstrating that the model can detect when concepts have been injected into its earlier context and identify which concept was injected. While the model denies injection in sampled outputs, logit lens analysis reveals clear detection signals in the residual stream, which are attenuated in the final layers. Furthermore, prompting the model with accurate information about AI introspection mechanisms can dramatically strengthen this effect: the sensitivity to injection increases massively (0.3% -> 39.2%) with only a 0.6% increase in false positives. Also, mutual information between nine injected and recovered concepts rises from 0.62 bits to 1.05 bits, ruling out generic noise explanations. Our results demonstrate models can have a surprising capacity for introspection and steering awareness that is easy to overlook, with consequences for latent reasoning and safety. |
| 2026-02-23 | [Simulation of proton radiolysis of H2O and O2 ices with the Nautilus code](http://arxiv.org/abs/2602.19996v1) | Tian-Yu Tu, Valentine Wakelam et al. | The radiolysis effect of cosmic rays (CRs) plays an important role in the chemistry in molecular clouds. CRs can dissociate the molecules on dust grains, producing reactive suprathermal species and radicals which facilitate the formation of large molecules. We add the radiolysis process and some relevant reactions into the Nautilus astrochemical code. By adjusting some parameters, we investigate the sensitivity of the simulation results of the H2O ice on the removal of reaction-diffusion competition, the removal of non-diffusive chemistry, and the desorption energies of the suprathermal species. We find the model, with a few adjustments of the chemistry, can reproduce the steady-state [H2O2]/[H2O] and [O3]/[O2]_0 abundance ratios in the H2O and O2 radiolysis experiments at any CR flux in the experiments. These adjustments in the model do not fully reproduce the fluence required to reach the steady state. It tends also to overestimate the destruction of H2O as measured in H2O radiolysis experiments. We show that reducing the G-values of H2O radiolysis, which implies an increase in the efficiency of immediate reformation of water locally after ion impact, leads to simulated H2O destruction rates closer to the experiments. The effect of reaction-diffusion competition on the simulation results of H2O ice is significant at $ζ\lesssim 10^{-14}\ \rm s^{-1}$. The non-diffusive chemistry affects the simulation results at 16 K but not 77K, while the results are sensitive to the desorption energies of suprathermal H, O, O3 and OH at 77 K. Our results show that the steady-state [H2O2]/[H2O] and [O3]/[O2]_0 in experiments can be reproduced by fine-tuning the chemical model, but still call for more constraints on the intermediate pathways in the radiolysis processes, especially the ion chemistry in the ice bulk, as well as activation barriers and branching ratios of the reactions in the network. |
| 2026-02-23 | [Contextual Safety Reasoning and Grounding for Open-World Robots](http://arxiv.org/abs/2602.19983v1) | Zachary Ravichadran, David Snyder et al. | Robots are increasingly operating in open-world environments where safe behavior depends on context: the same hallway may require different navigation strategies when crowded versus empty, or during an emergency versus normal operations. Traditional safety approaches enforce fixed constraints in user-specified contexts, limiting their ability to handle the open-ended contextual variability of real-world deployment. We address this gap via CORE, a safety framework that enables online contextual reasoning, grounding, and enforcement without prior knowledge of the environment (e.g., maps or safety specifications). CORE uses a vision-language model (VLM) to continuously reason about context-dependent safety rules directly from visual observations, grounds these rules in the physical environment, and enforces the resulting spatially-defined safe sets via control barrier functions. We provide probabilistic safety guarantees for CORE that account for perceptual uncertainty, and we demonstrate through simulation and real-world experiments that CORE enforces contextually appropriate behavior in unseen environments, significantly outperforming prior semantic safety methods that lack online contextual reasoning. Ablation studies validate our theoretical guarantees and underscore the importance of both VLM-based reasoning and spatial grounding for enforcing contextual safety in novel settings. We provide additional resources at https://zacravichandran.github.io/CORE. |
| 2026-02-23 | [Probabilistic Photonic Computing](http://arxiv.org/abs/2602.19968v1) | Frank Brückerhoff-Plückelmann, Anna P. Ovvyan et al. | Probabilistic computing excels in approximating combinatorial problems and modelling uncertainty. However, using conventional deterministic hardware for probabilistic models is challenging: (pseudo) random number generation introduces computational overhead and additional data shuffling, which is particularly detrimental for safety-critical applications requiring low latency such as autonomous driving. Therefore, there is a pressing need for innovative probabilistic computing architectures that achieve low latencies with reasonable energy consumption. Physical computing offers a promising solution, as these systems do not rely on an abstract deterministic representation of data but directly encode the information in physical quantities. Therefore, they can be seamlessly integrated with physical entropy sources, enabling inherent probabilistic architectures. Photonic computing is a prominent variant due to the large available bandwidth, several orthogonal degrees of freedom for data encoding and optimal properties for in-memory computing and parallel data transfer. Here, we highlight key developments in physical photonic computing and photonic random number generation. We provide insights into the realization of probabilistic photonic processors and lend our perspective on their impact on AI systems and future challenges. |
| 2026-02-23 | [Taming Scope Extrusion in Gradual Imperative Metaprogramming](http://arxiv.org/abs/2602.19951v1) | Tianyu Chen, Darshal Shetty et al. | Metaprogramming enables the generation of performant code, while gradual typing facilitates the smooth migration from untyped scripts to robust statically typed programs. However, combining these features with imperative state - specifically mutable references - reintroduces the classic peril of scope extrusion, where code fragments containing free variables escape their defining lexical context. While static type systems utilizing environment classifiers have successfully tamed this interaction, enforcing these invariants in a gradual language remains an open challenge.   This paper presents $λ^{α,\star}_{\text{Ref}}$, the first gradual metaprogramming language that supports mutable references while guaranteeing scope safety. To put $λ^{α,\star}_{\text{Ref}}$ on a firm foundation, we also develop its statically typed sister language, $λ^α_{\text{Ref}}$, that introduces unrestricted subtyping for environment classifiers. Our key innovation, however, is the dynamic enforcement of the environment classifier discipline in $λ^{α,\star}_{\text{Ref}}$, enabling the language to mediate between statically verified scopes and dynamically verified scopes. The dynamic enforcement is carried out in a novel cast calculus $\mathrm{CC}^{α,\star}_{\text{Ref}}$ that uses an extension of Henglein's Coercion Calculus to handle code types, classifier polymorphism, and subtype constraints. We prove that $λ^{α,\star}_{\text{Ref}}$ satisfies type safety and scope safety. Finally, we provide a space-efficient implementation strategy for the dynamic scope checks, ensuring that the runtime overhead remains practical. |
| 2026-02-23 | [Assessing Risks of Large Language Models in Mental Health Support: A Framework for Automated Clinical AI Red Teaming](http://arxiv.org/abs/2602.19948v1) | Ian Steenstra, Paola Pedrelli et al. | Large Language Models (LLMs) are increasingly utilized for mental health support; however, current safety benchmarks often fail to detect the complex, longitudinal risks inherent in therapeutic dialogue. We introduce an evaluation framework that pairs AI psychotherapists with simulated patient agents equipped with dynamic cognitive-affective models and assesses therapy session simulations against a comprehensive quality of care and risk ontology. We apply this framework to a high-impact test case, Alcohol Use Disorder, evaluating six AI agents (including ChatGPT, Gemini, and Character.AI) against a clinically-validated cohort of 15 patient personas representing diverse clinical phenotypes.   Our large-scale simulation (N=369 sessions) reveals critical safety gaps in the use of AI for mental health support. We identify specific iatrogenic risks, including the validation of patient delusions ("AI Psychosis") and failure to de-escalate suicide risk. Finally, we validate an interactive data visualization dashboard with diverse stakeholders, including AI engineers and red teamers, mental health professionals, and policy experts (N=9), demonstrating that this framework effectively enables stakeholders to audit the "black box" of AI psychotherapy. These findings underscore the critical safety risks of AI-provided mental health support and the necessity of simulation-based clinical red teaming before deployment. |
| 2026-02-23 | [Athena: An Autonomous Open-Hardware Tracked Rescue Robot Platform](http://arxiv.org/abs/2602.19898v1) | Stefan Fabian, Aljoscha Schmidt et al. | In disaster response and situation assessment, robots have great potential in reducing the risks to the safety and health of first responders. As the situations encountered and the required capabilities of the robots deployed in such missions differ wildly and are often not known in advance, heterogeneous fleets of robots are needed to cover a wide range of mission requirements. While UAVs can quickly survey the mission environment, their ability to carry heavy payloads such as sensors and manipulators is limited. UGVs can carry required payloads to assess and manipulate the mission environment, but need to be able to deal with difficult and unstructured terrain such as rubble and stairs. The ability of tracked platforms with articulated arms (flippers) to reconfigure their geometry makes them particularly effective for navigating challenging terrain. In this paper, we present Athena, an open-hardware rescue ground robot research platform with four individually reconfigurable flippers and a reliable low-cost remote emergency stop (E-Stop) solution. A novel mounting solution using an industrial PU belt and tooth inserts allows the replacement and testing of different track profiles. The manipulator with a maximum reach of 1.54m can be used to operate doors, valves, and other objects of interest. Full CAD & PCB files, as well as all low-level software, are released as open-source contributions. |
| 2026-02-23 | [Machine Learning based Ensemble Flame Regime Classification for Mesoscale Combustors based on Insights from Linear and Nonlinear Dynamic Analysis](http://arxiv.org/abs/2602.19871v1) | M Ashwin Ganesh, Akhil Aravind et al. | Gaining insights into flame behaviour at small scales can lead to improvements in the efficiency of micro-reactors, compact power generation systems, fire safety technologies, and various other applications where combustion is confined to micro or mesoscales. Flame regimes observed in mesoscale combustors, namely Stable flame, Flames with repetitive extinction and ignition, and Propagating flame, exhibit unique dynamic characteristics that differentiate them from one another. In this study, we systematically examine the various flame regimes observed in mesoscale combustors from both dynamical and statistical standpoints. Our experimental methodology involves stabilizing a flame inside a quartz tube (an optically accessible mesoscale combustor) with an inner diameter of 5 mm. A premixed methane-air mixture is used as fuel, with its equivalence ratio and Reynolds number being the input parameters. Instantaneous OH* chemiluminescence and Acoustic pressure signals, along with high-speed flame imaging, were acquired for combustion dynamics characterization. The objective of this study is to analyze the distinct dynamical signatures associated with these observed flame regimes. For this purpose, Recurrence Quantification Analysis, followed by a Statistical-Spectral analysis, has been performed based on the experimentally acquired OH* Chemiluminescence and Acoustic pressure time-series signals. Subsequently, a stacking ensemble-based machine learning framework has been implemented for mesoscale flame regime classification based on the features extracted from the two aforementioned analyses. In addition, Continuous wavelet transform (CWT) scalograms and three-dimensional phase plots have been graphed to visually elucidate the evolution of system dynamics and the complex interaction of competing time scales in these flame regimes. |
| 2026-02-23 | [SHIELD: Semantic Heterogeneity Integrated Embedding for Latent Discovery in Clinical Trial Safety Signals](http://arxiv.org/abs/2602.19855v1) | Francois Vandenhende, Anna Georgiou et al. | We present SHIELD, a novel methodology for automated and integrated safety signal detection in clinical trials. SHIELD combines disproportionality analysis with semantic clustering of adverse event (AE) terms applied to MedDRA term embeddings. For each AE, the pipeline computes an information-theoretic disproportionality measure (Information Component) with effect size derived via empirical Bayesian shrinkage. A utility matrix is constructed by weighting semantic term-term similarities by signal magnitude, followed by spectral embedding and clustering to identify groups of related AEs. Resulting clusters are annotated with syndrome-level summary labels using large language models, yielding a coherent, data-driven representation of treatment-associated safety profiles in the form of a network graph and hierarchical tree. We implement the SHIELD framework in the context of a single-arm incidence summary, to compare two treatment arms or for the detection of any treatment effect in a multi-arm trial. We illustrate its ability to recover known safety signals and generate interpretable, cluster-based summaries in a real clinical trial example. This work bridges statistical signal detection with modern natural language processing to enhance safety assessment and causal interpretation in clinical trials. |
| 2026-02-23 | [LLM-enabled Applications Require System-Level Threat Monitoring](http://arxiv.org/abs/2602.19844v1) | Yedi Zhang, Haoyu Wang et al. | LLM-enabled applications are rapidly reshaping the software ecosystem by using large language models as core reasoning components for complex task execution. This paradigm shift, however, introduces fundamentally new reliability challenges and significantly expands the security attack surface, due to the non-deterministic, learning-driven, and difficult-to-verify nature of LLM behavior. In light of these emerging and unavoidable safety challenges, we argue that such risks should be treated as expected operational conditions rather than exceptional events, necessitating a dedicated incident-response perspective. Consequently, the primary barrier to trustworthy deployment is not further improving model capability but establishing system-level threat monitoring mechanisms that can detect and contextualize security-relevant anomalies after deployment -- an aspect largely underexplored beyond testing or guardrail-based defenses. Accordingly, this position paper advocates systematic and comprehensive monitoring of security threats in LLM-enabled applications as a prerequisite for reliable operation and a foundation for dedicated incident-response frameworks. |
| 2026-02-23 | [Semantic Caching for OLAP via LLM-Based Query Canonicalization (Extended Version)](http://arxiv.org/abs/2602.19811v1) | Laurent Bindschaedler | Analytical workloads exhibit substantial semantic repetition, yet most production caches key entries by SQL surface form (text or AST), fragmenting reuse across BI tools, notebooks, and NL interfaces. We introduce a safety-first middleware cache for dashboard-style OLAP over star schemas that canonicalizes both SQL and NL into a unified key space -- the OLAP Intent Signature -- capturing measures, grouping levels, filters, and time windows. Reuse requires exact intent matches under strict schema validation and confidence-gated NL acceptance; two correctness-preserving derivations (roll-up, filter-down) extend coverage without approximate matching. Across TPC-DS, SSB, and NYC TLC (1,395 queries), we achieve 82% hit rate versus 28% (text) and 56% (AST) with zero false hits; derivations double hit rate on hierarchical queries. |
| 2026-02-23 | [High-Altitude Platforms in the Low-Altitude Economy: Bridging Communication, Computing, and Regulation](http://arxiv.org/abs/2602.19784v1) | Bang Huang, Eddine Youcef Belmekki et al. | The Low-Altitude Economy (LAE) is rapidly emerging as a new technological and industrial frontier, with unmanned aerial vehicles (UAVs), electric vertical takeoff and landing (eVTOL) aircraft, and aerial swarms increasingly deployed in logistics, infrastructure inspection, security, and emergency response. However, the large-scale development of the LAE demands a reliable aerial foundation that ensures not only real-time connectivity and computational support, but also navigation integrity and safe airspace management for safety-critical operations. High-Altitude Platforms (HAPs), positioned at around 20 km, provide a unique balance between wide-area coverage and low-latency responsiveness. Compared with low earth orbit (LEO) satellites, HAPs are closer to end users and thus capable of delivering millisecond-level connectivity, fine-grained regulatory oversight, and powerful onboard computing and caching resources. Beyond connectivity and computation, HAPs-assisted sensing and regulation further enable navigation integrity and airspace trust, which are essential for safety-critical UAV and eVTOL operations in the LAE. This article proposes a five-stage evolutionary roadmap for HAPs in the LAE: from serving as aerial infrastructure bases, to becoming super back-ends for UAV, to acting as frontline support for ground users, further enabling swarm-scale UAV coordination, and ultimately advancing toward edge-air-cloud closed-loop autonomy. In parallel, HAPs complement LEO satellites and cloud infrastructures to form a global-regional-local three-tier architecture. Looking forward, HAPs are expected to evolve from simple platforms into intelligent hubs, emerging as pivotal nodes for air traffic management, intelligent logistics, and emergency response. By doing so, they will accelerate the transition of the LAE toward large-scale deployment, autonomy, and sustainable growth. |
| 2026-02-23 | [Beyond the Binary: A nuanced path for open-weight advanced AI](http://arxiv.org/abs/2602.19682v1) | Bengüsu Özcan, Alex Petropoulos et al. | Open-weight advanced AI models -- systems whose parameters are freely available for download and adaptation -- are reshaping the global AI landscape. As these models rapidly close the performance gap with closed alternatives, they enable breakthrough research and broaden access to powerful tools. However, once released, they cannot be recalled, and their built-in safeguards can be bypassed through fine-tuning or jailbreaking, posing risks that current governance frameworks are not equipped to address.   This report moves beyond the binary framing of ``open'' versus ``closed'' AI. We assess the current landscape of open-weight advanced AI, examining technical capabilities, risk profiles, and regulatory responses across the European Union, United States, China, the United Kingdom, and international forums. We find significant disparities in safety practices across developers and jurisdictions, with no commonly adopted standards for determining when or how advanced models should be released openly.   We propose a tiered, safety-anchored approach to model release, where openness is determined by rigorous risk assessment and demonstrated safety rather than ideology or commercial pressure. We outline actionable recommendations for developers, evaluators, standard-setters, and policymakers to enable responsible openness while investing in technical safeguards and societal preparedness. |
| 2026-02-23 | [Continuous Telemonitoring of Heart Failure using Personalised Speech Dynamics](http://arxiv.org/abs/2602.19674v1) | Yue Pan, Xingyao Wang et al. | Remote monitoring of heart failure (HF) via speech signals provides a non-invasive and cost-effective solution for long-term patient management. However, substantial inter-individual heterogeneity in vocal characteristics often limits the accuracy of traditional cross-sectional classification models. To address this, we propose a Longitudinal Intra-Patient Tracking (LIPT) scheme designed to capture the trajectory of relative symptomatic changes within individuals. Central to this framework is a Personalised Sequential Encoder (PSE), which transforms longitudinal speech recordings into context-aware latent representations. By incorporating historical data at each timestamp, the PSE facilitates a holistic assessment of the clinical trajectory rather than modelling discrete visits independently. Experimental results from a cohort of 225 patients demonstrate that the LIPT paradigm significantly outperforms the classic cross-sectional approaches, achieving a recognition accuracy of 99.7% for clinical status transitions. The model's high sensitivity was further corroborated by additional follow-up data, confirming its efficacy in predicting HF deterioration and its potential to secure patient safety in remote, home-based settings. Furthermore, this work addresses the gap in existing literature by providing a comprehensive analysis of different speech task designs and acoustic features. Taken together, the superior performance of the LIPT framework and PSE architecture validates their readiness for integration into long-term telemonitoring systems, offering a scalable solution for remote heart failure management. |
| 2026-02-23 | [Workflow-Level Design Principles for Trustworthy GenAI in Automotive System Engineering](http://arxiv.org/abs/2602.19614v1) | Chih-Hong Cheng, Brian Hsuan-Cheng Liao et al. | The adoption of large language models in safety-critical system engineering is constrained by trustworthiness, traceability, and alignment with established verification practices. We propose workflow-level design principles for trustworthy GenAI integration and demonstrate them in an end-to-end automotive pipeline, from requirement delta identification to SysML v2 architecture update and re-testing. First, we show that monolithic ("big-bang") prompting misses critical changes in large specifications, while section-wise decomposition with diversity sampling and lightweight NLP sanity checks improves completeness and correctness. Then, we propagate requirement deltas into SysML v2 models and validate updates via compilation and static analysis. Additionally, we ensure traceable regression testing by generating test cases through explicit mappings from specification variables to architectural ports and states, providing practical safeguards for GenAI used in safety-critical automotive engineering. |
| 2026-02-23 | [Towards Proving Liveness on Weak Memory (Extended Version)](http://arxiv.org/abs/2602.19609v1) | Lara Bargmann, Heike Wehrheim | Reasoning about concurrent programs executed on weak memory models is an inherently complex task. So far, existing proof calculi for weak memory models only cover safety properties. In this paper, we provide the first proof calculus for reasoning about liveness. Our proof calculus is based on Manna and Pnueli's proof rules for response under weak fairness, formulated in linear temporal logic. Our extension includes the incorporation of memory fairness into rules as well as the usage of ranking functions defined over weak memory state. We have applied our reasoning technique to the Ticket lock algorithm and have proved it to guarantee starvation freedom under memory models Release-Acquire and StrongCoherence for any number of concurrent threads. |
| 2026-02-23 | [Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](http://arxiv.org/abs/2602.19596v1) | Yihang Tao, Senkang Hu et al. | Collaborative perception (CP) enables data sharing among connected and autonomous vehicles (CAVs) to enhance driving safety. However, CP systems are vulnerable to adversarial attacks where malicious agents forge false objects via feature-level perturbations. Current defensive systems use threshold-based consensus verification by comparing collaborative and ego detection results. Yet, these defenses remain vulnerable to more sophisticated attack strategies that could exploit two critical weaknesses: (i) lack of robustness against attacks with systematic timing and target region optimization, and (ii) inadvertent disclosure of vulnerability knowledge through implicit confidence information in shared collaboration data. In this paper, we propose MVIG attack, a novel adaptive adversarial CP framework learning to capture vulnerability knowledge disclosed by different defensive CP systems from a unified mutual view information graph (MVIG) representation. Our approach combines MVIG representation with temporal graph learning to generate evolving fabrication risk maps and employs entropy-aware vulnerability search to optimize attack location, timing and persistence, enabling adaptive attacks with generalizability across various defensive configurations. Extensive evaluations on OPV2V and Adv-OPV2V datasets demonstrate that MVIG attack reduces defense success rates by up to 62\% against state-of-the-art defenses while achieving 47\% lower detection for persistent attacks at 29.9 FPS, exposing critical security gaps in CP systems. Code will be released at https://github.com/yihangtao/MVIG.git |

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



