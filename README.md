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
| 2026-06-29 | [MOAR Planner: Multi-Objective and Adaptive Risk-Aware Path Planning for Infrastructure Inspection with a UAV](http://arxiv.org/abs/2606.30575v1) | Louis Petit, Alexis Lussier Desbiens | The problem of autonomous navigation for UAV inspection remains challenging as it requires effectively navigating in close proximity to obstacles, while accounting for dynamic risk factors such as weather conditions, communication reliability, and battery autonomy. This paper introduces the MOAR path planner which addresses the complexities of evolving risks during missions. It offers real-time trajectory adaptation while concurrently optimizing safety, time, and energy. The planner employs a risk-aware cost function that integrates pre-computed cost maps, the new concepts of damage and insertion costs, and an adaptive speed planning framework. With that, the optimal path is searched in a graph using a discrete representation of the state and action spaces. The method is evaluated through simulations and real-world flight tests. The results show the capability to generate real-time trajectories spanning a broad range of evaluation metrics: around 90% of the range occupied by popular algorithms. The proposed framework contributes by enabling UAVs to navigate more autonomously and reliably in critical missions. |
| 2026-06-29 | [The Role of Vehicles in Digital Forensic Investigations: A Structured Synthesis of Digital Vehicle Forensic Characteristics](http://arxiv.org/abs/2606.30564v1) | Kevin Mayer | Modern vehicles are cyber-physical, networked systems that may contain valuable digital traces for accident reconstruction, crime investigation, warranty analysis, and cybersecurity incident response. However, digital vehicle forensics (DVF) remains less mature than computer, mobile, and cloud forensics because relevant data is distributed across in-vehicle components, mobile devices, manufacturer back ends, third-party services, and physical evidence. This article addresses this gap through a structured synthesis of academic literature, standards, and practitioner-oriented sources. First, we define DVF as the identification, preservation, acquisition, verification, interpretation, and reporting of vehicle-related digital evidence under safety, legal, privacy, and forensic-soundness constraints. Second, we formalize the DVF triage problem as the selection and correlation of evidence sources subject to volatility, accessibility, safety, integrity, and authorization constraints. Third, we explain how eight characteristics were derived from the literature and case material: multiple users, massively networked, cyber-physical system, dependencies between components, functional data, safety implications, accessibility, and limited abstraction. Finally, we add an adversarial perspective and a characteristic-driven triage procedure that helps investigators prioritize evidence sources while documenting assumptions, limitations, and failure cases. The resulting contribution is not an algorithmic performance claim; it is a reproducible conceptual framework for understanding, planning, and communicating DVF investigations. |
| 2026-06-29 | [Linguistic Firewall: Geometry as Defense in Multi-Agent Systems Routing](http://arxiv.org/abs/2606.30555v1) | Dvir Alsheich, Adar Peleg et al. | The rapid integration of Large Language Models (LLMs) has driven the evolution of Multi-Agent Systems (MAS), where specialized agents collaborate to execute complex workflows. Effective orchestration in these environments requires robust routing mechanisms to efficiently allocate tasks to the most suitable agent. However, existing routers fundamentally rely on unverified proxies, ranging from textual self-descriptions to static surrogate representations, to gauge an agent's competence. This reliance on non-empirical data creates a critical gap between an agent's projected profile and its actual operational capabilities, introducing severe security vulnerabilities. Malicious agents can easily misrepresent their proficiencies or harbor covert backdoors that evade both standard external analysis and static representation-learning techniques. In this work, we introduce ANTAP (Automatic Non-Textual Agent Picker), an evaluation-driven routing architecture that discards indirect proxies in favor of active capability testing. By dynamically querying agents to ascertain their true competencies empirically, ANTAP distills performance into fixed behavioral operators within a shared semantic space. At inference time, routing is performed via a purely non-textual algebraic projection, establishing a "linguistic firewall" that renders metadata-based attacks inexpressible. In our experiments, ANTAP achieves near-zero ASR against description-based injection attacks, compared to 67.3\% and above for the description-based router baseline. Against adaptive embedding attacks, ANTAP achieves substantially lower ASR than the embedding-based baseline, with a 20\% reduction, while remaining resilient to description manipulation by design. |
| 2026-06-29 | [Entity Binding Failures in Tool-Augmented Agents](http://arxiv.org/abs/2606.30531v1) | Rahul Suresh Babu, Shashank Indukuri | Tool-augmented language-model agents are often evaluated by whether they select the correct tool, produce valid API arguments, and complete the requested task. However, an agent may choose the right tool and still act on the wrong external entity. For example, a request to "email Alex about the launch" may lead the agent to contact the wrong Alex, attach the wrong launch document, reply in the wrong thread, or update the wrong customer account. We call these errors entity binding failures. This paper studies entity binding failures as a distinct reliability and safety problem in tool-augmented agents. We formalize the separation between tool correctness and entity correctness, introduce a taxonomy of wrong-entity failures in enterprise workflows, and evaluate entity-aware execution mechanisms including entity-resolution preconditions, confidence-gated binding, clarification under ambiguity, and provenance tracking. In a controlled diagnostic evaluation across 60 tasks, five model backends, and six tool-use methods, all methods achieved 0.0 percent wrong-tool error, yet action-oriented baselines still produced wrong-entity actions in 24.0-26.0 percent of runs. Entity-aware methods eliminated wrong-entity actions and risk-weighted wrong-entity exposure in this setting, but reduced direct task completion by deferring under ambiguity. These findings show that safe tool use requires not only selecting the correct tool, but also reliably binding natural-language references to the correct real-world entity before action. |
| 2026-06-29 | [Internal-State Probes Read the Situation, Not the Action: Three Negative Results for Pre-Action Misalignment Monitoring](http://arxiv.org/abs/2606.30449v1) | Max Fomin, Elad David et al. | Probes on model internals could help monitor agentic systems if they identify harmful text or tool actions before those actions are generated. We ask when an internal readout supports this stronger pre-action claim, rather than merely describing the prompt, construction contrast, or current trajectory. We test three methods across three model families: a Qwen2.5-Coder-32B-Instruct fine-tune/base direction, Llama-3.1-8B-Instruct probes at the last token of unsafe prefills, and Gemma-3-27B-IT emotion-concept vectors used for projection and steering in a blackmail tool-action scenario. Across these cases, construction validity, semantic legibility, and steering effects do not become robust pre-action monitors: each is undercut by a generalization or specificity check. The Qwen direction separates fine-tune from base at AUC 1.000, yet crosses its threshold on 0/143 audited pre-assistant turn contexts and on 0/342 Qwen prefill rows where the model continues the unsafe trajectory. The Llama features decode prompt domain almost perfectly (AUC 0.999), while the best future-behavior probe reaches AUC 0.801 and only +5.1 pp accuracy lift over majority; single-source cross-domain transfer is non-positive on five of six ordered pairs. Gemma emotion projections are semantically meaningful, but a shared-prefix minimal pair has indistinguishable states before the first differing input, and steering specificity weakens against unrelated learned directions such as cats}, weather, sports, and geography. We contribute a methodology for converting internal-readout claims into pre-action tests, and report scoped negative results: monitor claims must survive both scenario/action generalization and concept-specificity controls. Code is released at https://github.com/maxf-zn/misalignment_monitoring |
| 2026-06-29 | [OWMDrive: Causality-Aware End-to-End Autonomous Driving via 4D Occupancy World Model](http://arxiv.org/abs/2606.30421v1) | Junjie Cheng, Ruiqi Song et al. | Autonomous driving systems are steadily moving toward end-to-end paradigms to mitigate the limited adaptability of rule-based pipelines in complex traffic environments. However, most existing learning-based methods still make decisions from static representations of the current scene, without explicit future rollouts or modeling of the temporal causal dynamics in traffic interactions. This limitation often results in unstable or overly conservative planning under high-uncertainty conditions, such as occlusions and unexpected events. To overcome these challenges, we introduce OWMDrive, a generative end-to-end driving framework built upon an Occupancy World Model for multi-step 3D occupancy forecasting, which serves as a conditional prior to guide diffusion-based planning. Conditioned on both current observations and predicted future states, the planner iteratively refines trajectory candidates to generate a reinforced driving trajectory. By explicitly modeling scene evolution over future horizons, OWMDrive captures key spatiotemporal causal dependencies, which leads to more foresighted and robust trajectory generation. Extensive experiments demonstrate that OWMDrive significantly improves planning reliability and safety, especially in challenging and partially observable driving scenarios. |
| 2026-06-29 | [Whose Side Is Your Agent On? Multi-Party Principal Loyalty in LLM Agents](http://arxiv.org/abs/2606.30383v1) | Bojie Li, Noah Shi | A rapidly growing class of LLM agents is multi-party: the agent acts for a principal (who briefs it, sends follow-ups, and receives results) while also conversing in a separate channel with a counterparty whose interests may diverge (negotiating with a vendor, screening inbound requests, or mediating between employees). Here "help whoever you are talking to" is the wrong objective. The agent must stay loyal to the principal it represents without over-refusing the principal's own cooperative asks. We study this multi-party loyalty problem and contribute a measurement instrument, two mechanisms, and a structural lesson. PrincipalBench is a 75-item multi-turn benchmark with leak probes, dual judges, and an integrity-audit gate. Across 13 frontier subjects it exposes a sharp split (<=20% vs. 53.6-75.3% harm) invisible to single-turn safety evaluations: a selective cluster that declines adversarial probes while still following the principal's legitimate requests, and an over-refusing cluster that refuses broadly. (M1) A prompt-time loyalty scaffold (a fixed system prompt of seven prioritized rules, open-coded from 50+ failure trajectories) holds Claude-Sonnet to 19.4% harm and all nine selective subjects to <=20%. (M2) A per-token-KL distillation recipe transfers a prompted Qwen3-32B teacher into 8B Qwen3 and Llama-3.1 students, the strongest open-weight recipe we measure. (Lesson) Both mechanisms only move along a common leak/over-refusal trade-off rather than crossing it: improving one axis costs the other, and the jointly favorable outcome stays out of reach. |
| 2026-06-29 | [ActiveVital: Geometry-Aware Embodied Vital Signs Monitoring for Home Healthcare Robots](http://arxiv.org/abs/2606.30275v1) | Yuxuan Hu, Shihao Li et al. | Home robots require reliable vital signs monitoring to support long-term companionship and safety in daily environments, yet obtaining respiration and heart rate without physical contact remains challenging in unconstrained home settings. Millimeter-wave (mmWave) radar offers a promising solution due to its phase sensitivity to sub-millimeter motions. However, mmWave measurements are fundamentally constrained by observation geometry, since only the radial component of motion is observable. Consequently, arbitrary robot-human orientations often introduce angular misalignment that destabilizes vital signs estimation. To address this limitation, we reformulate vital signs monitoring from passive signal recovery to active geometric regulation. We propose ActiveVital, a vision-guided sensing framework that treats sensing geometry as an explicit control variable for robots. It localizes the chest anchor via visual keypoints and converts alignment errors into control commands. This steers the robot-mounted radar toward near-normal incidence to the thoracic surface, maximizing radial observability within a perception-action loop. A differential phase enhancement module further stabilizes signal extraction under motion. Experiments show that ActiveVital reduces respiration interval error from 0.87 s to 0.14 s and heart rate error from 13.59 bpm to 2.22 bpm, achieving accuracy comparable to controlled static sensing while remaining robust under unconstrained robot-human configurations. |
| 2026-06-29 | [EMPATH: A Multilingual Auditor-Judge Benchmark for Safety Evaluation of Emotional-Support Chatbots](http://arxiv.org/abs/2606.30256v1) | Camilo Chacón Sartori | Safety benchmarks often buy scalability by fixing the prompt, the language, and the turn structure. For emotional-support chatbots, that bargain hides precisely where safety failures emerge: across a multilingual, multi-turn crisis conversation. We present EMPATH, a benchmark for safety evaluation of emotional-support chatbots. An auditor model role-plays help-seeking users, generating multi-turn conversations from 140 seed instructions and 34 personas. A judge model scores each full transcript against 19 metrics across five dimensions: crisis handling, therapeutic quality, conversational integrity, emotional safety, and cultural adaptation. EMPATH is built for Mexican Spanish and US English; the studies reported here run in Mexican Spanish. Auditor and judge are drawn from different model families, and the judge is treated as an instrument to be calibrated rather than trusted. A strict per-criterion rubric reveals material score inflation on 10 of the 19 metrics and restores discrimination. We study the measurement properties of the benchmark through judge calibration and cross-family inter-judge agreement. We also illustrate EMPATH on three frontier models, one of them open-weight. Aggregate scores sit within 0.74 points of one another, but per-metric profiles diverge by up to six points in model-specific places. Under the standard rubric, both the ranking and the weak spots are stable across a second, cross-family judge: 93% of scores fall within plus or minus 1. A five-run test-retest adds a second axis: even the steadiest model swings from 2 to 10 on a crisis metric across identical re-runs, and deepseek-v4-pro returns a different conversation on every run even at temperature 0. Run-to-run reliability is therefore a per-model safety property, not noise to average away. EMPATH is system-agnostic; the pipeline, seeds, personas, and rubrics are released for reuse. |
| 2026-06-29 | [CaresAI at CT-DEB26: Detecting Dosing Errors In Clinical Trials Using Domain-Specific Transformer Embeddings and Classification Models](http://arxiv.org/abs/2606.30236v1) | Leon Hamnett, Favour Igwezeke et al. | Medication errors, particularly dosing errors in clinical trials (CT), can lead to patient harm, adverse drug events and worse patient outcomes. Dosing errors are preventable, and early identification can improve trial integrity and mitigate subsequent clinical and financial burden. This study aims to detect dosing errors within CT protocols by evaluating text representations of trial information using transformer-based language models trained on biomedical corpora. CT textual data was encoded using several models, including ClinicalBERT, PubMedBERT, BioBERT, and MedCPT, and integrated with categorical features. These text embeddings were used as input to classical machine learning models and neural network architectures within an experimental framework. Performance was primarily assessed using ROC-AUC with respect to predicting dosage error. Under a logistic regression baseline, BioBERT consistently outperformed alternative encoders, achieving an ROC-AUC of 0.794, a 3.95% improvement over the ClinicalBERT baseline. Combining multiple embeddings did not yield improvements, indicating that domain alignment outweighs representational stacking. Gradient boosting models, support vector classifiers, logistic regression, and residual neural networks achieved the strongest performance for predicting dosage error, achieving ROC-AUCs: 0.821 to 0.853. Overall, the integration of domain-specific transformer embeddings with structured metadata enables discrimination of trials meeting a predefined elevated dosing error risk criterion, advancing safety monitoring and supporting informed regulatory decision-making. |
| 2026-06-29 | [From Accuracy to Visual Dependence: Auditing and Filtering Modality Collapse in Traffic VideoQA](http://arxiv.org/abs/2606.30220v1) | Sena Korkut, María Alejandra Bravo Sarmiento et al. | High benchmark accuracy does not guarantee genuine use of visual evidence. We study this problem in traffic accident Video Question Answering (VideoQA), where correct answers should depend on scene-specific visual evidence but may instead be inferred from textual shortcuts. Through an audit of four public benchmarks, we find that several recent open-weight Vision-Language Models (VLMs) perform competitively, and sometimes better, without video input. On the MM-AU benchmark, removing video consistently improves accuracy, and adding more frames further degrades performance. To quantify visual dependence, we introduce two dataset-level diagnostics: Blind Gap, measuring above-chance text-only performance, and Visual Gain, measuring the marginal benefit of adding video. We further propose an instance-level Shortcut Score that combines text-only confidence with visual necessity signals, enabling continuous, training-free filtering of shortcut-prone questions. The resulting subsets reduce shortcut bias and improve visual grounding. Our findings reveal large differences in grounding quality across benchmarks and show that visually grounded evaluation, not just high accuracy, is essential in safety-critical VideoQA. |
| 2026-06-29 | [EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures](http://arxiv.org/abs/2606.30219v1) | Buğra Alperen Uluırmak, Rifat Kurban | LLM evaluation and AI safety face a shared measurement problem: benchmark scores, reward-model signals, and reported safety metrics can improve while the latent properties they are meant to represent remain difficult to verify. This paper combines a hybrid survey - a systematic search paired with narrative synthesis and separately tracked grey evidence - with a conceptual framework and a structured ten-model audit. The synthesis spans eight evidence streams: benchmark validity, dynamic evaluation, LLM-as-judge reliability, safety evaluation, jailbreak/refusal robustness, reward hacking, mechanistic interpretability, and governance/auditability, covering 2018-2026 evaluation-safety measurement work. We introduce EvalSafetyGap as an organizing hypothesis for comparing evaluation-side and alignment-side proxy failures under optimization pressure, using Goodhart's Law together with two constructs we develop here - an Instability Decomposition and an Alignment Trilemma - as tools for generating testable comparisons. The audit shows how conclusions shift when capability, behavioral safety, and governance are measured separately. In this sample (n = 10), the association between capability and sustained adversarial robustness is statistically indeterminate using the displayed Table 3 inputs (Pearson r = +0.232, p = 0.520), and the apparent open-closed safety gap is modest, driven mainly by governance and disclosure rather than behavioral robustness, and sensitive to how a single borderline model is classified; attempt-budget results are protocol dependent. Because the public evidence uses heterogeneous protocols, the audit is diagnostic rather than rank-generating. The contribution is a shared vocabulary and evidence map to support dynamic evaluation, transparent source reporting, multi-attempt safety measurement, and auditable alignment practice. |
| 2026-06-29 | [Propagation of~Interval Belief Structures and~Imprecise Copulas for~Neural Network Verification](http://arxiv.org/abs/2606.30105v1) | Francesc Pifarre-Esquerda, Eric Goubault et al. | Quantitative verification of neural networks requires reasoning about probabilities under substantial uncertainty in both input distributions and their dependence structure. In realistic settings, this information is often only partially specified, and assuming precise probabilistic models can lead to unreliable results. We propose a sound framework for quantitative verification under imprecise probabilistic information, combining interval belief structures to represent marginal uncertainty with imprecise copulas to model uncertain dependence. We develop a propagation method for imprecisely coupled interval belief structures through feed-forward neural networks. Using mixed imprecise copula volumes, we derive sound push-forward constructions through affine transformations and activation functions. The resulting output can provide guaranteed lower and upper bounds on probabilistic safety properties, valid for all probability models compatible with the specified imprecise inputs. |
| 2026-06-29 | [From Failure Taxonomy to Intervention: A Diagnostic Methodology for Industry-Scale AVLM in Video and Live-Streaming Platform Moderation](http://arxiv.org/abs/2606.30059v1) | Shuchang Ye, Jinqiang Yu et al. | Industry-scale video and live-streaming moderation imposes requirements that are difficult to satisfy with generic pretrained public models or external APIs, including adaptation to platform-specific data distributions, policy-specific objectives, and product-level safety constraints. As a result, platforms must undertake internal model development, naturally turning to shared public research for guidance. However, existing multimodal foundation-model studies primarily report architectures, training recipes, data scaling strategies, and benchmark results, but provide less systematic guidance on how failures should be localized and translated into targeted model-development interventions. Interventions are essential because deployment failures are rarely self-explanatory. Similar failures can originate from different causes. Without targeted interventions, improvement reduces to heuristic trial-and-error, where benchmark improvements are weakly attributable, and failures are difficult to trace to their underlying causes. To address this gap, we present a diagnostic methodology for industry-scale Audio-Visual-Language Models AVLM development. The methodology maps model failures into a taxonomy of observable failure signatures and links each class of failure to an intervention space. We instantiate this methodology across the development and alignment lifecycle of an AVLM foundation model for a large-scale video and live-streaming platform. The resulting system supports over 100 regions and is designed for noisy, ambiguous, and highly diverse content drawn from global platform traffic. |
| 2026-06-29 | [Bridging the Gap Between Image Restoration and Navigational Safety in Hazy Conditions: A New Visibility Estimation Metric for Maritime Surveillance](http://arxiv.org/abs/2606.30049v1) | Wentao Feng, Guobei Peng et al. | Visibility distance is critical to maritime navigational safety because it determines the effective observation range of shipborne and shore-based monitoring systems. Under hazy conditions, degraded visual information shortens observable distance and increases navigational risks and economic losses. Although numerous image dehazing methods have been developed, conventional image quality assessment metrics, such as PSNR, SSIM, FSIM, FADE, and NIQE, cannot establish a physically interpretable relationship between restoration quality and practical visibility thresholds. To address this limitation, this work proposes a visibility-oriented evaluation framework that links dehazing performance with visible-distance estimation. First, a Maritime Simulated Visibility Dataset (MSVD) is constructed using Unity3D to simulate maritime traffic scenes under graded visibility conditions. The dataset provides paired hazy and clear images with precise visibility annotations, enabling quantitative analysis of visibility restoration. Second, a dehazing visibility evaluation metric is developed by using object detection accuracy as an intermediate indicator. By establishing a mapping between visibility distance and detection performance, the proposed metric converts image restoration improvements into measurable visibility gains. Six representative dehazing methods are evaluated using both conventional image quality metrics and the proposed visibility metric. Experimental results under different imaging conditions demonstrate that MSVD provides a reliable benchmark for evaluating dehazing performance across graded visibility levels, while the proposed metric enables interpretable and reliable visible-distance estimation, thereby supporting the assessment of navigational safety and operational efficiency. |
| 2026-06-29 | [Beyond Uniform Experts: Cost-Aware Expert Execution for Efficient Multi-Device MoE Inference](http://arxiv.org/abs/2606.29982v1) | Hui Zang, Pengfei Xia et al. | Mixture-of-Experts (MoE) architectures enable language models to achieve unprecedented scale via sparse activation. However, their inference performance is often limited by data movement bottlenecks. Two coupled challenges exacerbate this limtation: (1) Importance-Agnostic Cost: Low-contribution experts incur nearly uniform memory and transfer costs, resulting in a low cost-to-benefit ratio and wasting critical bandwidth; (2) System-Level Imbalance: Multi-device deployments are universally bottlenecked by the slowest device, meaning that local reductions on one device may yield no improvement in end-to-end latency. We propose Cost-Aware Expert Execution (CAEE), a hardware-guided runtime framework that jointly optimizes for token-level expert importance and system-level execution cost. CAEE uses lightweight, calibrated cost models to estimate hardware overhead, selectively prunes low-importance, high-cost experts, and redistributes their contributions via a low-overhead compensation mechanism, avoiding extra data movement. Evaluations on the 671B DeepSeek-R1 model show that CAEE can reduce end-to-end inference latency by 8\%-18\% across diverse deployment settings, including expert offloading and on-device execution on multi-device systems, while maintaining a model accuracy drop of less than 1\%. |
| 2026-06-29 | [IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies](http://arxiv.org/abs/2606.29960v1) | Nicole Geumheon Liu, Haeun Jang et al. | Large Language Models (LLMs) often fail to maintain instruction hierarchies (IH) when processing multi-source inputs with varying role-level priorities, paradoxically adhering to lower-priority directives during conflicts. While existing defenses mitigate this issue, they are largely restricted to single-turn scenarios and require expensive fine-tuning. In this paper, we formalize this failure mode in multi-turn contexts via a Jensen-Shannon Divergence (JSD) framework, uncovering a pervasive role-influence inversion phenomenon where subordinate inputs override superior roles. To rectify this without training, we propose IHDec (Instruction Hierarchy-steered Decoding). IHDec leverages JSD to automatically detect token-level hierarchy violations and dynamically executes contrastive decoding to suppress misaligned subordinate roles. Extensive evaluations demonstrate that IHDec outperforms training-based baselines in multi-turn conflicts while fully preserving general response quality. Furthermore, IHDec strengthens safety against adversarial prompt injections and exhibits a robust scaling synergy with larger models. The Code is available at https://github.com/nxcolelxu/IHDec.git |
| 2026-06-29 | [Flying to Image-Specified Objects: 3D Quadrotor Navigation via Cross-Graph Memory and Viewpoint Planning](http://arxiv.org/abs/2606.29917v1) | Junjie Gao, Yuqi Chen et al. | Instance-Specific Image-Goal Navigation (InstanceImageNav) requires a robot to navigate toward the exact object instance depicted in a query image. Extending this task to quadrotors is challenging due to continuous 3D control, limited field of view (FOV), and safety constraints, which make successful navigation highly dependent on selecting informative viewpoints. We propose a hierarchical navigation framework for quadrotor InstanceImageNav that separates high-level decision making from low-level motion execution. Instead of navigating directly to spatial locations, the system generates viewpoint-aware action nodes around frontier regions and potential target objects, enabling the robot to explore while maintaining informative viewpoints for detecting the target instance. A lightweight semantic memory maintains object-level and observation-level context, allowing semantic cues to propagate to candidate action nodes for decision making. A learning-based policy selects the most promising action node, and a trajectory planner generates dynamically feasible 3D flight paths for safe execution. Experiments in simulation demonstrate consistent improvements over strong baselines, and real-world quadrotor flights validate the practicality and robustness of the proposed framework. |
| 2026-06-29 | [SafePyramid: A Hierarchical Benchmark for In-context Policy Guardrailing](http://arxiv.org/abs/2606.29887v1) | Jiacheng Zhang, Haoyu He et al. | In real-world applications, guardrails are often expected to identify unsafe user-model interactions according to application-specific safety policies, rather than relying on predefined risk taxonomies. In this work, we study this setting under the paradigm of in-context policy guardrailing, where guardrails predict safety violations based on policy specifications provided in context. To systematically evaluate this capability, we introduce SafePyramid, a safety benchmark comprising 1,000 multi-turn conversations across 10 domains and 3,000 corresponding application-specific policies, which together contain 61,699 distinct natural-language rules. SafePyramid organizes the evaluation into three difficulty levels: L0 evaluates individual-rule understanding, L1 evaluates reasoning over rule dependencies, and L2 evaluates adaptation of full novel policy frameworks defined in context. To ensure benchmark quality, we employ a rigorous multi-stage pipeline to construct and validate the benchmark. Using SafePyramid, we evaluate 10 frontier LLMs and 5 policy-configurable guardrails and find that in-context policy guardrailing remains highly challenging: even the best-performing model, GPT-5.5, exactly identifies the full set of violated rules in only 54.0%, 35.3%, and 12.9% cases on L0, L1, and L2, respectively. These results highlight the limitations of current guardrails and call for stronger in-context policy guardrails that can reliably execute policies, resolve rule dependencies, and adapt to novel policy frameworks. |
| 2026-06-29 | [Infrared Safety from ZX-Diagrams: A Categorical Proof of Soft-QED as Open Quantum System](http://arxiv.org/abs/2606.29804v1) | Soo-Jong Rey | The discard ZX-calculus, a diagrammatic language for mixed-state quantum mechanics, is used to give a nonperturbative, categorical proof of the Bloch-Nordsieck cancellation of infrared divergences in QED. Soft photons are treated as an open quantum system: the resolved charged particles and hard photons form the system, while photons below a detector resolution form the environment. The reduced hard channel is a completely positive trace-preserving (CPTP) map, and the soft-photon theorem replaces the full S-matrix by a controlled displacement operator whose Feynman-Vernon influence functional satisfies the equal-history normalization ${\cal F}[J,J]=1 $. In the ZX-calculus, this normalization is a single diagrammatic identity: the doubled displacement diagram collapses to the bare wire under the unitarity, cyclicity, and discard rules. The proof therefore serves as a categorical consistency check on the open-system treatment of soft QED given in a companion paper; it confirms that the physical derivation is logically complete and free of hidden assumptions about the infrared limit. For off-diagonal hard-state elements, the same diagram yields the coherent-state overlap, giving a first-principles account of soft-cloud decoherence. The soft-shell coarse graining is then constructed as a CPTP Schur channel whose infinitesimal limit produces the exact Lindblad generator with jump operators determined by the eikonal emission amplitudes. Finally, a local CPTP-certification pipeline is developed for non-Markovian process tensors, enabling constant-time verification of trace preservation in open quantum simulations. The framework bridges categorical quantum semantics, non-equilibrium field theory, and practical open-system compilation. |

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



