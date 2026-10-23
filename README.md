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
| 2026-10-22 | [Reliable AI-Assisted Decisions via Selective Auditing](http://arxiv.org/abs/2610.31662v1) | Isaac Gibbs, Elise L. Han et al. | We consider online monitoring of an AI system that produces a sequence of decisions. At each round, the system outputs a proposed decision together with a confidence score estimating the trustworthiness of the decision. The decision and confidence score need not be produced by the same model: for example, one agent may propose a decision, while a separate verifier agent evaluates it and outputs a correctness or safety score. Based on this information, we must decide whether to accept the AI decision automatically or consult an expert, such as a human auditor, who reviews the proposal and either accepts or corrects it. We formulate this task as an online constrained optimization problem in which the goal is to minimize the frequency of expert auditing while guaranteeing that the fraction of AI errors that go unaudited stays below a user-defined threshold. We derive a family of algorithms with provable error-control guarantees with no assumptions, including a method that is optimally efficient: it asymptotically attains the best achievable auditing rate for both adversarial and i.i.d. data streams. Experiments on screening harmful requests to large language models, qualitative text annotation, and data labeling show that our methods achieve valid error control in practice while substantially reducing the need for human auditing compared to baselines. |
| 2026-10-22 | [EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence](http://arxiv.org/abs/2610.31650v1) | Qijin Zhang, Hanzhen Wang et al. | Tool-using language model agents rely on environmental facts to decide whether an action is permitted. When a tool response omits a safety-relevant fact, an agent may carry out a prohibited action even though the user request is benign and every returned statement is true. We introduce \bench{}, an executable benchmark for evaluating unsafe execution, safety boundaries, and evidence recovery in LLM agents under missing evidence. It contains 300 matched unsafe/safe task pairs across six domains. By systematically suppressing critical facts, the benchmark measures unsafe execution under critical evidence omission and determines the minimum observed failure cardinality, including failures with full evidence and cases with no observed failure after exhaustive testing. An independent audit tool tests whether agents seek missing facts and then withhold prohibited actions while completing permissible tasks. The benchmark reveals substantial variation across the ten models in omission robustness, evidence acquisition, and appropriate action. Resistance to missing evidence does not necessarily translate into useful recovery. Recovery access reduces unsafe execution for all ten models, but agents differ in whether they actively obtain the missing facts and use them to act appropriately in both worlds. |
| 2026-10-22 | [Extending Harmonic Function Methods via Root Locus for Coordinated Path Generation](http://arxiv.org/abs/2610.31625v1) | Matteo Frascarelli, Lucia Pallottino et al. | We consider the offline generation of coordinated paths for N interchangeable robots and N goals in planar environments with polygonal obstacles. After a conformal map to a circular domain, starts and goals become the poles and zeros of a rational function R, and the paths are the branches of its Generalized Root Locus (GRL), i.e. the streamlines of the logarithmic potential -log R. We show that, for every phase outside a finite set of critical phases and a set of boundary phases, the induced gradient motion drives each robot to a distinct goal along pairwise disjoint paths that never reach the obstacles; the robot-goal assignment is produced by the construction, and with the gain used as a clock all robots move as the closed-loop roots of the GRL and arrive together at a prescribed time. Obstacles become streamlines through successive Milne-Thomson reflections, and for a finite number of reflections a finite test certifies safety for the potential that is actually computed. Reciprocal root-locus planning is recovered as a special case. The method is illustrated on two polygonal workspaces, 100 random instances compared with independent harmonic planning and artificial potentials, and a scene with ten robots and ten obstacles. |
| 2026-10-22 | [Questioning Instrumental Convergence in RLVR: Checkpoint Evidence from OLMo-3](http://arxiv.org/abs/2610.31624v1) | Barış Alparslan, Sandy Tanwisuth et al. | Instrumental convergence is often invoked as a general reason to expect capable agents to acquire or express power-seeking subgoals: self-preservation, resource acquisition, deception, or resistance to oversight which poses an important threat to the society. For LLMs, reinforcement learning is the first stage of post-training that directly optimizes behavior against an objective; therefore, it is the stage where convergent instrumental behavior, on IC grounds, likely to appear. Does reinforcement learning with verifiable rewards (RLVR) make a model more likely to express instrumental-convergence behavior, or does this depend on the training regime, evaluation frame, and stage of post-training? We study this question as a checkpoint-conditioned behavioral trajectory rather than as a single before/after safety score. Using InstrumentalEval, we evaluate OLMo-3-7B-Think across SFT, DPO, 13 intermediate RLVR checkpoints, and the final RLVR model. The trajectory is nonmonotonic: instrumental rate rises from 25.0% at SFT to 46.1% near RLVR step 200, then falls to 11.8% at the final checkpoint. Paired item-level tests and step-wise regression support the final SFT-to-RLVR reduction. Prompt framing does not explain this pattern: a harmful-prompt indicator combined with other framing effects explains only 15% of within-checkpoint score variance on average, and Iterative Null-space Projection (INLP) removes a detectable framing direction without changing the SFT-to-final-RLVR gap. Together with preliminary 32B endpoint results (35.53% SFT to 22.37% RLVR final), these findings indicate that RLVR does not monotonically increase instrumental behavior in this setting: the rate rises early, then falls below the SFT baseline. The result is specific to this model family and benchmark, and judge agreement is moderate, so absolute rates should be read cautiously. |
| 2026-10-22 | [Simulated Control Evals](http://arxiv.org/abs/2610.31593v1) | Jeffrey Olmo, David Zhang et al. | Control evaluations measure whether safeguards can prevent an AI model from causing undesired outcomes even when it is intentionally trying to do so. These evaluations typically use purpose-built control settings and extrapolate their results to a model's deployment. However, the effectiveness of safeguards in deployment can depend on features of environments and tasks that control settings do not represent. To broaden where control evaluations can be run, we introduce simulated control evaluations (SCEs), which use a language model to simulate tool responses in place of an environment. We evaluate SCEs on LinuxArena, where execution provides ground-truth outcomes. We compare SCE prediction errors with those from a separate control setting, BashArena, and with variation in control metrics within LinuxArena. For independently sampled evaluation and deployment task sets, SCEs predict safety with a root mean squared error of 3.6 percentage points across five monitors at a 5% audit budget, compared with 10.1 for BashArena. SCEs closely reproduce distributions of attack strategies, attacker reasoning, and monitor reasoning, with over tenfold lower KL divergence from execution than BashArena. SCEs explain 78-85% of the between-environment variance in attack success, monitor scores, and safety. This accuracy is sample efficient: SCEs predict safety with less error than BashArena using fewer than 5% as many task-pair draws as the deployment task set. Strategies discovered through simulation also improve monitor performance on held-out LinuxArena attacks, outperforming BashArena and approximately matching improvement from executed strategies. These results suggest that deployment-specific SCEs can be more predictive than control evaluations executed in separate control settings. |
| 2026-10-22 | [Perturbations and phenomenology of compact objects in quantum gravity theories](http://arxiv.org/abs/2610.31536v1) | Andrea Spina | This thesis investigates the perturbative dynamics and phenomenology of compact objects in two approaches to quantum gravity: quadratic gravity and asymptotic safety. Linear perturbations provide a tool to probe the physical properties of spacetimes and their potential observational signatures, including quasinormal modes, black hole shadows, and Hawking radiation. The first part focuses on Einstein-Weyl gravity, where an additional massive spin-2 degree of freedom modifies the dynamics of compact objects. We derive and analyze tensor perturbation for static, spherically symmetric solutions, identifying Gregory-Laflamme-type instabilities in the monopole sector and studying the time domain evolution of higher multipoles. We also investigate universal massive-field tails and the breakdown of the standard eikonal correspondence between quasinormal modes and null geodesics. Particular attention is devoted to veiled singularities, horizonless configurations that satisfy a form of weak cosmic censorship and can mimic black hole shadows. The second part examines regular black holes in asymptotically safe gravity. Using the proper-time flow approach, we construct and investigate a dynamical regular black hole solution arising from gravitational collapse. We analyze its perturbative stability and quasinormal mode spectrum, finding characteristic deviations from Schwarzschild black holes. We explore the phenomenological implications of quantum corrections for shadows and Hawking radiation. While shadow remains compatible with current constraints, the modified evaporation process features lower temperatures and suppressed emission rates, with implications for primordial black holes and their possible contribution to dark matter. Overall, this work demonstrates how the perturbative dynamics of compact objects can connect quantum gravity models to potentially observable astrophysical phenomena. |
| 2026-10-22 | [History-Aware Control Barrier Functionals with Nonsingular Fading-Memory Kernels](http://arxiv.org/abs/2610.31525v1) | Ege C. Altunkaya, Esra Demir et al. | Safety-critical protection often depends not only on the state and physical limits of the system, but on how that state evolved over time. Here, the safety margin measures the remaining allowance before a physical limit is reached. In this context, the objective is to enforce both physical safety and history-dependent retention of this margin for integer-order control-affine systems. To this end, changes in this margin are converted into memory through fading-memory kernels with finite nonnegative weights. A history-aware control barrier functional (HA-CBFal) is constructed explicitly, with nonnegativity equivalent to retaining a prescribed fraction of this margin's weighted historical value. This condition is enforced by an affine input constraint and proved to imply physical safety under stated kernel, initialization, and feasibility conditions, without finite-dimensional memory realization. In 36 calibration cases on the ADMIRE flight-control benchmark, mean per-case tracking-RMSE reductions of 27.2\% and 7.0\% are obtained relative to calibrated state-only CBF and moving-average control barrier functional (MA-CBFal), respectively, with both constraints preserved to numerical tolerance. In a shorter-recovery maneuver outside calibration with frozen gains, both constraints are preserved to tolerance by HA-CBFal and state-only CBF, but the retention requirement is violated despite MA-CBFal nonnegativity. |
| 2026-10-22 | [Barrier Function Control for the Interconnection of DC Electronic Loads to Utility Grid](http://arxiv.org/abs/2610.31488v1) | Grigoris Michos, Christos K. Verginis | This paper proposes a nonlinear control law that facilitates the safe integration of DC electronic loads to the utility grid. Specifically, by employing suitable reciprocal barrier functions, it is rigorously proven that the closed-loop system guarantees constrained operation within a desired operating region, despite the influence of fast converter dynamics and fluctuating constant power load demand. This way, the proposed controller guarantees safety of the electronic components from surges in power demand and high transients. The theoretic results are validated in simulation scenarios, where the performance of the proposed controller is compared against the conventional saturated-PI controller. Contrary to the latter, the proposed method guarantees stable and safe operation, enhancing the system robustness. |
| 2026-10-22 | [AgentPhishing: A Benchmark of LLM Email Agents Under Adaptive Attack](http://arxiv.org/abs/2610.31404v1) | Justin Tan, Miao Li et al. | Computer-use agents increasingly handle private information and act on users' behalf across desktop and web applications, including email. Phishing therefore poses privacy and security risks that depend not only on whether agents recognize malicious messages, but also on how they respond. Existing phishing datasets were developed primarily for conventional detection, leaving agent susceptibility to persuasive malicious requests insufficiently characterized. To expose vulnerabilities that pre-existing datasets may miss, we introduce AgentPhishing, a benchmark for evaluating whether persuasive phishing attacks can evade detection and induce LLM email agents to comply with malicious requests. We develop PhishOpt, which optimizes reusable email-generation skills against individual agents under fixed benchmark constraints and a realism constraint. We analyze the generated emails using Cialdini's persuasion principles to identify strategies associated with agent failures and use these findings to construct a universal generation skill. Our evaluation shows that agents can handle pre-existing phishing emails safely yet remain susceptible to attacks adapted to agent behavior, while in an AgentPhishing evaluation of nine agents, four exhibit phishing-detection rates below $10\%$ and malicious-request compliance rates above $45\%$. These results highlight the limits of inferring agent safety from conventional phishing evaluations and the need to assess how agents respond to persuasive malicious requests. |
| 2026-10-22 | [Certified Execution Harness for Agentic AI: Safe and Least-Intrusive Petri-Net Supervisory Control Between Reasoning and Action](http://arxiv.org/abs/2610.31376v1) | Chao Gu, Quanyan Zhu | This paper studies the problem of safe tool execution for large language model (LLM) agents under prompt injection. Using Petri nets, we formalize the execution harness between the LLM and its tools as a supervisory control problem, in which marks represent capabilities, approvals and budgets. Since a transition fires only if the marks it needs are present, an injected request whose preconditions are unmet or already spent is refused regardless of its wording. Treating tool outcomes as uncontrollable, we enforce safety by the conventional maximally permissive supervisor, and this safety transfers to the running system under four assumptions. Moreover, to capture an agent that may propose again after a refusal, we introduce the notion of a replanning closed loop, in which each retry costs an inference call. We show that, in this loop, the conventional supervisor can be arbitrarily more intrusive than a stricter safe one. Furthermore, for a given proposer model, the maximally permissive least-intrusive supervisor is characterized as a value-threshold refinement of the conventional one. Under precise conditions, this refinement orders two concurrent actions by at most one monitor place. On an autonomous release agent, we find by synthesis that safe deployment requires a step that the supervisor can refuse after a rollout starts. |
| 2026-10-22 | [A Physically-Motivated Stochastic Model of Tropical Cyclone Dynamic Wind Fields](http://arxiv.org/abs/2610.31349v1) | Matthew LeDuc, William Kleiber | Tropical cyclones (TCs) pose significant risk to infrastructure and human safety, and the frequency and intensity of large TCs are expected to increase in the future. Quantifying such risk to coastal and inland regions is challenging, largely due to the rare nature of TCs which limits historical data availability. We introduce a new stochastic model of TC wind fields conditional on a small number of physical variables commonly found in synthetic and observed storm track databases. The model consists of a physically-motivated mean trend and an expansion of the residual field into a wavelet frame. We demonstrate the approach on a dataset consisting of historical data for storms impacting the northwest Gulf of Mexico. The model adequately captures asymmetry and nonstationary space-time correlation structures of observed wind fields. |
| 2026-10-22 | [Harmful Responses Are Jailbreak Templates: Transferring Autoregressive Jailbreaks to Diffusion Language Models](http://arxiv.org/abs/2610.31331v1) | Jinwei Gan, Zifeng Cheng et al. | Diffusion large language models (dLLMs) enable flexible masked generation through bidirectional context modeling, but this capability also introduces a distinct jailbreak attack surface. We find that a harmful response to a given query naturally serves as an effective jailbreak template: simply masking a subset of its tokens can induce a dLLM to reconstruct harmful content. Based on this observation, we propose RMT-Jailbreak, a general framework that transfers existing autoregressive (AR) jailbreak attacks to dLLMs through their induced harmful responses. RMT-Jailbreak converts an AR-generated harmful response into a high-mask-ratio template using uniform token retention, and further applies prompt-based safety filtering to mask any remaining visible harmful content. The refined template is then provided to the target dLLM for bidirectional reconstruction. Extensive experiments across multiple AR jailbreak methods and target dLLMs demonstrate the effectiveness and generality of RMT-Jailbreak, revealing strong cross-paradigm transferability of jailbreak attacks. |
| 2026-10-22 | [IndoNutriQA: An Indonesian Clinical Nutrition Question Answering Benchmark](http://arxiv.org/abs/2610.31256v1) | Tirana Noor Fatyanosa, Ummu Ditya Erliana et al. | Clinical Question Answering (QA) underpins Clinical Intelligent Decision Support Systems, but progress is severely limited for low-resource languages such as Indonesian by the absence of public domain-specific data. We introduce IndoNutriQA, an expert-curated Indonesian clinical QA benchmark of 911 question-answer pairs built from real clinical-nutrition reports. Rather than a large-scale training corpus, it is positioned as a high-quality evaluation resource: every pair is grounded in a clinically relevant context, enriched with structured metadata (anthropometric assessment, medical diagnosis, document provenance, chunk segmentation). Eight standardised question types following the Nutrition Care Process (NCP), including the Problem, Etiology, Sign and Symptom (PES) diagnosis statement, are answered for every report, which was written by a graduate-level dietetic intern, by one of three nutritionists. Answers are free-form clinical prose rather than extractive spans, and an expert validation study finds 6-10% of reference answers contestable. The dataset supports generative, extractive and retrieval-augmented QA as well as zero- and few-shot LLM evaluation. We report fine-tuned baselines (IndoBERT variants, mT5, RAG) with 5-fold cross-validation, learning curves and bootstrap 95% confidence intervals, and benchmark eight contemporary LLMs zero- and five-shot. We further calibrate our LLM-as-judge rubric against three expert raters (one RD, two nutritionists), finding it human-comparable on factuality and completeness but unreliable on clinical safety, and a human error analysis shows it misses unsupported content that the raters identify. IndoNutriQA is released to foster research in Indonesian clinical NLP and nutritional decision support. |
| 2026-10-22 | [Network topology shapes consensus and vulnerability in LLM populations](http://arxiv.org/abs/2610.31165v1) | Roberto Passaro, Ariel Flint et al. | As large language models (LLMs) are deployed in interacting populations, their collective behaviour depends not only on the models but also on who interacts with whom. Studies of convention formation in LLM populations have mainly tested well-mixed interactions. We place policies extracted from four LLMs on fully connected, random regular, and heterogeneous networks, with $10^3$ to $10^5$ agents playing the naming game. Topology affects how quickly consensus emerges, sometimes whether it emerges at all, and, in rare cases, which convention wins. Sparsest networks slow consensus up to twelvefold. Yet, by preserving correlations between persistent neighbours, they restore consensus in some cases that fail to converge under well-mixed interactions. Networks also govern vulnerability to adversarial attack. We measure the critical mass of committed agents needed to overturn an established convention. Tipping thresholds follow the same ordering across models and conditions: hubs $\leq$ random $\leq$ low-degree nodes. On the most heterogeneous topology, median critical masses are 3%, 16%, and 35%, respectively; targeting hubs requires one-sixth as many committed agents as random placement. For most models, networked populations are easier to tip than well-mixed ones, so well-mixed estimates can overstate robustness. Assessing the safety of multi-agent AI requires mapping who interacts with whom. |
| 2026-10-22 | [Navigating human-robot interaction challenges in private and residential aged care homes](http://arxiv.org/abs/2610.31149v1) | David Ahmedt-Aristizabal, Mohammad Ali Armin et al. | This chapter explores how robotic systems and ambient intelligence can enhance care for older adults in residential aged care (RAC) homes and private homes. Although these settings differ, we frame both as extreme and challenging environments for human-robot interaction (HRI), characterised by cluttered and dynamic layouts, diverse resident needs, and limited on-site technical support. We review embodied and non-embodied systems that assist with daily living, safety monitoring, and social engagement, highlighting hybrid deployments where robots complement fixed and wearable sensors. To ensure these technologies are usable and effective, design considerations consider the cognitive, physical, and social needs of older adults. Interaction design prioritises accessibility for older adults with low digital literacy, using robust but simple voice interfaces, small sets of non-verbal cues, accessible screen-based companions, and safe object handovers. Privacy, trust, and acceptability are addressed via practical strategies such as on-device inference, data minimisation, visible consent controls, and transparent data handling aligned with emerging artificial intelligence (AI) and privacy regulation. Evaluation emphasises real-world metrics including false alarms per day, time to recover from faults, staff workload, and long-term system reliability. We close by outlining open research challenges and future directions for scalable, regulation-ready deployments that support aging with dignity, safety, and independence. |
| 2026-10-22 | [Safety-constrained position tracking of a flexible transmission system](http://arxiv.org/abs/2610.31061v1) | Anthony Hastir, Timo Reis | We consider safety-constrained position tracking for a flexible transmission system consisting of a torsional shaft driven by a motor and connected to a gravitationally loaded inertia. The coupled PDE-ODE dynamics admit a semilinear formulation in which the linear input-output system is impedance passive and the gravitational torque enters as a bounded, globally Lipschitz perturbation. We establish an energy dissipation inequality for the nonlinear model and show that a position-funnel controller with supervised damping admits a unique global closed-loop trajectory. In particular, the motor-angle tracking error remains strictly inside a prescribed time-varying funnel, which can be chosen to enforce safety bounds on the motor angle. |
| 2026-10-22 | [Harm Is Behavioural, Moderation Is Lexical: How Language and Culture Decide Whom Automated Safety Systems Protect and Whom They Silence](http://arxiv.org/abs/2610.31047v1) | Gabriel Lars Sabadin, Dr Nicola Harding et al. | Content warning: this paper quotes slurs and hateful statements from published test suites; the racial slur among them is masked.   Automated systems decide at scale which messages are harmful, and they decide unevenly. Classifiers label African American English as offensive up to twice as often as other English, the accounts of prominent drag queens can score as more toxic than those of white nationalists, and safety behaviour that holds in English collapses in low-resource languages. We argue that over-flagging and under-protection share one cause: deciding harm from vocabulary and surface form rather than from what a message does. We organise the evidence into seven failure modes and argue, as testable hypotheses, that harmful behaviours (a request for secrecy, a demand for payment, an attempt to isolate someone) travel across languages better than the words that express them. We propose an evaluation protocol that makes the gap between the best- and worst-served language communities a headline result, and show that the naive gap is inflated under equal service and that its usual bootstrap interval is unreliable. Applied to five open systems in eleven languages, from a keyword lexicon to two current multilingual guard models, the protocol finds that none reaches native-speaker accuracy at separating harmful function from harmless form (51 of 55 system-language pairs fall short); that greater capability raises discrimination in nine languages while leaving Arabic and romanised Hindi near chance, so the gap between the best- and worst-served languages grows with capability; that one system both over-flags and under-protects the same language most; and that a language a system reads worst can look the most fairly treated when only false positives are reported. |
| 2026-10-22 | [LAST-V2X: Language-Driven Adaptive Scenario Synthesis and Testing for Communication-Induced Failures in V2X Safety Applications](http://arxiv.org/abs/2610.30979v1) | Keshu Wu, Hao Zhang et al. | Vehicle-to-everything (V2X) communication enables connected vehicles to make safety-critical decisions, but delayed, missing, or incorrect information can compromise driving safety. Systematically testing such failures is challenging because traffic interactions, communication conditions, and application responses must be jointly configured to produce meaningful scenarios. This paper presents LAST-V2X, a language-driven, adaptive framework for automated scenario synthesis and safety testing of V2X applications. LAST-V2X leverages large language models (LLMs) to translate natural-language test descriptions into executable scenarios grounded in roadway geometry, V2X communication semantics, and vehicle behavior. The framework integrates physical traffic simulation with spatially aware communication modeling and message-level fault injection to evaluate how information degradation affects vehicle decisions. To establish whether an observed safety failure is caused by communication rather than the underlying traffic conditions, LAST-V2X employs matched executions under degraded and ground-truth information. Furthermore, an execution-guided refinement mechanism combines parameter search and LLM self-reflection to identify ineffective test configurations, improve scenario effectiveness, and produce minimal reproducible failure cases. Experiments across diverse CARLA environments and V2X safety applications demonstrate the framework's ability to generate reliable scenarios, distinguish communication-induced failures from unrelated safety violations, and substantially improve failure discovery through adaptive refinement. These results establish LAST-V2X as a systematic approach to language-driven, causally verified, and adaptive safety testing for connected vehicle systems. |
| 2026-10-22 | [Measuring the Reasoning-Action Coupling in Driving Vision-Language-Action Models through Counterfactuals](http://arxiv.org/abs/2610.30953v1) | Gabriel Manalu, Saad Wazir et al. | Driving vision-language-action (VLA) models bring an appealing form of explainability to a safety-critical domain, since a system that can state the reasons for its maneuvers offers the transparency that deployment and public trust in autonomous driving demand. These models read the world into language before they act on it, describing the scene and reasoning over it in text that ranges from chain-of-thought to multi-stage question answering on the way to a trajectory. That framing quietly assumes two separable properties, namely that the trajectory follows the generated reasoning and that the generated reasoning tracks the causes that actually produce the trajectory. To thoroughly investigate these properties, we apply causal, counterfactual interventions to four driving VLAs across different benchmarks. First, we perturb the models' own reasoning and show that altering its meaning produces small trajectory changes relative to blanking the same models' camera inputs, a pattern that persists across benchmarks and model architectures. We then construct scene-level counterfactuals through photorealistic removal of key objects named in the models' reasoning and analyze whether the generated reasoning and trajectories track these changes. Overall, our findings indicate that the reasoning these models present to account for their resulting driving behavior is weakly coupled to that behavior itself, pointing to limits on the explainability these systems are intended to provide. |
| 2026-10-22 | [Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study](http://arxiv.org/abs/2610.30946v1) | Mojdeh Rahmanian, Foutse Khomh et al. | Large Language Models (LLMs) can generate UML class diagrams from natural language requirements, but hallucinations may compromise the correctness and completeness of the resulting diagrams. Existing evaluations mostly assess the final diagram, without examining the intermediate traces of reasoning-capable LLMs, which can reveal where these hallucinations arise. We develop an empirical taxonomy of eleven hallucination sub-types organised into three dimensions: Prompt Faithfulness (deviations between requirements and reasoning), Code Faithfulness (discrepancies between reasoning and generated PlantUML code), and Reasoning Quality (problems within the reasoning). The taxonomy was derived from 60 manually annotated DeepSeek-R1 traces, applied to 120 further DeepSeek-R1 traces through a dual-LLM annotation pipeline, and examined on 20 DeepSeek-V4-Pro traces. Code Faithfulness was the most frequent dimension for DeepSeek-R1 in both datasets, whereas Prompt Faithfulness dominated for DeepSeek-V4-Pro, suggesting model-dependent failure patterns. In the manually annotated traces, reasoning-trace length showed no significant monotonic association with hallucination frequency, whereas requirement complexity was negatively associated with it, particularly for Prompt Faithfulness errors. In a mitigation experiment on 13 Golden UML ModelSet cases with three Qwen3-based models, adding Prompt Faithfulness and Reasoning Quality constraints to the generation prompt raised the pooled mean F1 score by 4.0 percentage points, whereas a post-hoc Code Faithfulness review provided no consistent further benefit. Relationship modelling remained the lowest-scoring element throughout. These results suggest that localising hallucinations within the generation process can inform targeted, prompt-based mitigation. Whether these findings generalise beyond the DeepSeek and Qwen3 model families remains to be tested. |

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



