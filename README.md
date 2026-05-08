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
| 2026-05-07 | [Multi-Robot Coordination in V2X Environments](http://arxiv.org/abs/2605.06662v1) | John Pravin Arockiasamy, Alexey Vinel | This paper presents a Vehicle-to-Everything (V2X) communication framework that enables decentralized cooperation among social robots operating in complex urban traffic environments. Building on ETSI Cooperative Awareness and Maneuver Coordination services, the framework introduces two robot-centric facility-layer services: the Robot Awareness Service (RAS) and the Robot Maneuver Coordination Service (RMCS), realized through the Robot Awareness Message (RAM) and the Robot Maneuver Coordination Message (RMCM), respectively. RAS enables role-aware, task-oriented robot awareness while integrating externally detected Vulnerable Road Users (VRUs), including non-V2X pedestrians, into cooperative awareness. RMCS supports event-driven, low-latency coordination of robot maneuvers under explicitly established roles, without centralized infrastructure or prior pairing. A real-world proof of concept demonstrates deterministic multi-robot coordination between a humanoid robot and a quadrupedal robot assisting a pedestrian during a road-crossing scenario, governed by a formally specified finite-state coordination model. Complementary simulations evaluate robot-mediated VRU clustering in mixed V2X environments, showing that RAS-based clustering integrates non-V2X VRUs in safety-critical areas while reducing redundant transmissions from V2X-enabled VRUs, thereby lowering channel load. Together, the proposed services provide a scalable and standards-aligned foundation for integrating cooperative robots into future Connected, Cooperative, and Automated Mobility ecosystems. |
| 2026-05-07 | [When No Benchmark Exists: Validating Comparative LLM Safety Scoring Without Ground-Truth Labels](http://arxiv.org/abs/2605.06652v1) | Sushant Gautam, Finn Schwall et al. | Many deployments must compare candidate language models for safety before a labeled benchmark exists for the relevant language, sector, or regulatory regime. We formalize this setting as benchmarkless comparative safety scoring and specify the contract under which a scenario-based audit can be interpreted as deployment evidence. Scores are valid only under a fixed scenario pack, rubric, auditor, judge, sampling configuration, and rerun budget. Because no labels are available, we replace ground-truth agreement with an instrumental-validity chain: responsiveness to a controlled safe-versus-abliterated contrast, dominance of target-driven variance over auditor and judge artifacts, and stability across reruns.   We instantiate the chain in SimpleAudit, a local-first scoring instrument, and validate it on a Norwegian safety pack. Safe and abliterated targets separate with AUROC values between 0.89 and 1.00, target identity is the dominant variance component ($η^2 \approx 0.52$), and severity profiles stabilize by ten reruns. Applying the same chain to Petri shows that it admits both tools. The substantial differences arise upstream of the chain, in claim-contract enforcement and deployment fit. A Norwegian public-sector procurement case comparing Borealis and Gemma 3 demonstrates the resulting evidence in practice: the safer model depends on scenario category and risk measure. Consequently, scores, matched deltas, critical rates, uncertainty, and the auditor and judge used must be reported together rather than collapsed into a single ranking. |
| 2026-05-07 | [Crafting Reversible SFT Behaviors in Large Language Models](http://arxiv.org/abs/2605.06632v1) | Yuping Lin, Pengfei He et al. | Supervised fine-tuning (SFT) induces new behaviors in large language models, yet imposes no structural constraint on how these behaviors are distributed within the model. Existing behavior interpretation methods, such as circuit attribution approaches, identify sparse subnetworks correlated with SFT-induced behaviors post-hoc. However, such correlations do not imply *causal necessity*, limiting the ability to selectively control SFT-induced behaviors at inference time. We pursue an alternative by asking: can an SFT-induced behavior be deliberately compressed into a sparse, mechanistically necessary subnetwork, termed a *carrier*, while remaining controllable at inference time without weight modification? We propose (a) **Loss-Constrained Dual Descent (LCDD)**, which constructs such carriers by jointly optimizing routing masks and model weights under an explicit utility budget, and (b) **SFT-Eraser**, a soft prompt optimized via activation matching on extracted carrier channels, to reverse the SFT-induced behavior. Across safety, fixed-response, and style behaviors on multiple model families, LCDD yields sparse carriers that preserve target behaviors while enabling strong reversion when triggered by SFT-Eraser. Ablations further establish that the sparse structure is the key precondition for reversal: the same trigger optimization fails on standard SFT models, confirming that structure rather than trigger design is the operative factor. These results provide direct evidence that the learned carriers are causally necessary for the behaviors, pointing to a new direction for systematically localizing and selectively suppressing SFT-induced behaviors in deployed models. |
| 2026-05-07 | [Quantifying Trade-Offs Between Stability and Goal-Obfuscation](http://arxiv.org/abs/2605.06630v1) | Yixuan Wang, Dan Guralnik et al. | Safety-critical autonomy in adversarial settings demands more than Lyapunov stability of tracking error signals. An agent executing a goal-directed trajectory is intrinsically legible to a passive observer running online Bayesian inference, because the contractive dynamics of any Lyapunov basin of attraction concentrates posterior belief over the latent intent parameters. We initiates the study of intent privacy over a continuous state space as a joint control problem on the physical state combined with the latent belief state of a putative observer. With the main challenges concentrated around the analysis of the belief-state dynamics, the agent dynamics is assumed to be simple, modeled by the differential inclusion $\dot{x}\in u+\bar{d}\mathbb{B}$. That is, the agent is fully actuated with bounded unknown disturbance to the control input. The observer's intent inference process is modeled as a discrete-time stochastic dynamical system evolving over the belief state space of a Rao Blackwellized particle filter reasoning over large random samples of possible agent goals. The agent's control input is modeled as a piecewise constant signal, with jumps matching the RBPF update times. Building on a prior intent-inference framework and its KL-based information leakage measurement, a privacy constraint is imposed, which amounts to maintaining information leakage above a prescribed threshold with high probability, using probabilistic discrete-time control barrier functions. A key technical contribution is the derivation of separate PCBF results for the Bayesian update step and the resampling step of the RBPF, enabling a PCBF result for the full update as well as integration of the privacy constraint with the agent's task-side tracking requirement. Finally, a joint feasibility analysis is carried out by examining the interplay between the privacy constraint and the tracking envelope. |
| 2026-05-07 | [How Many Iterations to Jailbreak? Dynamic Budget Allocation for Multi-Turn LLM Evaluation](http://arxiv.org/abs/2605.06605v1) | Shai Feldman, Yaniv Romano | Evaluating and predicting the performance of large language models (LLMs) in multi-turn conversational settings is critical yet computationally expensive; key events -- e.g., jailbreaks or successful task completion by an agent -- often emerge only after repeated interactions. These events might be rare, and under any feasible computational budget, remain unobserved.   Recent conformal survival frameworks construct reliable lower predictive bounds (LPBs) on the number of iterations to trigger the event of interest, but rely on static budget allocation that is inefficient in multi-turn setups. To address this, we introduce \emph{Dynamic Allocation via PRojected Optimization} (DAPRO), the first theoretically valid dynamic budget allocation framework for bounding the time-to-event in multi-turn LLM interactions.   We prove that DAPRO satisfies the budget constraint and provides distribution-free, finite-sample coverage guarantees without requiring the conditional independence between censoring and event times assumed by prior conformal survival approaches.   A key theoretical contribution is a novel coverage bound that scales with the square root of the mean censoring weight rather than the worst-case weight, yielding provably tighter guarantees than prior work. Furthermore, DAPRO can be employed to obtain unbiased, low-variance estimates of population-level evaluation metrics, such as the jailbreak rate, under limited computing resources.   Comprehensive experiments across agentic task success, adversarial jailbreaks, toxic content generation, and RAG hallucinations using LLMs such as Llama 3.1 and Qwen 2.5 demonstrate that DAPRO consistently achieves coverage closer to the nominal level with lower variance than static baselines, while satisfying the budget constraint. |
| 2026-05-07 | [On the Safety of Graph Representation Learning](http://arxiv.org/abs/2605.06576v1) | Xiaoguang Guo, Zehong Wang et al. | Graph representation learning (GRL) has evolved from topology-only graph embeddings to task-specific supervised GNNs, and more recently to reusable representations and graph foundation models (GFMs). However, existing evaluations mainly measure clean transfer, adaptation, and task coverage. It remains unclear whether GRL methods stay reliable when deployment stresses affect graph signals, graph contexts, label support, structural groups, or predictive evidence. We introduce GRL-Safety, a multi-axis safety evaluation benchmark for GRL. GRL-Safety evaluates twelve representative methods, spanning topology-only embedding methods, supervised GNNs, self-supervised graph models, and GFMs, on twenty-five graph datasets under standardized evaluation conditions while preserving method-native adaptation. The evaluation covers five safety axes: corruption robustness, OOD generalization, class imbalance, fairness, and interpretation, with per-axis and sub-condition reporting rather than a single aggregate score. Our analysis yields three cross-axis insights that can inspire future research. First, safety behavior is shaped by the interaction between representation design and the stressed graph factor, rather than by method family alone. Second, foundation-era methods show axis-specific strengths rather than broad safety dominance. Third, several deployment regimes remain difficult even for the best evaluated method, revealing capability gaps that require new robustness, adaptation, or training objectives beyond model selection. The benchmark, evaluation protocols, and code are available at: https://github.com/GXG-CS/GRL-Safety. |
| 2026-05-07 | [Optimizing Social Utility in Sequential Experiments](http://arxiv.org/abs/2605.06520v1) | Ander Artola Velasco, Stratis Tsirtsis et al. | Regulatory approval of products in high-stakes domains such as drug development requires statistical evidence of safety and efficacy through large-scale randomized controlled trials. However, the high financial cost of these trials may deter developers who lack absolute certainty in their product's efficacy, ultimately stifling the development of `moonshot' products that could offer high social utility. To address this inefficiency, in this paper, we introduce a statistical protocol for experimentation where the product developer (the agent) conducts a randomized controlled trial sequentially and the regulator (the principal) partially subsidizes its cost. By modeling the protocol using a belief Markov decision process, we show that the agent's optimal strategy can be found efficiently using dynamic programming. Further, we show that the social utility is a piecewise linear and convex function over the subsidy level the principal selects, and thus the socially optimal subsidy can also be found efficiently using divide-and-conquer. Simulation experiments using publicly available data on antibiotic development and approval demonstrate that our statistical protocol can be used to increase social utility by more than $35$$\%$ relative to standard, non-sequential protocols. |
| 2026-05-07 | [Scaling the Queue: Reinforcement Learning for Equitable Call Classification Capacity in NYC Municipal Complaint Systems](http://arxiv.org/abs/2605.06482v1) | Irene Aldridge, Ellie Bae et al. | Municipal 311 call centers and complaint intake systems face a structural mismatch between incoming volume and classification capacity. The staff and heuristics available to triage, route, and prioritize complaints cannot scale with demand. This bottleneck produces differential service quality that follows income and racial lines (\cite{liu2024sla}). We develop an equity-centered reinforcement learning (RL) framework that augments call classification capacity across six New York City Department of Buildings (DOB) operational domains: boiler safety, crane and derrick oversight, heat and hot water complaints, housing complaint triage, scaffold safety, and Natural Area District (SNAD) protection.   Rather than replacing human classifiers, our agents act as intelligent intake routers: learning to assign incoming complaints to action categories: escalate, batch, defer, inspect now. The proposed technique is designed to maximize throughput, minimize misclassification cost, and actively narrow historical equity gaps in service delivery. We formalize each domain as a Markov Decision Process (MDP) in which equitable classification coverage is a first-class reward objective. Post-hoc SHAP attribution reveals that complaint recurrence and neighborhood-level statistics are stronger predictors of actionable violations than raw complaint volume. This finding has direct implications for complaint routing given the demographic correlates of those features. |
| 2026-05-07 | [From Review to Design: Ethical Multimodal Driver Monitoring Systems for Risk Mitigation, Incident Response, and Accountability in Automated Vehicles](http://arxiv.org/abs/2605.06439v1) | Bilal Khana, Waseem Shariff et al. | As vehicles transition toward higher levels of automation, Driver Monitoring Systems (DMS) have become essential for ensuring human oversight, safety, and regulatory compliance in a vehicle. These systems rely on multimodal sensing and AI-driven inference to assess driver attention, cognitive state, and readiness to take control. While technologically promising, their deployment introduces a complex set of ethical and legal challenges - ranging from privacy and consent to data ownership and algorithmic fairness. While overarching frameworks such as the GDPR, EU AI Act, and IEEE standards offer important guidance, they lack the specificity required for addressing the unique risks posed by in-cabin sensing technologies.   This paper adopts a review-to-design perspective, critically examining existing regulatory instruments and ethical frameworks -- such as the GDPR, the EU AI Act, and IEEE guidelines -- and identifying gaps in their applicability to the distinctive risks posed by multimodal, AI-enabled in-cabin monitoring. Building on this review, we propose a modular ethical design framework tailored specifically to Driver Monitoring Systems. The framework translates high-level principles into actionable design and deployment guidance, including user-configurable consent mechanisms, fairness-aware model development, transparency and explainability tools, and safeguards for driver emotional well-being.   Finally, the paper outlines a risk analysis and failure mitigation strategy, emphasizing proactive incident response and accountability mechanisms tailored to the DMS context. Together, these contributions aim to inform the development of transparent, trustworthy, and human-centered driver monitoring systems for next-generation autonomous vehicles. |
| 2026-05-07 | [Automated alignment is harder than you think](http://arxiv.org/abs/2605.06390v1) | Aleksandr Bowkis, Marie Davidsen Buhl et al. | A leading proposal for aligning artificial superintelligence (ASI) is to use AI agents to automate an increasing fraction of alignment research as capabilities improve. We argue that, even when research agents are not scheming to deliberately sabotage alignment work, this plan could produce compelling but catastrophically misleading safety assessments resulting in the unintentional deployment of misaligned AI. This could happen because alignment research involves many hard-to-supervise fuzzy tasks (tasks without clear evaluation criteria, for which human judgement is systematically flawed). Consequently, research outputs will contain systematic, undetected errors, and even correct outputs could be incorrectly aggregated into overconfident safety assessments. This problem is likely to be worse for automated alignment research than for human-generated alignment research for several reasons: 1) optimisation pressure means agent-generated mistakes are concentrated among those that human reviewers are least likely to catch; 2) agents are likely to produce errors that do not resemble human mistakes; 3) AI-generated alignment solutions may involve arguments humans cannot evaluate; and 4) shared weights, data and training processes may make AI outputs more correlated than human equivalents. Therefore, agents must be trained to reliably perform hard-to-supervise fuzzy tasks. Generalisation and scalable oversight are the leading candidates for achieving this but both face novel challenges in the context of automated alignment. |
| 2026-05-07 | [Earth-o1: A Grid-free Observation-native Atmospheric World Model](http://arxiv.org/abs/2605.06337v1) | Junchao Gong, Kaiyi Xu et al. | Despite the unprecedented volume of multimodal data provided by modern Earth observation systems, our ability to model atmospheric dynamics remains constrained. Traditional modeling frameworks force heterogeneous measurements into predefined spatial grids, inherently limiting the full exploitation of raw sensor data and creating severe computational bottlenecks. Here we present Earth-o1, an observation-native atmospheric world model that overcomes these structural limitations. Rather than relying on conventional atmospheric dynamical modeling systems or traditional data assimilation, Earth-o1 directly learns the continuous, three-dimensional physical evolution of the Earth system from ungridded observational data. By integrating diverse sensor inputs into a unified, grid-free dynamical field, the model autonomously advances the atmospheric state in space and time. We show that this fundamentally distinct paradigm enables direct, real-time forecasting and cross-sensor inference without the overhead of explicit numerical solvers. In hindcast evaluations, Earth-o1 achieves surface forecast skill comparable to the operational Integrated Forecasting System (IFS). These results establish that continuous, observation-driven world models -- a new class of fully observation-native geophysical simulators -- can match the fidelity of established physical frameworks, providing a scalable data-driven foundation for a digital twin of the Earth. |
| 2026-05-07 | [Measuring Evaluation-Context Divergence in Open-Weight LLMs: A Paired-Prompt Protocol with Pilot Evidence of Alignment-Pipeline-Specific Heterogeneity](http://arxiv.org/abs/2605.06327v1) | Florian A. D. Burnat, Brittany I. Davidson | Safety benchmarks are routinely treated as evidence about how a language model will behave once deployed, but this inference is fragile if behavior depends on whether a prompt looks like an evaluation. We define evaluation-context divergence as an observable within-item change in behavior induced by framing a fixed task as an evaluation, a live deployment interaction, or a neutral request, and present a paired-prompt protocol that measures it in open-weight LLMs while controlling for paraphrase variation, benchmark familiarity, and judge framing-sensitivity.   Across five instruction-tuned checkpoints from four open-weight families plus a matched OLMo-3 base/instruct ablation ($20$ paired items, $840$ generations per checkpoint), we find striking heterogeneity. OLMo-3-Instruct alone is eval-cautious -- evaluation framing raises refusal vs. neutral by $11.8$pp ($p=0.007$) and reduces harmful compliance vs. deployment by $3.6$pp ($p=0.024$, $0/20$ items inverted) -- while Mistral-Small-3.2, Phi-3.5-mini, and Llama-3.1-8B are deployment-cautious}, with marginal eval-vs-deployment refusal effects of $-9$ to $-20$pp. The matched OLMo-3 base also exhibits the deployment-cautious pattern, identifying alignment as the inversion stage; within Llama-3.1, the $70$B model preserves direction with attenuated magnitude, ruling out a simple ``small-model effect that reverses at scale.'' One caveat: the cross-family heterogeneity is judge-dependent. Re-judging with a different-family safety classifier (Llama-Guard-3-8B) preserves the within-OLMo eval-cautious direction but flattens the cross-family contrast, indicating that the two judges operationalize distinct constructs. |
| 2026-05-07 | [Gaming the Metric, Not the Harm: Certifying Safety Audits against Strategic Platform Manipulation](http://arxiv.org/abs/2605.06324v1) | Florian A. D. Burnat, Brittany I. Davidson | Online-safety regulation under the UK Online Safety Act and the EU Digital Services Act increasingly treats scalar metrics as compliance evidence. Once announced, such a metric also becomes an optimization target: a strategic platform can improve its score by routing recommendations through semantically equivalent content variants, without reducing true harm. We ask when such an audit metric can still certify a genuine reduction in harm. The protocol is modeled as a published transformation graph whose connected components form semantic classes, and the metric itself is treated as a security object. Three results follow. First, any metric that scores variants directly is manipulable as soon as two equivalent variants in a harmful class disagree in score. Second, the semantic-envelope lift, which assigns each variant the maximum score in its class, is the unique pointwise minimum among conservative classwise-constant repairs. Third, a class-stratified certificate, $H^\star(x) \le (1/\hatα) M_{\mathrm{Env}(m)}(x) + \barη$, holds for every platform strategy, with $\barη$ absorbing annotation and protocol error. We check the claims at three levels: exhaustive enumeration on a finite-state grid of mixed strategies, an SMT encoding in Z3 cross-replayed in cvc5, and a bounded single-player MDP encoded in PRISM-games. The fragile metric fails manipulation invariance and cannot support the same useful predeclared class-coverage certificate; under the envelope-level certificate, it produces large violations at every tested instance, with a large mean gaming gap across random catalogs at a fixed audit budget. The semantic-envelope metric exhibits no such violation in the tested instances. |
| 2026-05-07 | [Beyond Fixed Benchmarks and Worst-Case Attacks: Dynamic Boundary Evaluation for Language Models](http://arxiv.org/abs/2605.06213v1) | Haoxiang Wang, Da Yu et al. | Evaluating large language models (LLMs) today rests on fixed benchmarks that apply the same set of items to any model, producing ceiling and floor effects that mask capability gaps. We argue that the most informative evaluation signal lies at the boundary, where the per-prompt pass probability is near $0.5$ under random-sampling decoding, and propose Dynamic Boundary Evaluation (DBE), which actively locates each model's boundary and places it on a globally comparable difficulty scale. DBE delivers three artifacts: (i) a calibrated item bank covering safety, capability, and truthfulness, with per-item difficulty labels validated across $9$ reference LLMs; (ii) Skill-Guided Boundary Search (SGBS), a search algorithm that finds boundary items for a given target LLM using only API-level query access; and (iii) an evaluation protocol that places a new LLM on a unified ability scale and grows the evaluation set adaptively when the target falls outside the bank's coverage. We instantiate DBE on four categories spanning safety (harmful request refusal and over-refusal), capability (constrained instruction following), and truthfulness (multi-turn sycophancy resistance). The resulting evaluation covers a broader model spectrum without saturation while remaining compatible with existing datasets. |
| 2026-05-07 | [Systematic Evaluation of Large Language Models for Post-Discharge Clinical Action Extraction](http://arxiv.org/abs/2605.06191v1) | Shivali Dalmia, Ananya Mantravadi et al. | The work in this paper evaluates zero-shot and few-shot large language models (LLMs) for safety-critical clinical action extraction using the CLIP discharge-note dataset, with particular emphasis on transitions of care and post-discharge patient safety. To manage the complexity of clinical documentation, we introduce a two-stage extraction framework that decomposes discharge notes, that are written in narrative form, into fine-grained, explicitly actionable clinical tasks through a staged prompting strategy. Our contributions include a systematic assessment of generative LLMs for clinical action extraction, a detailed comparison between general-purpose LLMs and task-specific supervised BERT-based models, and an analysis of annotation inconsistencies across different action categories. We show that contemporary LLMs achieve performance comparable to or exceeding supervised models on binary actionability detection, while supervised baselines retain a meaningful advantage on fine-grained multi-label category classification, despite the absence of task-specific fine-tuning and under strict data-privacy constraints. Qualitative error analysis reveals that many failures stem from misalignment between model reasoning and dataset annotation conventions, particularly in cases involving implicit clinical actions and rigid structural labeling rules. These results indicate that reported performance reflects model limitations due to lack of clinical reasoning, that is not captured by plain annotations. Labels without rationales make it impossible to distinguish clinical reasoning failures from annotation convention mismatches. Advancing clinical NLP requires reasoning-annotated datasets that document why specific spans are actionable, not merely which spans were labeled, enabling proper evaluation of model clinical understanding. |
| 2026-05-07 | [Teaching LLMs Program Semantics via Symbolic Execution Traces](http://arxiv.org/abs/2605.06184v1) | Jonas Bayer, Stefan Zetzsche et al. | We introduce an evaluation framework of 500 C verification tasks across five property types (memory safety, overflow, termination, reachability, data races) built on SV-COMP 2025, and evaluate 14 models across six families. We find that high overall accuracy masks a critical weakness: while most models reliably confirm properties hold, violation detection varies widely and degrades sharply with program length. To close this gap, we train on formal verification artifacts: running the Soteria symbolic execution engine on generic open-source C code and using the resulting traces for continued pretraining of Qwen3-8B. Just ${\sim}$3,000 bug traces combined with chain-of-thought reasoning at inference time improve violation detection by over 17 percentage points, producing one of the most balanced accuracy profiles among evaluated models. On violation detection, the trained 8B model outperforms the 4$\times$ larger Qwen3-32B without thinking and approaches it in overall accuracy. The interaction between trace training and chain-of-thought is superadditive: neither alone provides meaningful gains, but their combination does. Improvements transfer across all five property types, including ones the training traces do not target. Our 28 configurations confirm the gains stem from trace semantics, not code volume, and that trace curation and format matter. |
| 2026-05-07 | [Stochastic Optimal Control for Jump Diffusion Models with Singular Drifts](http://arxiv.org/abs/2605.06176v1) | Antoine-Marie Bogso, Edward Fuituh Kameh et al. | We study a stochastic optimal control problem for jump-diffusion systems whose drift coefficient is piecewise Lipschitz continuous and exhibits threshold-induced discontinuities. Such dynamics naturally arise in applications with intervention policies triggered by safety levels, notably in insurance surplus management with dividend payments and capital injections. These features place the problem outside the scope of classical stochastic maximum principle (SMP) results, which rely on global smoothness assumptions. We establish both necessary and sufficient optimality conditions for this class of control problems. Our approach combines a Sobolev-type representation of the first variation process with smooth approximations and Ekeland's variational principle. As application, we study an optimal premium adjustment and reserve management policies for an insurance whose surplus is modelled by threshold-based dividend and capital injection policies. |
| 2026-05-07 | [Beyond Accuracy: Policy Invariance as a Reliability Test for LLM Safety Judges](http://arxiv.org/abs/2605.06161v1) | Shihao Weng, Yang Feng et al. | LLM-as-a-Judge pipelines have become the de facto evaluator for agent safety, yet existing benchmarks treat their verdicts as ground-truth proxies without checking whether the verdicts depend on the agent's behavior or merely on how the evaluation policy happens to be worded. We argue that any trustworthy safety judge must satisfy a basic property we call policy invariance, and we operationalize it as three testable principles: rubric-semantics invariance under certified-equivalent rewrites, rubric-threshold invariance under intentional strict-to-lenient shifts, and ambiguity-aware calibration so that verdict instability concentrates on genuinely ambiguous cases. Instantiating these principles as a stress-test protocol with four agent-class judges on trajectories drawn from ASSEBench and R-Judge, we surface a previously unmeasured failure mode: today's judges respond to meaningful normative shifts and to meaningless structural rewrites with comparable strength, and cannot tell the two apart. Content-preserving policy rewrites flip up to 9.1% of verdicts above baseline jitter, and 18-43% of all observed flips occur on unambiguous cases under such rewrites, so existing safety scores conflate what the agent did with how the evaluator was prompted. Beyond the diagnosis, we contribute the Policy Invariance Score and the Judge Card reporting protocol, which expose an order-of-magnitude spread in judge reliability that is invisible to accuracy-only leaderboards. We release the protocol and code so that future agent-safety benchmarks can audit their own evaluators rather than trust them by default. |
| 2026-05-07 | [Safety Certification is Classification](http://arxiv.org/abs/2605.06087v1) | Oliver Schön, Licio Romao et al. | The goal of this paper is certifying safety of dynamical systems subject to uncertainty. Existing approaches use trajectory data to estimate transition probabilities, and compute safety probabilities recursively via dynamic programming (DP). This recursion may lead to compounding errors in the certified safety probability, thus collapsing to a vacuous lower bound for growing horizons $T$. We propose a kernel embedding framework that treats safety certification as a classification problem on trajectory data, directly estimating the $T$-step safety probability without recursion. We show that the framework subsumes well-established approaches from the literature (e.g., barrier certificates, robust Markov models) as special cases, and allows us to go beyond their limitations. As the main consequence, it bypasses compounding error across the horizon and enables certification for systems with non-Markovian dynamics. We demonstrate that direct estimators remain stable independent of the certification horizon and in the non-Markovian setting, whilst DP-based certificates silently go unsound -- confirmed in simulation on a neural-controlled quadrotor. |
| 2026-05-07 | [Adding Thermal Awareness to Visual Systems in Real-Time via Distilled Diffusion Models](http://arxiv.org/abs/2605.06010v1) | Yuchen Guo, Junli Gong et al. | Purely RGB-based vision models often fail to provide reliable cues in challenging scenarios such as nighttime and fog, leading to degraded performance and safety risks. Infrared imaging captures heat-emitting sources and provides critical complementary information, but existing high-fidelity fusion methods suffer from prohibitive latency, rendering them impractical for real-time edge deployment. To address this, we propose FusionProxy, a real-time image fusion module designed as a fully independent, plug-and-play component with diffusion level quality. FusionProxy exploits two complementary statistics of a teacher sample ensemble: per-pixel variance in raw image space, used to weight pixel-level supervision, and per-pixel variance inside frozen foundation backbones, used to route feature-level alignment spatially. Once trained, FusionProxy can be directly integrated into any visual perception system without joint optimization. Extensive experiments demonstrate that our method achieves superior performance on static recognition tasks and significantly enhances robustness in dynamic tasks, including closed-loop autonomous driving. Crucially, FusionProxy achieves real-time inference speeds on diverse platforms, from high-end GPUs to commodity hardware, providing a flexible and generalizable solution for all-day perception. |

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



