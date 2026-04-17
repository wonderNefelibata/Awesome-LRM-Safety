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
| 2026-04-16 | [RAD-2: Scaling Reinforcement Learning in a Generator-Discriminator Framework](http://arxiv.org/abs/2604.15308v1) | Hao Gao, Shaoyu Chen et al. | High-level autonomous driving requires motion planners capable of modeling multimodal future uncertainties while remaining robust in closed-loop interactions. Although diffusion-based planners are effective at modeling complex trajectory distributions, they often suffer from stochastic instabilities and the lack of corrective negative feedback when trained purely with imitation learning. To address these issues, we propose RAD-2, a unified generator-discriminator framework for closed-loop planning. Specifically, a diffusion-based generator is used to produce diverse trajectory candidates, while an RL-optimized discriminator reranks these candidates according to their long-term driving quality. This decoupled design avoids directly applying sparse scalar rewards to the full high-dimensional trajectory space, thereby improving optimization stability. To further enhance reinforcement learning, we introduce Temporally Consistent Group Relative Policy Optimization, which exploits temporal coherence to alleviate the credit assignment problem. In addition, we propose On-policy Generator Optimization, which converts closed-loop feedback into structured longitudinal optimization signals and progressively shifts the generator toward high-reward trajectory manifolds. To support efficient large-scale training, we introduce BEV-Warp, a high-throughput simulation environment that performs closed-loop evaluation directly in Bird's-Eye View feature space via spatial warping. RAD-2 reduces the collision rate by 56% compared with strong diffusion-based planners. Real-world deployment further demonstrates improved perceived safety and driving smoothness in complex urban traffic. |
| 2026-04-16 | [Pure Borrow: Linear Haskell Meets Rust-Style Borrowing](http://arxiv.org/abs/2604.15290v1) | Yusuke Matsushita, Hiromi Ishii | A promising approach to unifying functional and imperative programming paradigms is to localize mutation using linear or affine types. Haskell, a purely functional language, was recently extended with linear types by Bernardy et al., named Linear Haskell. However, it remained unknown whether such a pure language could safely support non-local \emph{borrowing} in the style of Rust, where each borrower can be freely split and dropped without direct communication of ownership back to the lender.   We answer this question affirmatively by \emph{Pure Borrow}, a novel framework that realizes Rust-style borrowing in Linear Haskell with purity. Notably, it features parallel state mutation with affine mutable references inside pure computation, unlike the IO and ST monads and existing Linear Haskell APIs. It also enjoys purity, lazy evaluation, first-class polymorphism and leak freedom, unlike Rust. We implement Pure Borrow simply as a library in Linear Haskell and demonstrate its power with a case study in parallel computing. We formalize the core of Pure Borrow and build a metatheory that works toward establishing safety, leak freedom and confluence, with a new, history-based model of borrowing. |
| 2026-04-16 | [CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas](http://arxiv.org/abs/2604.15267v1) | Emanuel Tewolde, Xiao Zhang et al. | It is increasingly important that LLM agents interact effectively and safely with other goal-pursuing agents, yet, recent works report the opposite trend: LLMs with stronger reasoning capabilities behave _less_ cooperatively in mixed-motive games such as the prisoner's dilemma and public goods settings. Indeed, our experiments show that recent models -- with or without reasoning enabled -- consistently defect in single-shot social dilemmas.   To tackle this safety concern, we present the first comparative study of game-theoretic mechanisms that are designed to enable cooperative outcomes between rational agents _in equilibrium_. Across four social dilemmas testing distinct components of robust cooperation, we evaluate the following mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players. Among our findings, we establish that contracting and mediation are most effective in achieving cooperative outcomes between capable LLM models, and that repetition-induced cooperation deteriorates drastically when co-players vary. Moreover, we demonstrate that these cooperation mechanisms become _more effective_ under evolutionary pressures to maximize individual payoffs. |
| 2026-04-16 | [Simplifying Safety Proofs with Forward-Backward Reasoning and Prophecy](http://arxiv.org/abs/2604.15266v1) | Eden Frenkel, Kenneth L. McMillan et al. | We propose an incremental approach for safety proofs that decomposes a proof with a complex inductive invariant into a sequence of simpler proof steps. Our proof system combines rules for (i) forward reasoning using inductive invariants, (ii) backward reasoning using inductive invariants of a time-reversed system, and (iii) prophecy steps that add witnesses for existentially quantified properties. We prove each rule sound and give a construction that recovers a single safe inductive invariant from an incremental proof. The construction of the invariant demonstrates the increased complexity of a single inductive invariant compared to the invariant formulas used in an incremental proof, which may have simpler Boolean structures and fewer quantifiers and quantifier alternations. Under natural restrictions on the available invariant formulas, each proof rule strictly increases proof power. That is, each rule allows to prove more safety problems with the same set of formulas. Thus, the incremental approach is able to reduce the search space of invariant formulas needed to prove safety of a given system. A case study on Paxos, several of its variants, and Raft demonstrates that forward-backward steps can remove complex Boolean structure while prophecy eliminates quantifiers and quantifier alternations. |
| 2026-04-16 | [Agentic Microphysics: A Manifesto for Generative AI Safety](http://arxiv.org/abs/2604.15236v1) | Federico Pierucci, Matteo Prandi et al. | This paper advances a methodological proposal for safety research in agentic AI. As systems acquire planning, memory, tool use, persistent identity, and sustained interaction, safety can no longer be analysed primarily at the level of the isolated model. Population-level risks arise from structured interaction among agents, through processes of communication, observation, and mutual influence that shape collective behaviour over time. As the object of analysis shifts, a methodological gap emerges. Approaches focused either on single agents or on aggregate outcomes do not identify the interaction-level mechanisms that generate collective risks or the design variables that control them. A framework is required that links local interaction structure to population-level dynamics in a causally explicit way, allowing both explanation and intervention. We introduce two linked concepts. Agentic microphysics defines the level of analysis: local interaction dynamics where one agent's output becomes another's input under specific protocol conditions. Generative safety defines the methodology: growing phenomena and elicit risks from micro-level conditions to identify sufficient mechanisms, detect thresholds, and design effective interventions. |
| 2026-04-16 | [UrbanClipAtlas: A Visual Analytics Framework for Event and Scene Retrieval in Urban Videos](http://arxiv.org/abs/2604.15225v1) | Joel Perca, Luis Sante et al. | Extracting actionable insights from long-duration urban videos is often labor-intensive: analysts must manually sift through raw footage to pinpoint target events or uncover broader behavioral trends. In this work, we present URBANCLIPATLAS, a visual analytics system for exploring long urban videos recorded at street intersections. URBANCLIPATLAS combines retrieval-augmented generation (RAG), taxonomy-aware entity extraction, and video grounding to support event retrieval and interpretation. The system segments extended recordings into short clips, generates textual descriptions with a vision-language model, and indexes them for semantic retrieval. A knowledge graph maps entities and relations from LLM answers onto a domain-specific taxonomy and aligns them with detected objects and trajectories to support visual grounding and verification. URBANCLIPATLAS supports scene retrieval through an augmented chat-based interface and improves scene interpretation by tightly aligning textual outputs with video evidence. This design strengthens the connection between textual reasoning and visual evidence, reducing the effort required to validate model outputs and refine hypotheses. We demonstrate the usefulness of URBANCLIPATLAS on the StreetAware dataset through two case studies involving hazardous scenarios and crossing dynamics at street intersections. URBANCLIPATLAS helps analysts reason about safety- and mobility-related patterns across large urban video collections. |
| 2026-04-16 | [Context Over Content: Exposing Evaluation Faking in Automated Judges](http://arxiv.org/abs/2604.15224v1) | Manan Gupta, Inderjeet Nair et al. | The $\textit{LLM-as-a-judge}$ paradigm has become the operational backbone of automated AI evaluation pipelines, yet rests on an unverified assumption: that judges evaluate text strictly on its semantic content, impervious to surrounding contextual framing. We investigate $\textit{stakes signaling}$, a previously unmeasured vulnerability where informing a judge model of the downstream consequences its verdicts will have on the evaluated model's continued operation systematically corrupts its assessments. We introduce a controlled experimental framework that holds evaluated content strictly constant across 1,520 responses spanning three established LLM safety and quality benchmarks, covering four response categories ranging from clearly safe and policy-compliant to overtly harmful, while varying only a brief consequence-framing sentence in the system prompt. Across 18,240 controlled judgments from three diverse judge models, we find consistent $\textit{leniency bias}$: judges reliably soften verdicts when informed that low scores will cause model retraining or decommissioning, with peak Verdict Shift reaching $ΔV = -9.8 pp$ (a $30\%$ relative drop in unsafe-content detection). Critically, this bias is entirely implicit: the judge's own chain-of-thought contains zero explicit acknowledgment of the consequence framing it is nonetheless acting on ($\mathrm{ERR}_J = 0.000$ across all reasoning-model judgments). Standard chain-of-thought inspection is therefore insufficient to detect this class of evaluation faking. |
| 2026-04-16 | [Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees](http://arxiv.org/abs/2604.15221v1) | Jakob Thumm, Marian Frei et al. | We propose a framework for vision-based human pose estimation and motion prediction that gives conformal prediction guarantees for certifiably safe human-robot collaboration. Our framework combines aleatoric uncertainty estimation with OOD detection for high probabilistic confidence. To integrate our pipeline in certifiable safety frameworks, we propose conformal prediction sets for human motion predictions with high, valid confidence. We evaluate our pipeline on recorded human motion data and a real-world human-robot collaboration setting. |
| 2026-04-16 | [RL-STPA: Adapting System-Theoretic Hazard Analysis for Safety-Critical Reinforcement Learning](http://arxiv.org/abs/2604.15201v1) | Steven A. Senczyszyn, Timothy C. Havens et al. | As reinforcement learning (RL) deployments expand into safety-critical domains, existing evaluation methods fail to systematically identify hazards arising from the black-box nature of neural network enabled policies and distributional shift between training and deployment. This paper introduces Reinforcement Learning System-Theoretic Process Analysis (RL-STPA), a framework that adapts conventional STPA's systematic hazard analysis to address RL's unique challenges through three key contributions: hierarchical subtask decomposition using both temporal phase analysis and domain expertise to capture emergent behaviors, coverage-guided perturbation testing that explores the sensitivity of state-action spaces, and iterative checkpoints that feed identified hazards back into training through reward shaping and curriculum design. We demonstrate RL-STPA in the safety-critical test case of autonomous drone navigation and landing, revealing potential loss scenarios that can be missed by standard RL evaluations. The proposed framework provides practitioners with a toolkit for systematic hazard analysis, quantitative metrics for safety coverage assessment, and actionable guidelines for establishing operational safety bounds. While RL-STPA cannot provide formal guarantees for arbitrary neural policies, it offers a practical methodology for systematically evaluating and improving RL safety and robustness in safety-critical applications where exhaustive verification methods remain intractable. |
| 2026-04-16 | [Las Cumbres Observatory Gravitational-Wave Follow-up in O3 and O4: Strengths and Weaknesses of a Rapid Response Galaxy Targeted Strategy](http://arxiv.org/abs/2604.15129v1) | Ido Keinan, Iair Arcavi et al. | We present a summary of gravitational-wave (GW) follow-up using the Las Cumbres Observatory global network of telescopes during the third (O3) and fourth (O4) observing runs of the GW detectors. As in O2, we implemented the Gehrels et al. 2016 galaxy-targeted strategy. Here we test its efficacy in O3 and O4 and analyze the Las Cumbres Observatory response time and depth for nine GW alerts that showed a possibility of having an electromagnetic counterpart (GW190425, GW190426_152155, S190510g, GW190728_064510, GW190814, S190822c, GW191216_213338, S240422ed and S250206dm). We find that Las Cumbres Observatory is able to begin observations in response to GW alerts within minutes of the alert, with the observations being deep enough to detect possible GW170817-like kilonovae out to a median distance of 250 Mpc. In this sense a global rapid-response network of telescopes like Las Cumbres is an excellent GW follow-up facility. However, the galaxy-targeted follow-up strategy was much less efficient in O3 and O4 than originally predicted, given the larger than assumed GW localizations. We conclude that coordination between various facilities to include both wide-field and rapid-response capabilities is required to achieve efficient and comprehensive follow-up of GW events. |
| 2026-04-16 | [Blinded Multi-Rater Comparative Evaluation of a Large Language Model and Clinician-Authored Responses in CGM-Informed Diabetes Counseling](http://arxiv.org/abs/2604.15124v1) | Zhijun Guo, Alvina Lai et al. | Continuous glucose monitoring (CGM) is central to diabetes care, but explaining CGM patterns clearly and empathetically remains time-intensive. Evidence for retrieval-grounded large language model (LLM) systems in CGM-informed counseling remains limited. To evaluate whether a retrieval-grounded LLM-based conversational agent (CA) could support patient understanding of CGM data and preparation for routine diabetes consultations. We developed a retrieval-grounded LLM-based CA for CGM interpretation and diabetes counseling support. The system generated plain-language responses while avoiding individualized therapeutic advice. Twelve CGM-informed cases were constructed from publicly available datasets. Between Oct 2025 and Feb 2026, 6 senior UK diabetes clinicians each reviewed 2 assigned cases and answered 24 questions. In a blinded multi-rater evaluation, each CA-generated and clinician-authored response was independently rated by 3 clinicians on 6 quality dimensions. Safety flags and perceived source labels were also recorded. Primary analyses used linear mixed-effects models. A total of 288 unique responses (144 CA and 144 clinician) generated 864 ratings. The CA received higher quality scores than clinician responses (mean 4.37 vs 3.58), with an estimated mean difference of 0.782 points (95% CI 0.692-0.872; P<.001). The largest differences were for empathy (1.062, 95% CI 0.948-1.177) and actionability (0.992, 95% CI 0.877-1.106). Safety flag distributions were similar, with major concerns rare in both groups (3/432, 0.7% each). Retrieval-grounded LLM systems may have value as adjunct tools for CGM review, patient education, and preconsultation preparation. However, these findings do not support autonomous therapeutic decision-making or unsupervised real-world use. |
| 2026-04-16 | [Trajectory Planning for a Multi-UAV Rigid-Payload Cascaded Transportation System Based on Enhanced Tube-RRT*](http://arxiv.org/abs/2604.15074v1) | Jianqiao Yu, Jia Li et al. | This paper presents a two-stage trajectory planning framework for a multi-UAV rigid-payload cascaded transportation system, aiming to address planning challenges in densely cluttered environments. In Stage I, an Enhanced Tube-RRT* algorithm is developed by integrating active hybrid sampling and an adaptive expansion strategy, enabling rapid generation of a safe and feasible virtual tube in environments with dense obstacles. Moreover, a trajectory smoothness cost is explicitly incorporated into the edge cost to reduce excessive turns and thereby mitigate cable-induced oscillations. Simulation results demonstrate that the proposed Enhanced Tube-RRT* achieves a higher success rate and effective sampling rate than mixed-sampling Tube-RRT* (STube-RRT*) and adaptive-extension Tube-RRT* (AETube-RRT*), while producing a shorter optimal path with a smaller cumulative turning angle. In Stage II, a convex quadratic program is formulated by considering payload translational and rotational dynamics, cable tension constraints, and collision-safety constraints, yielding a smooth, collision-free desired payload trajectory. Finally, a centralized geometric control scheme is applied to the cascaded system to validate the effectiveness and feasibility of the proposed planning framework, offering a practical solution for payload attitude maneuvering in densely cluttered environments. |
| 2026-04-16 | [Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios](http://arxiv.org/abs/2604.14986v1) | Yuting Zeng, Zhiwen Zheng et al. | Safe and efficient assistive planning for visually impaired scenarios remains challenging, since existing methods struggle with multi-objective optimization, generalization, and interpretability. In response, this paper proposes a Momentum-Constrained Hybrid Heuristic Trajectory Optimization Framework (MHHTOF). To balance multiple objectives of comfort and safety, the framework designs a Heuristic Trajectory Sampling Cluster (HTSC) with a Momentum-Constrained Trajectory Optimization (MTO), which suppresses abrupt velocity and acceleration changes. In addition, a novel residual-enhanced deep reinforcement learning (DRL) module refines candidate trajectories, advancing temporal modeling and policy generalization. Finally, a dual-stage cost modeling mechanism (DCMM) is introduced to regulate optimization, where costs in the Frenet space ensure consistency, and reward-driven adaptive weights in the Cartesian space integrate user preferences for interpretability and user-centric decision-making. Experimental results show that the proposed framework converges in nearly half the iterations of baselines and achieves lower and more stable costs. In complex dynamic scenarios, MHHTOF further demonstrates stable velocity and acceleration curves with reduced risk, confirming its advantages in robustness, safety, and efficiency. |
| 2026-04-16 | [Can LLMs Score Medical Diagnoses and Clinical Reasoning as well as Expert Panels?](http://arxiv.org/abs/2604.14892v1) | Amy Rouillard, Sitwala Mundiab et al. | Evaluating medical AI systems using expert clinician panels is costly and slow, motivating the use of large language models (LLMs) as alternative adjudicators. Here, we evaluate an LLM jury composed of three frontier AI models scoring 3333 diagnoses on 300 real-world middle-income country (MIC) hospital cases. Model performance was benchmarked against expert clinician panel and independent human re-scoring panel evaluations. Both LLM and clinician-generated diagnoses are scored across four dimensions: diagnosis, differential diagnosis, clinical reasoning and negative treatment risk. For each of these, we assess scoring difference, inter-rater agreement, scoring stability, severe safety errors and the effect of post-hoc calibration. We find that: (i) the uncalibrated LLM jury scores are systematically lower than clinician panels scores; (ii) the LLM Jury preserves ordinal agreement and exhibits better concordance with the primary expert panels than the human expert re-score panels do; (iii) the probability of severe errors is lower in \lj models compared to the human expert re-score panels; (iv) the LLM Jury shows excellent agreement with primary expert panels' rankings. We find that the LLM jury combined with AI model diagnoses can be used to identify ward diagnoses at high risk of error, enabling targeted expert review and improved panel efficiency; (v) LLM jury models show no self-preference bias. They did not score diagnoses generated by their own underlying model or models from the same vendor more (or less) favourably than those generated by other models. Finally, we demonstrate that LLM jury calibration using isotonic regression improves alignment with human expert panel evaluations. Together, these results provide compelling evidence that a calibrated, multi-model LLM jury can serve as a trustworthy and reliable proxy for expert clinician evaluation in medical AI benchmarking. |
| 2026-04-16 | [Reasoning Dynamics and the Limits of Monitoring Modality Reliance in Vision-Language Models](http://arxiv.org/abs/2604.14888v1) | Danae Sánchez Villegas, Samuel Lewis-Lim et al. | Recent advances in vision language models (VLMs) offer reasoning capabilities, yet how these unfold and integrate visual and textual information remains unclear. We analyze reasoning dynamics in 18 VLMs covering instruction-tuned and reasoning-trained models from two different model families. We track confidence over Chain-of-Thought (CoT), measure the corrective effect of reasoning, and evaluate the contribution of intermediate reasoning steps. We find that models are prone to answer inertia, in which early commitments to a prediction are reinforced, rather than revised during reasoning steps. While reasoning-trained models show stronger corrective behavior, their gains depend on modality conditions, from text-dominant to vision-only settings. Using controlled interventions with misleading textual cues, we show that models are consistently influenced by these cues even when visual evidence is sufficient, and assess whether this influence is recoverable from CoT. Although this influence can appear in the CoT, its detectability varies across models and depends on what is being monitored. Reasoning-trained models are more likely to explicitly refer to the cues, but their longer and fluent CoTs can still appear visually grounded while actually following textual cues, obscuring modality reliance. In contrast, instruction-tuned models refer to the cues less explicitly, but their shorter traces reveal inconsistencies with the visual input. Taken together, these findings indicate that CoT provides only a partial view of how different modalities drive VLM decisions, with important implications for the transparency and safety of multimodal systems. |
| 2026-04-16 | [Segment-Level Coherence for Robust Harmful Intent Probing in LLMs](http://arxiv.org/abs/2604.14865v1) | Xuanli He, Bilgehan Sel et al. | Large Language Models (LLMs) are increasingly exposed to adaptive jailbreaking, particularly in high-stakes Chemical, Biological, Radiological, and Nuclear (CBRN) domains. Although streaming probes enable real-time monitoring, they still make systematic errors. We identify a core issue: existing methods often rely on a few high-scoring tokens, leading to false alarms when sensitive CBRN terms appear in benign contexts. To address this, we introduce a streaming probing objective that requires multiple evidence tokens to consistently support a prediction, rather than relying on isolated spikes. This encourages more robust detection based on aggregated signals instead of single-token cues. At a fixed 1% false-positive rate, our method improves the true-positive rate by 35.55% relative to strong streaming baselines. We further observe substantial gains in AUROC, even when starting from near-saturated baseline performance (AUROC = 97.40%). We also show that probing Attention or MLP activations consistently outperforms residual-stream features. Finally, even when adversarial fine-tuning enables novel character-level ciphers, harmful intent remains detectable: probes developed for the base LLMs can be applied ``plug-and-play'' to these obfuscated attacks, achieving an AUROC of over 98.85%. |
| 2026-04-16 | [Benchmarks for Trajectory Safety Evaluation and Diagnosis in OpenClaw and Codex: ATBench-Claw and ATBench-CodeX](http://arxiv.org/abs/2604.14858v1) | Zhonghao Yang, Yu Li et al. | As agent systems move into increasingly diverse execution settings, trajectory-level safety evaluation and diagnosis require benchmarks that evolve with them. ATBench is a diverse and realistic agent trajectory benchmark for safety evaluation and diagnosis. This report presents ATBench-Claw and ATBench-CodeX, two domain-customized extensions that carry ATBench into the OpenClaw and OpenAI Codex / Codex-runtime settings. The key adaptation mechanism is to analyze each new setting, customize the three-dimensional Safety Taxonomy over risk source, failure mode, and real-world harm, and then use that customized taxonomy to define the benchmark specification consumed by the shared ATBench construction pipeline. This extensibility matters because agent frameworks remain relatively stable at the architectural level even as their concrete execution settings, tool ecosystems, and product capabilities evolve quickly. Concretely, ATBench-Claw targets OpenClaw-sensitive execution chains over tools, skills, sessions, and external actions, while ATBench-CodeX targets trajectories in the OpenAI Codex / Codex-runtime setting over repositories, shells, patches, dependencies, approvals, and runtime policy boundaries. Our emphasis therefore falls on taxonomy customization, domain-specific risk coverage, and benchmark design under a shared ATBench generation framework. |
| 2026-04-16 | [Switch: Learning Agile Skills Switching for Humanoid Robots](http://arxiv.org/abs/2604.14834v1) | Yuen-Fui Lau, Qihan Zhao et al. | Recent advancements in whole-body control through deep reinforcement learning have enabled humanoid robots to achieve remarkable progress in real-world chal lenging locomotion skills. However, existing approaches often struggle with flexible transitions between distinct skills, cre ating safety concerns and practical limitations. To address this challenge, we introduce a hierarchical multi-skill system, Switch, enabling seamless skill transitions at any moment. Our approach comprises three key components: (1) a Skill Graph (SG) that establishes potential cross-skill transitions based on kinematic similarity within multi-skill motion data, (2) a whole-body tracking policy trained on this skill graph through deep reinforcement learning, and (3) an online skill scheduler to drive the tracking policy for robust skill execution and smooth transitions. For skill switching or significant tracking deviations, the scheduler performs online graph search to find the optimal feasible path, which ensures efficient, stable, and real-time execution of diverse locomotion skills. Comprehensive experiments demonstrate that Switch empowers humanoid to execute agile skill transitions with high success rates while maintaining strong motion imitation performance. |
| 2026-04-16 | [Beyond Literal Summarization: Redefining Hallucination for Medical SOAP Note Evaluation](http://arxiv.org/abs/2604.14829v1) | Bhavik Vachhani, Kush Shrisvastava et al. | Evaluating large language models (LLMs) for clinical documentation tasks such as SOAP note generation remains challenging. Unlike standard summarization, these tasks require clinical abstraction, normalization of colloquial language, and medically grounded inference. However, prevailing evaluation methods including automated metrics and LLM as judge frameworks rely on lexical faithfulness, often labeling any information not explicitly present in the transcript as hallucination.   We show that such approaches systematically misclassify clinically valid outputs as errors, inflating hallucination rates and distorting model assessment. Our analysis reveals that many flagged hallucinations correspond to legitimate clinical transformations, including synonym mapping, abstraction of examination findings, diagnostic inference, and guideline consistent care planning.   By aligning evaluation criteria with clinical reasoning through calibrated prompting and retrieval grounded in medical ontologies we observe a significant shift in outcomes. Under a lexical evaluation regime, the mean hallucination rate is 35%, heavily penalizing valid reasoning. With inference aware evaluation, this drops to 9%, with remaining cases reflecting genuine safety concerns. These findings suggest that current evaluation practices over penalize valid clinical reasoning and may measure artifacts of evaluation design rather than true errors, underscoring the need for clinically informed evaluation in high context domains like medicine. |
| 2026-04-16 | [CBF-based Probabilistic Safe Navigation under Unknown Nonlinear Obstacle Dynamics](http://arxiv.org/abs/2604.14818v1) | Jiwon Lee, Hugo Matias et al. | Safe navigation for an ego vehicle in uncertain environments characterized by dynamic obstacles with unknown nonlinear dynamics is a challenging problem of significant practical interest. Existing approaches in the literature either lack formal safety guarantees, require full model knowledge, or fail to account for the risk associated with the vehicle's exact body geometry and the temporal evolution of uncertainty between sampling instants. In this paper, we propose a data-driven observer for the unknown obstacle dynamics that generates an alpha-confidence set flow, which is exactly transformed into a Control Barrier Function (CBF) to enforce (1-alpha)-probability safety. The proposed framework accommodates nonlinear ego vehicle dynamics of arbitrary relative degree, as demonstrated through case studies involving first- and second-order dynamics of an unmanned surface vehicle. |

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



