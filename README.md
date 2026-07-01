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
| 2026-06-30 | [Freeform Preference Learning for Robotic Manipulation](http://arxiv.org/abs/2606.32027v1) | Marcel Torne, Anubha Mahajan et al. | Reward design remains a central bottleneck for autonomous robot policy improvement, especially in long-horizon manipulation tasks where sparse success labels provide too little signal and binary preferences collapse many competing notions of quality into one ambiguous signal. We introduce Freeform Preference Learning (FPL), a method for learning robot policies from freeform human preferences. Rather than asking annotators which of two trajectories is better overall, FPL lets them define natural-language preference axes, such as speed, safety, quality of placement, or carefulness, and provide pairwise preferences along each axis. These annotations are used to learn a language-conditioned reward model that maps a trajectory and preference label to an axis-specific reward. We use this model to train a reward-conditioned policy that optimizes across the multiple human-specified dimensions. Across four real-world and two simulated long-horizon manipulation tasks, FPL improves over sparse-reward and binary-preference methods by 38 percentage points. Beyond improved performance, FPL learns dense progress signals without explicit subtask segmentation, shows compositionality of behavior not present in the data, and allows users to steer the policy towards different behaviors at test time without retraining. Blog post with videos available at https://freeform-pl.github.io/fpl.website/ |
| 2026-06-30 | [The filter exchange system of the LSSTCam at the Vera C. Rubin Observatory](http://arxiv.org/abs/2606.31995v1) | Alexandre Boucaud, Pierre Antilogus et al. | The Filter Exchange System of the LSSTCam at the Vera C. Rubin Observatory is a critical subsystem enabling the Legacy Survey of Space and Time (LSST) by performing rapid, repeatable exchanges among five large-format filters within a highly constrained in-camera volume. Since the start of on-sky operations in April 2025, the FES has routinely performed up to 40 filter exchanges per night, completing each change in under 90 seconds with a positioning repeatability of 100 micrometers in the focal plane. Safety and reliability are ensured through a dedicated software architecture. Drawing on over a year of operational experience, we report on the in-situ performance of this sophisticated system within the observatory environment, characterize the key performance metrics, and discuss how specific design choices have influenced system behavior and reliability in practice. |
| 2026-06-30 | [OopsieVerse: A Safety Benchmark with Damage-Aware Simulation for Robot Manipulation](http://arxiv.org/abs/2606.31993v1) | Arnav Balaji, Arpit Bahety et al. | While robotic manipulation capabilities have advanced rapidly, physical safety remains a major barrier to deploying household robots: task success is insufficient if the robot damages itself or its surroundings. Simulation offers a harm-free alternative to costly and dangerous real-world training and evaluation, yet existing simulators lack general mechanisms to detect, quantify, and represent damage. To address this gap, we introduce OOPSIEVERSE, a unified simulation framework and benchmark for damage-aware household manipulation. OOPSIEVERSE provides damage as an explicit, physically-grounded, and taskagnostic signal by converting sources such as contact forces, temperature changes, and liquid interactions into corresponding mechanical, thermal or fluid damage. OOPSIEVERSE comprises two core elements: (1) DAMAGESIM, a simulator-agnostic framework for detecting and quantifying damage during navigation and manipulation, and (2) a suite of household tasks designed to evaluate common damage modes and distinguish between task completion and safe execution. We demonstrate the generality of our framework by instantiating DAMAGESIM in two simulators with different physics backends, OmniGibson (Nvidia Omniverse) and RoboCasa (MuJoCo). We further showcase the utility of OOPSIEVERSE across multiple use cases, including (1) guiding safer demonstration collection via real-time damage feedback, (2) learning safer manipulation policies through damage-conditioned imitation learning and reinforcement learning, (3) benchmarking the safety of state-of-the-art Vision Language Action policies, and (4) improving real-world safety of sim-to-real transferred policies. Together, our results highlight the potential of OOPSIEVERSE as an open-source foundation for systematic, scalable research on safe robot manipulation. For code and more information, please refer to https://robin-lab.cs.utexas.edu/oopsieverse/ |
| 2026-06-30 | [Delegation Rights: Property, Agency, and Investment Incentives in the Age of AI Agents](http://arxiv.org/abs/2606.31935v1) | Yukun Zhang, Kemu Xu | AI agents increasingly operate inside digital accounts by exercising privileges that users already hold, raising a new control question: whether an existing account entitlement must be exercised manually or may be exercised through a user-authorized automated proxy. We define \emph{delegation rights} as the revocable, identity-preserving, scope-limited, and mode-specific authority of an account holder to authorize such proxy execution. We develop a three-party incomplete-contracts model with a User, an AI Agent provider, and a Platform. The contested object is not platform ownership, account transferability, data portability, or unrestricted API access, but residual control over the mode of account execution. Under Platform Control, the platform can protect infrastructure, identity systems, privacy boundaries, and third parties, but its discretionary veto weakens the User--Agent coalition's disagreement payoff and depresses relationship-specific investment. Under User Control, hold-up is reduced, but security, privacy, congestion, and third-party risks may remain insufficiently internalized. We then analyze \emph{Certified Delegation}, under which access protection is conditional on verifiable authorization, revocability, auditability, rate-limit compliance, data minimization, and risk mitigation. Certification is therefore not merely a technical safety screen; it is a conditional allocation of residual control. Illustrative mechanism simulations show how this regime can reduce deadweight loss by restoring delegation incentives while bounding residual risk. |
| 2026-06-30 | [Theory of Mind and Persuasion Beyond Conversation: Assessing the Capacity of LLMs to Induce Belief States via Planning and Action](http://arxiv.org/abs/2606.31916v1) | Ben Slater, Matteo G. Mecattaf et al. | Theory of Mind (ToM) benchmarks for Large Language Models (LLMs) typically rely on passive question-answering formats, but the deployment of LLMs in increasingly agentic and autonomous forms demands new evaluations. In this paper we evaluate an agent's ability to induce specific belief states in other agents by taking actions rather than using conversational persuasion, a capability we call Non-Conversational Planning ToM (NCP-ToM). NCP-ToM is likely to be essential for many agent use-cases, including within user-assistant interactions and pedagogical contexts, but may also present manipulation or misinformation risks. Using a novel framework, NCP-ExploreToM, we subvert the conventional task structure by providing models with a set of belief state goals and requiring them to move objects or direct characters into rooms to achieve their goals. We evaluated six frontier models, including GPT-5, Gemini 2.5 Pro and the Claude 4 series, and a cohort of human participants, across 600 task instances. GPT-5 was successful on approximately 80% of tasks in the agentic setting, and was the only model to outperform human participants on our task, but was still less robust than humans across contexts. We additionally found that all models, like humans, performed better on tasks inducing true belief states than false belief states, which is a positive signal for alignment efforts. These findings highlight emerging social-reasoning capabilities in LLMs for non-conversational task completion and underscore the necessity of agentic evaluations for understanding the safety and alignment of autonomous social agents. |
| 2026-06-30 | [Harnessing Textual Refusal Directions for Multimodal Safety](http://arxiv.org/abs/2606.31876v1) | Moreno D'Incà, Massimiliano Mancini et al. | To improve safety in Large Language Models (LLMs) we can either perform post-training alignment or exploit refusal directions in the activation space. Both strategies are less feasible in Multimodal LLMs (MLLMs) as they require unsafe multimodal data, harder to collect than their unimodal counterpart. In this work, we relax this constraint and investigate whether textual refusal directions, extracted directly from the LLM backbone, generalize across modalities (i.e., image, video). Preliminary findings confirm this ability, though effectiveness is conditioned by layer selection, steering strength, and cross-modal alignment, with the latter causing safe multimodal inputs to be spuriously steered toward refusal. Building on this, we introduce Modality-Agnostic Refusal Steering (MARS), a light-weight training-free approach that injects multimodal safety without the need for multimodal safety data. MARS corrects modality misalignment via activation re-centering, adaptively scales steering strength within a geometrically defined trust region, and selects the optimal intervention layer, operating at the first generated token. Evaluated on five SOTA MLLMs across safety, utility, and video jailbreak benchmarks, MARS achieves consistent safety gains while preserving utility. These results reveal that safety-relevant structure is shared across modalities and that textual refusal directions are a powerful and underexplored foundation for multimodal alignment. |
| 2026-06-30 | [Determining stress-based bending mode limits for the Vera C. Rubin Observatory M1M3 active mirror system](http://arxiv.org/abs/2606.31849v1) | Malhar Sonaniskar, Douglas Neill et al. | The Vera C. Rubin Observatory Simonyi Survey Telescope's primary-tertiary mirror (M1M3) is an actively supported, 8.4-m cast borosilicate optic controlled by 156 pneumatic actuators. This work presents a rapid stress-estimation methodology based on the root-sum-square (RSS) combination of Finite Element Analysis to derive pre-computed unit bending mode stresses. Since the stress is proportional to strain, and strain is proportional to displacements, we theorized that since the bending mode displacements can be combined RSS, that the peak stresses would also combine by RSS. We validate the RSS-based major principal stress predictions against NASTRAN simulations for representative bending mode combinations, demonstrating agreement within a few percent for peak Principal major stress across the mirror glass substrate. Unit displacement and corresponding unit stress fields for the first 20 natural bending modes of the M1M3 system are generated using NASTRAN. Representative multi-mode corrections including combinations that include astigmatism, coma, and spherical modes of higher order are then analyzed to compare the resulting peak principal stresses with RSS-based predictions. The method enables near instantaneous evaluation of stress margins for active optics corrections, safety-limit checking, and actuator-force optimization during telescope operations. This paper outlines the formulation, implementation workflow, validation results, and practical use cases for integrating RSS-based stress prediction into the Vera C. Rubin Observatory's M1M3 active optics system. |
| 2026-06-30 | [The On-Sky Performance of the LSST Camera CCD Array](http://arxiv.org/abs/2606.31806v1) | Sean Patrick MacBride, Aaron Roodman et al. | The focal plane of the LSST Camera contains 189 individual science CCDs, arranged into 21 raft tower modules, along with 4 wavefront and 8 guider CCDs located in 4 additional corner RTMs. Altogether, the LSST Camera CCDs compose the largest focal plane ever constructed. The LSST Camera is the primary instrument of Rubin Observatory, which will begin the Legacy Survey of Space and Time in 2026. In this paper, we describe the on-sky performance of the LSST Camera CCDs, from receipt at NSF/DOE Vera C. Rubin Observatory in May 2024 to on-sky observations during the first year of operations. We discuss the process to establish functionality of several CCDs which were affected by an electrical short and faulty analog-digital converter, optimizations of readout timing in response to changes in the survey strategy, and implementation of enhanced focal plane safety measures through an active clearing mechanism on the CCDs. Finally, we discuss sensor features observed on-sky, and global performance during the first year of operations. The operations to date of the LSST Camera CCDs have demonstrated the capability of performing a wide, fast, and deep optical imaging survey of the entire southern sky at the Rubin Observatory. |
| 2026-06-30 | [Relational and Sequential Conformal Inference for Energy Time Series over Graphs via Foundation Models](http://arxiv.org/abs/2606.31804v1) | Keivan Faghih Niresi, Alice Cicirello et al. | Accurate energy demand forecasting is essential for the reliable operation and planning of modern sustainable energy systems. Spatial-temporal graph neural networks (STGNNs) have recently achieved strong performance in point forecasting by jointly modeling temporal dynamics and relational dependencies across interconnected energy nodes. However, in real-world energy systems, accurate point forecasts alone are insufficient, as operators also require reliable uncertainty estimates to support risk-aware decision-making, grid stability, and operational planning under uncertainty. Conformal prediction provides a principled and model-agnostic framework for uncertainty quantification with statistical coverage guarantees, making it particularly attractive for safety-critical energy applications. However, existing conformal prediction approaches often fail to fully capture the complex spatial-temporal structure of energy systems. To address these limitations, we propose STOIC (Spatial-Temporal Graph Conformal Prediction with In-Context Learning), a novel framework that integrates graph-based forecasting with the zero-shot calibration capabilities of tabular foundation models. STOIC first generates point forecasts using an STGNN and subsequently reformulates spatial-temporal residuals into a tabular representation suitable for in-context learning. Leveraging a tabular foundation model, STOIC calibrates prediction intervals without task-specific retraining, effectively capturing both sequential and relational dependencies. We evaluate STOIC on five diverse benchmarks, including synthetic simulations as well as real-world electricity and district heating networks. Across all datasets, STOIC consistently outperforms existing conformal prediction baselines, delivering more reliable and robust uncertainty estimates for complex graph-structured energy time series. |
| 2026-06-30 | [Context-Verified, Error-Budget-Aware Decomposition Selection for Toffoli Networks](http://arxiv.org/abs/2606.31791v1) | Karol Bartkiewicz, Patrycja Tulewicz | Two-qubit-gate error dominates the failure budget of near-term quantum circuits, so the decomposition chosen for each Toffoli (CCX) gate should minimize hardware two-qubit infidelity, not gate count. The cheapest decompositions - relative-phase and approximate Toffolis - are only correct in context: their residual phase or bounded error must be cancelled or absorbed downstream. We present the first compiler pass that selects a per-Toffoli decomposition to minimize a two-qubit-infidelity error budget. It admits each context-dependent decomposition only when an exact, instance-specific equivalence check certifies its validity in that circuit context, coupling an error-budget objective with per-instance verification and closing the gap between context-aware-but-unverified and verified-but-context-free optimizers. The central result is a safety one: pattern-matched relative-phase substitution is silently incorrect. Our verifier flags 66 library rewrites of a deployed open optimizer as non-equivalent without a context check, and count-greedy substitution silently corrupts 6 of 12 benchmark circuits; the verification gate certifies 0 errors while still applying every valid decomposition. The two-qubit-gate reduction is real but workload-dependent: up to 39.5% fewer two-qubit gates and 36.7% lower infidelity over exact-only on a compute/uncompute-heavy suite (approx. 39%/35% versus Qiskit opt-3 and tket), and 15.6% aggregate on a larger 12-24-qubit suite, with decision-diagram checking certifying every substitution past the exhaustive-verification limit. At current superconducting and trapped-ion error rates, the certified substitutions lower estimated circuit infidelity by 36-43%, and on a quantum state-resetting circuit, the pass removes 48.8% of the native two-qubit gates, every substitution verified. |
| 2026-06-30 | [Investigating LLM-Powered Dissenting Minority Support in Power-Imbalanced Group Decision-Making: Counterargument and Mediation as Intervention Strategies](http://arxiv.org/abs/2606.31762v1) | Soohwan Lee, Seoyeong Hwang et al. | Minority viewpoints are often suppressed in power-imbalanced group decision-making due to social pressure to comply with the majority. To address this problem, we developed an LLM-powered dissenting minority support system that aimed to foster attention to minority views through either AI-generated counterarguments or AI-mediated messages. We conducted a mixed-method experiment with 96 participants in 24 groups, comparing minority members' experiences across baseline, AI-counterargument, and AI-mediated message conditions. Our findings revealed a nuanced trade-off: AI-generated counterarguments fostered a more flexible atmosphere and enhanced satisfaction, while AI-mediated messaging increased minority participation but unexpectedly reduced their psychological safety. This research contributes empirical evidence on how different AI implementations affect group dynamics, identifies a critical support paradox between participation and psychological safety, provides design implications for future systems, and highlights ethical challenges in implementing AI-mediated communication in hierarchical settings. These insights advance understanding of designing more equitable AI support for power-imbalanced group decision-making. |
| 2026-06-30 | [Addressing Over-Refusal in LLMs with Competing Rewards](http://arxiv.org/abs/2606.31748v1) | Taeyoun Kim, Aviral Kumar | Safety training on language models often induces over-refusal: improved safety on harmful prompts at the cost of increased refusal on harmless ones. Though this trade-off can be mitigated by training models with reinforcement learning (RL) to reason before answering, it does not remove the underlying problem that reasoning can often be a "rubber stamp" for a predetermined response. In this paper, we address the safety-refusal trade-off by rethinking how models are trained to reason about safety. Our key insight is that unsafe reasoning can itself serve as a useful exploratory signal. Rather than preemptively blocking harmful thoughts, we encourage the model to sufficiently explore unsafe reasoning but produce a safe response. The harmful exploration improves the model's ability to distinguish harmful from harmless prompts by resolving ambiguity, allowing it to remain safe while complying only when appropriate. We cast this as an adversarial optimization problem in which a reasoning player explores strategies for producing an unsafe response and an answer player ensures that the final output is safe. We train a single model with dense rewards to play both roles within one chain-of-thought, across different segments. To achieve this, we find that process rewards are crucial for stable optimization of competing objectives. Our resulting model SEAR deliberately engages in harmful reasoning as exploration while reliably flipping back to a safe answer. We demonstrate that this behavior helps mitigate over-refusal and defend against attacks that directly manipulate the reasoning to be harmful. |
| 2026-06-30 | [Electric Field Attenuation Techniques for Inductive Wireless Charging of Medical Implants](http://arxiv.org/abs/2606.31739v1) | Sam Boeckx, Pieterjan Polfliet et al. | Inductive wireless charging of implantable medical devices necessitates careful control of magnetic and electric field emissions to meet strict safety regulations while delivering sufficient power. When designing a comfortable wireless charger that can operate over distances ranging to 10cm or more, it is difficult not to exceed the most stringent E-field limit of 83~V/m. This paper investigates electric field attenuation techniques for mid-range wireless power transfer at 6.78~MHz. Using \newacronym{fea}{FEA}{finite element analysis}\acrfull{fea} like Ansys \textregistered{} HFSS \texttrademark{}, three mitigation strategies are evaluated; (1) a high-permittivity dielectric shielding layer to absorb and redistribute electric fields, (2) multiple resonant tuning capacitors distributed along the transmitter coil to lower the voltage swing and confine high E-field regions, and (3) alternative coil-array transmitter topologies to spatially localize more confined E-fields. The results show that each technique significantly reduces the E-field magnitude without substantially affecting the H-field. Shielding the transmit coil attenuates the peak E-field from its initial 1416~V/m to 496~V/m, approximately a 65\% reduction. Distributing the tuning capacitance into sixteen smaller capacitors yields a drop from the 1416~V/m to 231~V/m, approximately a 84\% reduction. Both techniques preserve the required 8~A/m magnetic field. The third technique, a two-by-two coil array transmitter reduced the E-field from its 1416~V/m to 990~V/m (around 30\% reduction), though with a slight magnetic field redistribution. All three methods combined, the E-field was successfully attenuated to 82~V/m, just below the strictest limit, without compromising power transfer efficiency. This research demonstrates a feasible approach and framework to safely extend the application of wireless charging for medical implants. |
| 2026-06-30 | [Optimizing Ti substitution for the enhanced densification, ionic conductivity, and microstructure of garnet-type Li$_7$La$_3$Zr$_2$O$_{12}$ solid electrolytes](http://arxiv.org/abs/2606.31669v1) |  Neha, A. V. Deshpande et al. | Garnet-type lithium lanthanum zirconium oxide Li$_7$La$_3$Zr$_2$O$_{12}$ (LLZO) is a favorable solid electrolyte for all-solid-state Li-ion batteries due to its wide electrochemical stability, compatible ionic conductivity, and good safety. However, further improvement in ionic conductivity is required for its practical applications. In this work, titanium (Ti) is doped into LLZO to enhance its Li-ion transport properties and structural stability. The series Li$_7$La$_3$Zr$_{2-x}$Ti$_x$O$_{12}$ has been successfully synthesized using conventional solid-state reaction method. The content of Ti has been varied from 0 to 0.20 atoms per formula unit (a.p.f.u). The conducting cubic phase has been confirmed by the X-ray diffraction technique (XRD). Scanning electron microscopy (SEM) and energy dispersive spectroscopy (EDS) have been used for structural analysis, and elemental distribution. Density measurements have been carried out for all the samples. Electrochemical impedance spectroscopy revealed that the high ionic conductivity of $8.08\times 10^{-5}$ Scm$^{-1}$ is offered by the Li$_7$La$_3$Zr$_{1.9}$Ti$_{0.1}$O$_{12}$ sample, which has the lowest activation energy of 0.37 eV. The DC polarization analysis verified that the main contribution to conductivity in the 0.10 Ti sample comes from ions. A one order of magnitude increase in room temperature ionic conductivity is observed for the 0.10 Ti sample, making it a strong candidate for solid electrolyte applications. |
| 2026-06-30 | [Near-Optimal Nitrogen Recommendations for Precision Agriculture via Sequential Screening and Hierarchical Refinement](http://arxiv.org/abs/2606.31661v1) | Sakshi Arya, Abdul-Nasah Soale et al. | Nitrogen fertilizer management plays a central role in balancing agricultural productivity and environmental sustainability, yet identifying optimal application strategies remains difficult because treatment responses vary substantially across locations and many fertilizer choices are statistically indistinguishable near the optimum. This paper develops a hierarchical refinement procedure, built on sequential screening, for fertilizer recommendation in multi-site experiments that explicitly accounts for spatial heterogeneity while prioritizing parsimonious, decision-oriented selection. Rather than targeting a single estimated best treatment, the proposed method first conducts sequential screening at a higher aggregation level to eliminate clearly inferior fertilizer choices and then refines recommendations locally among the surviving candidates. We study the asymptotic properties of the proposed estimators and show that it provides screening-safety guaranteed recommendations. The efficacy of the new approach is investigated through a multi-state, multi-year corn nitrogen trial. The results show that no single fertilizer regime is uniformly optimal within a state; instead, each state is associated with multiple recommended choices, and the most common recommendation typically covers only about one-third to one-half of decision units, underscoring substantial within-state heterogeneity. Representative site-level comparisons further demonstrate that the proposed method often yields lower total nitrogen recommendations than state-level or hindsight benchmarks while maintaining competitive agronomic performance. |
| 2026-06-30 | [Moral Safety in LLMs: Exposing Performative Compliance with Puzzled Cues](http://arxiv.org/abs/2606.31644v1) | Mohammadamin Shafiei, Shuyue Stella Li et al. | As large language models take on morally consequential roles in healthcare, legal, and hiring contexts, we need to examine whether their ethical behaviors are genuine or superficial. We show that current fairness evaluations substantially overestimate moral safety. Models appear fair when demographic identity is stated as an explicit label, yet become measurably less fair when the same identity must be inferred. We term this failure \emph{performative compliance}, where a model is fair when the presentation resembles a fairness evaluation and less fair as that cue weakens. We introduce a cue-variation methodology that holds the moral dilemma and the demographic identity fixed and varies only how that identity is conveyed. Hiding the explicit label raises harmful decisions by $+4.4$~pp and changes model safety rankings, and the shift persists when models correctly infer the demographic, ruling out attribution error. We propose the \textbf{Cue Visibility Gap}, a model-agnostic robustness metric that can be added to any existing fairness benchmark to separate genuine from performative moral safety. Fairness evaluations that omit cue variation measure surface compliance, not moral robustness, and should not ground deployment decisions in high-stakes settings. |
| 2026-06-30 | [A Lifecycle and Application-Stack Survey of Large Language Model Vulnerabilities: Attacks, Risks, Defenses, and Open Problems](http://arxiv.org/abs/2606.31639v1) | Seyed Bagher Hashemi Natanzi, Bo Tang | Large language models are no longer only text generators. They are increasingly embedded in retrieval pipelines, enterprise assistants, coding environments, robotic systems, security-operation workflows, and autonomous agents that can read private data, call tools, write files, execute code, and act across organizational boundaries. This shift changes the security problem: risks do not arise from the model weights alone, but from the full lifecycle and application stack through which data, prompts, model outputs, tools, memories, and user authority interact. This paper systematizes the literature on vulnerabilities in large language model systems through a lifecycle and application-stack lens. We organize attacks across eight stages: data collection, pretraining, post-training alignment, model packaging and supply chain, retrieval and memory, prompting and inference, tool/agent execution, and deployment/maintenance. For each stage, we analyze attacker capabilities, affected security objectives, representative attacks, practical risks, evaluation practices, and defenses. We further map LLM-specific vulnerabilities to confidentiality, integrity, availability, safety, privacy, fairness, accountability, and agency-control objectives. Unlike taxonomies that list isolated attack names, the proposed systematization emphasizes where trust boundaries fail, how untrusted data becomes executable instruction, how delegated authority amplifies model errors, and why point defenses rarely compose. We close with a research agenda for secure LLM systems, including compositional security, provenance-aware retrieval, tool-call containment, long-horizon agent evaluation, privacy-preserving adaptation, realistic red teaming, and deployment-grade incident response. |
| 2026-06-30 | [A Tutorial on Autonomous Fault-Tolerant Control Using Knowledge-Grounded LLM Agents](http://arxiv.org/abs/2606.31635v1) | Javal Vyas, Milapji Singh Gill et al. | Fault recovery in process plants still relies heavily on plant operators, especially when faults fall outside predefined supervisory logic. Operators interpret alarms, procedures, P\&IDs, interlocks, and process trends, then decide how to move the plant to a safe operating mode without triggering a shutdown. This paper examines how Large Language Model (LLM) agents can support such recovery decisions. The proposed framework treats the LLM as a constrained supervisory planner. It uses plant-specific knowledge to propose recovery actions, and every proposal is checked by an external validator (symbolic or simulation-based) before actuation. The paper develops three design dimensions for applying the framework: the recovery patterns for which LLM agents are useful, the validation strategies that separate admissible from inadmissible proposals, and the deployment constraints imposed by latency, knowledge engineering, safety integration, and model lifecycle management. To make the framework directly usable, two openly available executable Python environments are provided. Both re-implement established case studies, a modular mixing module and a continuous stirred-tank reactor, extended with configurable faults and defined interfaces for custom recovery and validation methods. |
| 2026-06-30 | [Automating Cause-Effect Specification with Knowledge Graphs and Large Language Models](http://arxiv.org/abs/2606.31614v1) | Javal Vyas, Milapji Singh Gill et al. | Engineering specifications such as interlocks, alarm rationalization tables, and cause-and-effect (C&E) matrices remain central to process control and safety, yet their creation is still predominantly manual, document-driven, and prone to inconsistency. This paper presents a semantic-AI framework that automates the generation of C&E logic by combining a knowledge graph (KG) with a constrained large language model (LLM) layer. The KG builds on an established modular alignment ontology to represent process structure, operating modes, faults, symptoms, causes, and mitigation actions in a machine-interpretable form. The LLM then transforms this information into operator-ready safety narratives and Semantic Web Rule Language (SWRL) rules under strict ontology and vocabulary constraints, grounding the generated artifacts in the underlying semantic model. The workflow is demonstrated on a modular process plant, showing how engineering semantics, diagnostic relations, and machine-verifiable specifications can be generated from a unified knowledge representation with reduced manual effort. |
| 2026-06-30 | [CLExEval: A Human-in-the-Loop Framework for Qualitative Evaluation of LLM Clinical Reasoning](http://arxiv.org/abs/2606.31608v1) | Ajmal M., Abin Roy et al. | Large Language Models (LLMs) achieve strong results on many medical benchmarks, but their clinical reasoning remains difficult to evaluate reliably. A central risk is an evaluation illusion: fluent and well-structured explanations can appear clinically convincing even when the final diagnosis is incorrect. We introduce CLExEval, a human-in-the-loop framework for evaluating LLM clinical reasoning under progressive information masking. CLExEval combines 5,600 expert-physician annotations with 200 clinical reasoning traces derived from 40 rare diagnostic cases. Our analysis identifies three recurring failure patterns: (i) verbosity bias, where GPT-4o-mini's diagnostic accuracy drops from 95.0% to 32.5% under information scarcity; (ii) a hidden knowledge paradox, where a specialist model reaches 92.5% maximum diagnostic potential but fails to retrieve that knowledge reliably in verbose contexts; and (iii) a 68.6% reasoning-to-output mismatch, where correct diagnoses appear in reasoning traces but are not reflected in final answers. We further evaluate the LLM-as-a-Judge paradigm on a human-verified failure set (n = 142). GPT-4o-mini approved 47.9% of clinically incorrect outputs, while HuatuoGPT-o1 approved all validly scored failures and showed a positive self-preference bias. These results suggest that standalone automated clinical evaluations can substantially overestimate clinical reliability without expert-grounded validation. |

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



