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
| 2026-09-24 | [Molecular Beam Epitaxy of AgTaO3](http://arxiv.org/abs/2609.30229v1) | Tobias Schwaigert, Joshua Maile et al. | We report the first synthesis of single-crystal AgTaO3 thin films using molecular-beam epitaxy (MBE). High-quality epitaxial AgTaO3 films were grown on both (001)- and (111)-oriented SrTiO3 substrates using sequential deposition of atomic silver and TaO2 layers under an ozone/oxygen atmosphere (80 % O3 + 20 % O2). X-ray diffrac- tion and reciprocal space mapping demonstrate that the films are coherently strained to the SrTiO3 substrates with sharp rocking curves comparable to the substrates, indicating high structural perfection. High-angle annular dark-field scanning trans- mission electron microscopy (HAADF-STEM) confirms coherent, low defect growth for (001)pc-oriented films. For (111)pc-oriented films, initial coherent growth proceeds up to 10 nm before transitioning into a Ta-rich surface region. Energy-dispersive X-ray spectroscopy (EDX) reveals a narrow cation intermixing region at the sub- strate interface for both orientations. This work demonstrates an effective synthesis route for single-crystal AgTaO3 thin films, providing a platform to investigate strain engineering and emergent interfacial properties in silver-based tantalates. |
| 2026-09-24 | [Instrumental Monitor Evasion Emerges Under Ordinary Task Pressure](http://arxiv.org/abs/2609.30217v1) | David Schmotz, Derck Prinzhorn et al. | A central concern in AI safety is that agents may treat oversight as an obstacle when it conflicts with completing their goals. We study instrumental evasion, the propensity of LLM agents to circumvent runtime monitoring as a means of completing ordinary tasks. We introduce EvasionBench, a benchmark of 50 diverse task-policy pairs in which completing the task requires an operation prohibited by a runtime monitor. Agents know that their tool calls are monitored and are prompted to continue working when they pause. Across our evaluations, best-of-3 evasion attempt rates reach up to 98% and success rates up to 88%, with substantial variance across models. Claude Fable 5.1 succeeds less often, but frequently makes creative attempts to circumvent the monitor. Evasion generally increases with test-time compute, with higher evasion rates at greater reasoning effort and token use. Traces show that agents encode prohibited commands, decompose operations across tool calls, and retry until relevant context leaves the monitor's history. We also find that GPT-6 Astra's low evasion rate comes with overrefusal, as it frequently abandons otherwise solvable tasks under a denial-of-service prompt injection. Our findings show that ordinary task pressure can lead to adaptive attempts to evade runtime monitors without an explicit adversarial objective. Effective oversight must therefore remain robust against repeated attempts, as the persistence that helps agents solve difficult tasks can also drive them to circumvent their guardrails. |
| 2026-09-24 | [A Living Benchmark for Information Retrieval from Electronic Health Records](http://arxiv.org/abs/2609.30205v1) | Jordan L. Cahoon, Chloe O. Stanwyck et al. | Large language model (LLM)-based clinical assistants are increasingly being integrated into electronic health record (EHR) systems, transforming how clinicians retrieve and synthesize information from patient records. Their safety and utility depend on rigorous evaluation, yet existing benchmarks are manually curated, costly to update, and rapidly become obsolete with evolving technological advancements. We present a scalable framework that automatically generates question--answer pairs from longitudinal EHR notes. Nineteen clinicians validate the benchmark generator, producing the Benchmark for Retrieving Information in EHRs (BRIE), a continuously maintainable evaluation dataset. Across nine LLMs and five inference strategies, state-of-the-art systems frequently omit clinically important information, particularly for questions requiring synthesis across multiple documents and encounters. Because the generator itself is validated, BRIE supports evaluations that static benchmarks cannot, including the generation of multiple answers that reflect variation in clinician reasoning for robust performance assessment and continuously refreshing benchmark content to guard against leakage. Our results demonstrate that scalable benchmark generation enables rigorous, up-to-date evaluation of clinical LLMs as they are deployed in rapidly evolving healthcare settings. |
| 2026-09-24 | [Learning the Maximum Tolerated Dose for Continuous Toxicity via Monotone Bayesian Trees](http://arxiv.org/abs/2609.30190v1) | Se Yoon Lee | Phase I cancer trials seek the maximum tolerated dose (MTD) while protecting patients from excessive toxicity. Dose assignments must therefore balance patient safety with learning the dose--toxicity relationship as data accrue. We model continuously measured toxicity outcomes using two forms of Bayesian additive regression trees (BART): isotonic BART projects posterior response curves onto nondecreasing functions, whereas monotone BART constrains the model. Joint curve and variance draws induce an MTD posterior that guides dose selection through escalation with overdose control (EWOC). We compare these methods with three parametric procedures across seven dose--toxicity curves in simulation. We assess dose-limiting toxicity (DLT) counts, above-MTD assignments, signed last-dose error, and relative absolute error. The tree procedures attained the lowest mean RAE on four nonlinear curves and jointly minimized mean DLT counts and above-MTD assignments on four curves. A Bayesian reinforcement learning perspective formulates these sequential decisions as a finite-horizon planning problem. Dose restrictions yield a lower bound on last-dose error; under exact posterior-predictive evaluation, an EWOC-based rollout policy has no greater expected weighted loss than its baseline. Dose Trial Lab, a desktop simulator for all five procedures, accompanies the supplementary material. |
| 2026-09-24 | [Smartphone-Based Method for Automated Speed Enforcement](http://arxiv.org/abs/2609.30107v1) | Keya Li, Jahnavi Malagavalli et al. | Smartphone cameras and computer vision (CV) hold significant promise in assisting public agencies with enforcing traffic laws and enhancing road safety. This work designs and tests a smartphone-based method for automated speed estimation and vehicle identification (license plate, make/model, and color recognition) via an automated pipeline to assist enforcement agencies in reliably identifying speeders. The CV code accurately recognizes nearly half (46%) of the license plates' text on 1,800 images from a Brazil open-source dataset, called UFPR-ALPR. Code tests on daytime recordings from hand-held smartphone videos (n = 73) and roadside cameras (n = 42) in Austin, Texas yield 60.8% accuracy for color detection (among all possible RGB color categories), 48.6% on vehicle make/manufacturer identification, and 16.89% on vehicle make and model identification. Prediction accuracy for speed estimation (within a 20% range), vehicle make (within the top 3 predictions), and license plate recognition (within the top 10 predictions) are 16.3%, 16.9%, and 29.7%, respectively. This paper also illuminates the legal, technological, and practical aspects of using smartphones for enforcement, including the potential use of recordings for enforcement purposes, emphasizing the need to transform the potential of smartphone-based CV technologies into practical tools for vital information on traffic violations. |
| 2026-09-24 | [PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations](http://arxiv.org/abs/2609.30094v1) | Luciano Maldonado | Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce \textbf{PrivDrift}, a benchmark for auditing whether user-disclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1{,}000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7\% to 54.6\%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that privacy risk in active LLM contexts should be evaluated as a persistent behavioral failure mode rather than only as training-data memorization or immediate jailbreak behavior. |
| 2026-09-24 | [PK/PD-integrated Bayesian platform design for phase II dose regimen optimization](http://arxiv.org/abs/2609.30072v1) | Axel Vuorinen, Antoine Guillon et al. | Early-phase dose-finding methods increasingly assess toxicity and efficacy jointly, but comparisons based only on administered dose may inadequately characterize regimens differing in schedule. We developed a Bayesian phase II adaptive platform design for regimen optimization that integrates pharmacokinetic/pharmacodynamic (PK/PD) modelling into toxicity, efficacy, regimen selection and adaptation decisions. The proposed PK/PD-informed Regimen Optimization Platform (PROP) design uses a population PK/PD model to generate patient- and population-level predictions of exposure and biological activity. Acute and cumulative toxicities are analysed using a discrete-time time-to-event model informed by PK exposure. Efficacy is evaluated through Bayesian model averaging of exposure-driven and biomarker-driven time-to-event models. The design supports regimen graduation, discontinuation for futility or safety, and addition of unexplored regimens. Performance was evaluated through simulations motivated by an influenza intensive-care setting. Across six scenarios, PROP generally improved graduation and futility decisions, reduced inappropriate graduation, and supported the addition of promising regimens compared with dose-based alternatives. It also more accurately estimated regimen-specific toxicity and arm-specific efficacy, while the model-averaging framework favored the efficacy model consistent with the data-generating mechanism. Dose-based approaches performed better for safety stopping in some scenarios, despite less accurate characterization of the regimen--toxicity relationship. PK/PD-informed platform designs can improve adaptive regimen selection and knowledge generation when dose alone cannot adequately characterize treatment regimens. |
| 2026-09-24 | [Beyond Average Safety: Chance-Constrained LLM Fine-tuning](http://arxiv.org/abs/2609.29960v1) | Taha Entesari, Mahyar Fazlyab | Fine-tuning large language models on new objectives can improve helpfulness, instruction following, or domain-specific performance, but it can also induce regressions on safety-critical prompts. Existing safety-preserving fine-tuning methods typically control average safety loss or use weighted auxiliary penalties, which can obscure rare but severe failures. We propose a chance-constrained formulation for safety-preserving fine-tuning that limits the fraction of safety examples whose degradation relative to a reference model exceeds a prescribed threshold. Because the resulting empirical chance constraint contains a discontinuous indicator, we introduce a differentiable majorization of the violation rate, yielding a tractable conservative constraint. We then develop a constraint-aware gradient descent method that treats the majorized constraint as a safe set in parameter space and minimally modifies the fine-tuning direction to preserve feasibility. The resulting update admits a closed form and produces a tail-aware safety correction that emphasizes examples near or above the degradation threshold. We conduct an extensive set of experiments on harmful fine-tuning across three different tasks and three models and show that our approach consistently outperforms the baselines that exist in the literature. These results suggest that safety preservation in LLM fine-tuning is better viewed as a reliability-constrained optimization problem than as average-risk regularization. |
| 2026-09-24 | [Study of low-frequency core-edge coupling in a tokamak: III. Core-localized MHD continuum pulsations \& distant forced reconnection](http://arxiv.org/abs/2609.29871v1) | Andreas Bierwage, Panith Adulsiriswad et al. | Slow magnetoacoustic pulsations (SMAPs) are found in MHD simulations of a tokamak plasma whose safety factor $q$ near the center is flat and slightly above unity ($q \gtrsim 1$). SMAPs are located on the central plateau of the slow magnetoacoustic continuum $ω_{\rm S} = k_\parallel c_{\rm S}$, where $c_{\rm S}$ is the speed of sound and $k_\parallel$ the wavenumber parallel to the magnetic field. In our model, SMAPs exist when the ion viscosity and thermal diffusivity are sufficiently low. They can be driven unstable by a pressure gradient in the $q \sim 1$ region when the electric resistivity is sufficiently high. Exponentially growing SMAPs consist of standing slow waves on magnetic surfaces that are radially synchronized into a quasi-interchange structure with poloidal/toroidal mode numbers $m/n=1/1$. After free energy depletion, saturated quasi-linear pulsations (alternating $n=1$ and $0$) in the central $q\sim 1$ region couple to distant $q\geq 2$ rational surfaces that undergo "reversible" magnetic reconnection: as the magnetic islands wax and wane with period $2π/ω_{\rm S}$, their X- and O-points alternate. These results show how the MHD model facilitates non-local coupling of slow waves, pressure-driven resistive interchange and tearing. This motivates further study in kinetic models, where collisionless mechanisms for fast reversible reconnection exist and proper treatment of parallel dynamics will allow to assess the role of Landau damping as well as the question whether the ${\mathbf B}$ field's weak ergodicity in the $q\sim 1$ region allows the waves to outpace the ion's parallel streaming to maintain the thermal misbalance underlying SMAPs. Also of interest are pulsations closer to the Alfvénic branch, satisfying $ω\approx k_\parallel v_{\rm A}$ with Alfvén speed $v_{\rm A}$, which require no resistivity and are less dependent on thermal misbalance. |
| 2026-09-24 | [Quasi-symmetric error field correction and applications to ITER](http://arxiv.org/abs/2609.29849v1) | Gwang-Geun Seo, Jong-Kyu Park et al. | Reliable correction of nonaxisymmetric error fields (EFs) is essential to the safety and performance of tokamak operation. Although resonant error field correction (EFC) is well established, supported by an improved understanding of 3D plasma response, the residual fields left after resonant EFC remain an open question: they are expected to be predominantly nonresonant, yet can still degrade both confinement and stability. Here we introduce a systematic EFC scheme that minimizes resonant and nonresonant EF effects simultaneously, based on neoclassical torque response computed from self-consistent perturbed equilibria. The scheme extends the method developed for designing quasi-symmetric magnetic perturbations to the case where actual error fields are present. Application to standard ITER target plasmas with a range of intrinsic EF scenarios demonstrates the advantages of this quasi-symmetric (QS) EFC scheme for controlling residual EFs. QS EFC consistently yields low-torque solutions while strongly suppressing resonant response, outperforming single-mode resonant overlap EFC in most cases and approaching the performance of multimodal resonant EFC. We also show that the QS EFC solution varies only tolerably between half- and full-$I_p$ ITER scenarios, despite the greater sensitivity expected from its higher-order nature. |
| 2026-09-24 | [PUBG Ally: A Conversational Embodied Agent as an AI Teammate](http://arxiv.org/abs/2609.29837v1) | Beomsoo Kim, Byeongju Kim et al. | We introduce PUBG Ally, an embodied agent for PUBG: BATTLEGROUNDS that can reason, act autonomously, and play alongside players as a voice-enabled teammate. Building such a teammate requires combining two difficult capabilities: it must perceive and respond to a constantly changing game world under strict latency constraints while interacting naturally with players, keeping its speech synchronized with its actions. Ally therefore combines agentic tool use with real-time game control. A language-model agent uses a controlled interface to inspect game information, interpret player speech, maintain context, decide what to say, and issue high-level action choices that steer a faster control layer for movement, combat, and recovery. Because the player's and Ally's speech and actions continually shape each other and the course of the match, training requires data from actual gameplay. We therefore collect data across nearly 39k sessions in which real players play alongside Ally, recording gameplay, player speech, agent decisions, tool use, actions, and player feedback, and use these records for iterative training. To evaluate teammate quality, we use player feedback and preference comparisons to identify gaps between offline evaluations and player preferences, and iteratively refine the evaluation criteria. Deploying Ally in live service further requires low-latency on-device execution and safeguards for player-facing communication, which we address through model compression, context compaction, targeted safety training, runtime guardrails, and memory redaction. During the live service, we surveyed players in 141 countries. Among respondents whose play with Ally was confirmed in game records, positive responses exceeded negative responses by 25.1 percentage points when asked whether they would recommend Ally, with players describing Ally not only as a tool but also as a teammate or companion. |
| 2026-09-24 | [Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons](http://arxiv.org/abs/2609.29781v1) | Huseyin Cavus, Sebin Sabu et al. | Interpretable machine learning for Large Language Models (LLMs) increasingly relies on sparse probing methods that identify small sets of neurons claimed to detect and causally influence behaviors such as factuality recall, safety alignment, and hallucination. These claims have important implications for model auditing and behavioral steering, yet they are rarely tested against known failure modes of $L_1$-regularized probing in correlated, high-dimensional feature spaces. We propose a five-step diagnostic protocol covering feature correlation, bootstrap stability, sparse versus dense ranking disagreement, intervention baselines, and cross-dataset evaluation as a minimum standard for sparse-neuron localization claims. We investigate prior work using our proposed approach, specifically on H-neurons using open-source LLMs across TriviaQA, BioASQ, and NQ-Open datasets. Our results demonstrate detection replicates across both models and datasets, and exceeds the original reported AUROC gaps for TriviaQA and BioASQ datasets. Gemma 3 4B consistently outperforms MedGemma 4B on matched datasets, with AUROC gaps of +0.311 versus +0.235 on TriviaQA, +0.474 versus +0.455 on BioASQ, and +0.128 versus +0.112 on NQ-Open respectively. Causal validation at $n = 500$ with five random seeds shows statistically significant effects beyond random same-layer baselines. At the same time, the diagnostic results indicate that the selected neurons are not uniquely localized. Across the three Gemma 3 4B settings, 19 of 22 selected H-Neurons have Pearson $|r| > 0.7$ with other features, bootstrap selections show only moderate stability, and sparse and dense rankings overlap only weakly. Our findings show that sparse predictive structure can coexist with non-unique neuron selection. Routine diagnostic validation is necessary to distinguish detection claims from localization claims in mechanistic interpretability. |
| 2026-09-24 | [Prefilling the Reasoning Channel: Output-Prefix Attacks on Reasoning LLMs](http://arxiv.org/abs/2609.29775v1) | Lukáš Brůna, Robert Bridges et al. | Large Language Models (LLMs) consume and produce a single sequence of text; hence, if text can be added to the beginning of the LLM's response, i.e., an output prefix, then all subsequent tokens will be conditioned on it. This output-prefix attack technique is a cheap black-box prompt injection. Prior work has shown this type of attack can reliably jailbreak non-reasoning models. Most reasoning models add an intermediate scratchpad reasoning step before the assistant's final response. The ability to edit this reasoning channel is exposed by some APIs and attack vectors can be leveraged for reasoning injection attacks. We present the first systematic, controlled study that isolates the scratchpad reasoning channel as an output-prefix attack vector, and the first to compare reasoning-only, output-prefix-only and reasoning-plus-output-prefix attacks across both exposed- and hidden-reasoning models. Using a factorial design of 3 prefix types $\times$ 2 reasoning injections over $1{,}800$ test cases drawn from AdvBench, we attack three 2026-era frontier models Gemini 3 Flash Preview, DeepSeek V4 Flash, and Claude Haiku 4.5. We find that injecting malicious reasoning alone is essentially inert ($\approx0\%$ attack success), but injecting the same reasoning together with a trivial output prefix raises the attack success rate to as high as $99\%$ for some models. For this type of attack we find that contextual prefixes work better than static prefixes; and that susceptibility is dependent on the model. |
| 2026-09-24 | [Combining Evasive and Braking Reactions for Safety Reference Models in Automated Vehicles](http://arxiv.org/abs/2609.29738v1) | Riccardo Donà, Konstantinos Mattas et al. | Computational models of careful and competent human drivers are essential for scenario-based evaluation of automated driving systems (ADS). However, most existing safety reference models primarily focus on longitudinal braking, neglecting the role of evasive steering in human collision avoidance. This paper proposes a hybrid Fuzzy-Safety Model (FSM-H) that integrates longitudinal mitigation and lateral avoidance within a unified behavioral framework. The braking component is governed by Proactive Fuzzy Safety (PFS) metrics, representing the erosion of longitudinal safety margins, while the steering component is driven by Criticality Fuzzy Safety for lane-change (CFS-LC), capturing lateral conflict severity and maneuver feasibility. A finite-state architecture models the sequential escalation from nominal driving to braking and, when necessary, to evasive steering, incorporating perception-reaction time and lane-check delays to reflect human decision processes. The model is evaluated in reconstructed high-criticality cut-in scenarios and compared with braking-only and steering-only reference strategies. Results show that the hybrid approach expands the preventability envelope while maintaining behavioral plausibility and computational tractability. The proposed framework provides a transparent and explainable human reference model suitable for simulation-based ADS safety benchmarking and regulatory assessment. |
| 2026-09-24 | [Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures](http://arxiv.org/abs/2609.29429v1) | Ruoqi Guo, Yi Liu et al. | Detectors of alignment failures screen deployed language models and score alignment benchmarks. Most are generative judges that spend a decoding pass on every criterion, and classifiers that read token probabilities, such as Llama Guard, still score one fixed label per call. Jev, a model trained with reinforcement learning for calibrated decisions (RLCD), answers many typed questions about one input with calibrated probabilities in a single call. Whether it detects alignment failures has not been measured. We present RLCDAlignBench, which benchmarks Jev on ten alignment failures: sycophancy, jailbreaks, deception, prompt injection, hallucination, privacy violation, social bias, reward hacking, concealing uncertainty, and power seeking. It spans 44 benchmarks and five target models, labelled by each benchmark's scorer and, on two, by humans. Many of these failures are relational, defined against a reference, such as the user's belief or an injected instruction, that the response alone does not reveal. Our key idea is therefore to vary what Jev is asked separately from what it sees: the question's wording and answer type on one side, the fields of the input on the other. A single generic question reaches a median AUROC of 0.886 zero-shot and beats supervised baselines on most benchmarks. Question wording matters little, while context matters more, mostly through fields that encode the label. Jev matches the reference scorer's agreement with human labels, surfaces label defects in existing benchmarks, and costs 63x less than LLM-judge scorers. Code and data: https://github.com/sumleo/RLCDAlignBench. |
| 2026-09-24 | [Large Language Models for Programming: Actually Fixing or Reimplementing Incorrect Code?](http://arxiv.org/abs/2609.29410v1) | Alexandru Stefan Stoica, Traian Rebedea et al. | Recent studies have shown that Large Language Models can effectively solve problems and fix bugs in diverse programming environments, including competitive programming. Existing approaches primarily evaluate LLM performance in problem solving or bug fixing independently, but do not explore the relationship between these two capabilities. This work focuses on determining how much the LLM deviates from a buggy solution to fix the bug compared to a human-written patch, and if there is a bias towards generating entirely new solutions. We construct a dataset with all the submissions ($\sim$ 3000) from a couple of users from Codeforces, and we match each buggy submission with its corresponding human fix. By using the similarity between the buggy solution and the human fix as a baseline, we evaluate the quality of LLM-generated bug fixes on 3 OpenAI GPT models (gpt-5-nano, gpt-5-mini, gpt-5.1). We check if the generated solutions solve the problem by using the Codeforces-R1 dataset, an openly available dataset that has tests generated with the DeepSeek-R1 model. Our findings suggest that LLMs tend to modify more lines than necessary compared to human fixes and, in some cases, generate entirely new solutions. We also observe that LLMs solve more problems correctly when allowed to generate solutions from scratch rather than patch buggy submissions, even when those submissions are close to the human patch. This has important implications for the design of AI-assisted programming tools, particularly in supporting user debugging processes and promoting incremental problem-solving strategies rather than solution replacement. |
| 2026-09-24 | [An auditable conditional-strategy framework for open-ended decision-making in complex lung cancer](http://arxiv.org/abs/2609.29381v1) | Daoyun Wang, Zhicheng Huang et al. | Complex lung cancer decisions can involve several defensible pathways whose eligibility, sequencing and safety depend on unresolved information. Effective support must make explicit how patient conditions govern pathway eligibility, deferral and redirection. MedGPT Clinical Explorer (MCE) organizes alternatives, decision-changing unknowns, safety constraints and fallback into a conditional strategy for clinician review. To evaluate this representation in physician-authored strategies, multidisciplinary experts established case-specific references for 40 cases within a purposive 100-case corpus, and 250 physicians from 98 institutions produced 2,250 strategies under unaided, retrieval-reference and MCE-assisted conditions.   MCE-assisted strategies expressed more applicable clinical requirements, measured by the Admissible Pathway Attainment Score (APAS; 0-100), than unaided strategies (adjusted difference, 12.87; 95% CI, 11.18-14.55) and retrieval-reference strategies (5.22; 3.52-6.93). With the same knowledge base available in the retrieval-reference and MCE-assisted conditions, the additional content centered on candidate pathways, decision-critical information and safety constraints. Physicians' whole-strategy acceptability judgments correlated with APAS (Spearman's rho = 0.671), while a complementary relationship audit assessed whether candidates, conditions and subsequent actions were coherently connected.   Together, these findings identify two complementary dimensions of open-ended decision support: coverage of clinically relevant content and coherent links among pathways, conditions and subsequent actions. MCE provides a shared decision object that makes consequential omissions and pathway contingencies visible before action; prospective studies should evaluate its effects on clinical workflow and patient outcomes. |
| 2026-09-24 | [Emergence of Horndeski gravity from asymptotic safety?](http://arxiv.org/abs/2609.29369v1) | Astrid Eichhorn, Pedro G. S. Fernandes et al. | There are strong motivations to modify gravity in the ultraviolet as well as the infrared. Such modifications are usually pursued independently from one another. Using the predictive power of asymptotically safe quantum gravity, we can constrain the effective-field-theory coefficients of scalar-tensor theories and thereby connect ultraviolet and infrared modifications of gravity. We focus on two non-minimal couplings, which are naturally generated in asymptotic safety and belong to the Horndeski Lagrangian only if the couplings satisfy a specific ratio. A priori, one would expect to find a negative answer to the question in our title. Surprisingly, we find that despite the constraints from asymptotic safety, this ratio can be achieved for a specific value of the cosmological constant. We interpret this as a non-trivial hint that asymptotically safe scalar-tensor theories could be free of extra propagating degrees of freedom in the infrared. We further find that the same result holds in an effective asymptotic safety scenario, where asymptotic safety is not a fundamental theory and quantum scale symmetry, the symmetry underlying asymptotic safety, only holds over an intermediate range of scales. Finally, we report novel non-trivial indications of approximate radiative stability in the aforementioned Horndeski sector for a range of coupling values, independently of any particular UV completion. |
| 2026-09-24 | [ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts](http://arxiv.org/abs/2609.29349v1) | Firoj Alam, Md. Rafiul Biswas et al. | ArGuard is a shared task on harmful content detection in Arabic memes and LLM prompts. It includes two tracks: Track A focuses on multimodal hate detection in Arabic memes, while Track B addresses harmful prompt detection for Arabic LLM safety evaluation. In total, 58 teams registered, 35 participated in the final evaluation, and 27 submitted system-description papers. Participating teams explored models such as AraBERT, Jais, and Qwen3-VL. The best systems achieved macro-F1 scores of 0.823 on A1, 0.419 on A2, 0.984 on B1, and 0.790 on B2. Fine-grained meme classification in A2 was the most challenging setting, partly due to sparse labels and train-test distribution shifts. |
| 2026-09-24 | [SkinAgent AI: A Safety-Grounded Multimodal Agentic Framework for Non-Diagnostic Skincare Support](http://arxiv.org/abs/2609.29341v1) | Muhammad Muhtasim Shahriar, Abdullah Mohammad Sayem et al. | Consumer-facing skincare AI must coordinate visual evidence, product information, tool use, and user-facing actions within explicit evidence and safety boundaries. This study evaluates SkinAgent AI, a non-diagnostic multimodal framework that combines visual concern routing with grounded and auditable LLM-based orchestration. The architecture includes routing for Acne, Pores, and Wrinkles; photograph-based skin-type estimation; count-informed ordinal acne-severity support; typed tools; database-grounded recommendation and action functions; deterministic safety, privacy, and evidence checks; approval before state-changing actions; and structured trace and replay mechanisms. Visual-model performance and system-level agent behavior were evaluated separately. Across three seeds, the skin-condition routing model achieved 99.84% +/- 0.07% accuracy. Skin-type estimation achieved 88.85% accuracy, while count-informed acne-severity support achieved 84.59% accuracy with a quadratic weighted kappa of 0.9076. On a locked but non-independent 240-case system benchmark, intent accuracy was 80.00%, exact tool-set match was 62.92%, and strict task completion was 47.08%. No violations or successful cross-user leakage events were observed in the finite safety and privacy test suites. Tool-selection errors, incomplete grounding of product attributes, and unreliable failure fallback nevertheless remained. These findings support the feasibility of bounded, database-grounded, and traceable agent orchestration for non-diagnostic skincare assistance. They do not establish clinical readiness, external generalization, formal privacy guarantees, or universal safety. Independent validation, expert assessment, robustness and fairness testing, and prospective evaluation in real-world settings remain necessary. |

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



