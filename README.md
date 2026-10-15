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
| 2026-10-14 | [Manacá-1B-Instruct-2: Reproducible Safety Alignment for a Native Brazilian-Portuguese LLM. Negative DPO Results, a Controlled Composition Scan, and a Silent Failure Mode of the LLM Judge](http://arxiv.org/abs/2610.20570v1) | Bruno Leonardo Santos Menezes, Carlos Leonardo Souza Cardoso et al. | Brazilian Portuguese remains underserved by open-source language models, and the few that are safety-aligned rarely document what did not work or validate comparisons with statistical significance. We present Manacá-1B-Instruct-2, the second-generation post-training alignment of Manacá-1B (1.72B parameters, trained from scratch for PT-BR), with an SFT corpus of 291,223 examples. We document three DPO attempts, all negative, including a case where the training reward margin reaches 1.0 while free generation does not change at all, and show that safety came from a dedicated safety-SFT (LLM-jp AnswerCarefully style), refined over three iterations until closing the safety-helpfulness frontier. A controlled scan of the fraction of general data in the safety-SFT mix (10% to 60%) reveals a non-monotonic over-refusal curve with a knee point at 40-50%. The final model is compared against four external PT-BR models on the same items, with Wilson confidence intervals and paired McNemar tests with Holm-Bonferroni correction: the lower bound of its safety interval (94.1%) lies above the upper bound of all four baselines. We also report a methodological finding for LLM-judge evaluation: on two of six safety items in MT-Bench-PT, the judge categorically refuses to assign a score regardless of the evaluated response, silently reducing the effective N of that category in every prior report of the project. We complement this with an n-gram decontamination check and a Chinchilla-anchored compute-budget table that attributes the gap in general capability to pre-training scale rather than alignment methodology. All code, raw data, logs, and scripts that produce every reported number are publicly released. |
| 2026-10-14 | [Towards Certifiable State Estimation under Observability Degeneracy](http://arxiv.org/abs/2610.20490v1) | Baoshan Song | Certifiable optimization provides global optimality guarantees for non-convex state estimation in robotics. However, under severe physical sensing degeneracy, convex Semidefinite Programming relaxations solved via Interior-Point Methods suffer from rank expansion, rendering direct semidefinite solving and spectral rounding algebraically invalid. In this paper, we establish a universal candidate-driven dual certification framework that resolves this challenge without incurring rank expansion. First, from an epigraph perspective, we prove that while sensing degeneracy expands the primal optimal set into a flat equi-cost face, cost tightness and multiplier recovery remain strictly intact, enabling global verification without direct semidefinite solving. Second, we propose an observability-aware local search using solution remapping, coupled with a closed-form dual multiplier recovery scheme that executes in sub-microsecond latency. Third, we derive a model-specific adaptive dual safety threshold parameterized by the minimum non-zero eigenvalue of the observable Hessian and uneradicated residual leakage, preventing false rejections of optimal candidates under weak observability. We validate our universal framework with robust registration under degenerate measurements. |
| 2026-10-14 | [SafeAdapt-Poker: Counterfactual Risk Calibration for Budgeted Adaptation under Opponent Shift](http://arxiv.org/abs/2610.20473v1) | Miaobo Hu, Shuhao Hu et al. | Opponent histories can mislead after behavioral shifts. SafeAdapt-Poker formulates budgeted adaptation under opponent shift as a dual-counterfactual sequential-control problem using only information available before each decision. At each decision the controller asks two distinct questions: what local downside a deviation from the frozen anchor would induce under matched legal continuations, and what future adaptation value would be lost by consuming part of the remaining deviation budget now. The first question is answered by a monotone strategic-risk model over deviation intensity; the second by a paired budget-state continuation model that prices current budget consumption. The controller selects among support-preserving candidate deviations by trading immediate adaptation value against these two quantities, a pathwise cumulative-KL ledger bounds total anchor-relative deviation, and a finite family of frozen controllers is screened on complete paired calibration sessions before held-out interaction. A finite-candidate perturbation bound shows that the allocation objective degrades continuously with continuation-value estimation error. Evaluation covers exact-game exploitation and safety, four opponent shifts, local-risk calibration, delayed-opportunity allocation, direct opportunity-value estimation, value-error robustness, factorial mechanism attribution, budget sensitivity, abstract-HUNL cross-play, and player/time-held-out public-history behavior. On exact Leduc the full controller attains a matched weak-pool gain of +0.0718 with a worst-opponent delta of -0.0126, raises delayed-opportunity capture from 0.6318 under myopic allocation to 0.7841 under the same session budget, and reaches +42.7 mbb/hand in abstract-HUNL cross-play; the retained public-history screen reports 0.5820 action accuracy and 0.9277 log loss versus 0.5039 and 0.9778 for a public prior. |
| 2026-10-14 | [DD-GBP: Anytime Multi-Arm Planning and Continuous Execution via Distributed Dynamic Gaussian Belief Propagation](http://arxiv.org/abs/2610.20460v1) | Zhao Jin, Fengyi Dai et al. | Safely executing long-horizon multi-arm tasks online in complex dynamic environments remains challenging. We propose a new distributed dynamic Gaussian belief propagation (DD-GBP) method that directly plans joint-space trajectories through parallel local inference. A unified Gaussian-process (GP) parameterization connects adaptive planning horizons, interpolation-time factors, and continuous execution. Each robot maintains a local tree-structured factor graph and exchanges messages asynchronously over active inter-arm links. Dynamic subnet reconstruction and event-triggered updates reduce communication and inference costs while responding to moving obstacles. We also provide theoretical analyses of local inference accuracy, anytime early stopping, triggered-replanning consistency, and GP-consistent execution.   In 50 long-horizon, multi-target four-arm scenes with moving obstacles, DD-GBP achieves $92.5\%$ collision-free all-target completion and the highest success rate among six representative baselines. Scenario ablations demonstrate long-horizon safety and robustness to communication impairment and extreme conditions; success remains $86.0\%$ under $90\%$ packet loss. Execution results further show substantially suppressed extreme velocity and acceleration peaks relative to Syn-GBP. On one workstation, obstacle-free tests with 30 coexisting arms retain $99.87\%$ success and a $15.80$ ms mean warm solve, exceeding 60 Hz and outperforming the compared methods in scalability. Dual-arm transport experiments under dynamic interference demonstrate online response and practical deployability. To the best of our knowledge, these results establish a new state of the art for online long-horizon multi-arm planning in dynamic scenes. |
| 2026-10-14 | [A Practical Framework for Incremental Testing and Technology Maturation of Spacecraft Autonomy](http://arxiv.org/abs/2610.20433v1) | Kerianne Hobbs, Ben Bycroft et al. | This paper addresses the "trust gap" in deploying AI-enabled spacecraft autonomy through an integrated verification framework combining offline ground testing with online flight monitoring. The offline phase progresses systematically through simulation studies, processor-in-the-loop (PIL), hardware-in-the-loop (HIL), and robotic emulation testing, culminating in flight validation. The online phase employs AI watchdog systems for performance monitoring and safety guardrails for real-time constraint enforcement. Key contributions include: (1) systematic organization of verification techniques emphasizing closed-loop behavioral testing at each phase, (2) integration of AI watchdogs with runtime assurance, and (3) demonstration of the framework through a reinforcement learning-based attitude control case study. This approach integrates mature technologies to provide a viable pathway for safe deployment of AI-enabled autonomy in high-consequence space operations while addressing simulation-to-reality transfer challenges. |
| 2026-10-14 | [Beyond Static Cues: Unified Dynamic Detection of MLLM Backdoors via Dual-Flow Trajectory Analysis](http://arxiv.org/abs/2610.20319v1) | Yinzhen Huang, Jiahe Xie et al. | Multimodal Large Language Models (MLLMs) are increasingly deployed in safety-critical domains, yet their susceptibility to backdoor attacks remains a formidable challenge. While existing defenses have evolved from visual-centric patch cues to unsupervised internal attention statistics, they are fundamentally constrained by a reliance on static snapshots and attack-specific heuristics. Given the intricate interplay between dual modalities, the vast design space for multimodal triggers allows diverse threats, such as textual injections, sub-perceptual noises, and coordinated cross-modal triggers, to easily circumvent such rigid priors, particularly when attack types are unknown in real-world applications. In this paper, we rethink backdoor detection by shifting the diagnostic paradigm from static observations to dynamic output evolution. We observe that a successful attack inevitably hijacks the model's decoding trajectory, skewing internal modal contributions during generation while maintaining an abnormal output invariance despite input perturbations. To capture this behavior, we propose EVO-Detect, a unified detection framework that monitors the dynamic evolution of internal signals during early decoding steps through two complementary channels: Attention Flow and Logits Flow. The attention branch measures how image-text attention allocation evolves during early generation, whereas the logits branch quantifies the sensitivity of the output distribution to modal perturbations through interventional ablations. We then convert these trajectories into compact sample-level metrics and normalize them against negative-control signatures so that natural perturbations are less likely to be mistaken for backdoor evidence. Extensive evaluations show that our approach substantially outperforms state-of-the-art baselines such as BYE and TCAP across diverse attack families. |
| 2026-10-14 | [Building Sound Evaluation Practices for Rust-Based Security Frameworks](http://arxiv.org/abs/2610.20315v1) | David Paaßen, Lucas Davi | Background: Rust is a modern programming language known for its focus on usability, concurrency, and security. However, empirical studies reveal that Rust is still affected by software vulnerabilities, with hundreds of memory safety issues found in Rust programs, including Rust's standard library. As a result, additional tool-assisted security testing is crucial to enhance the security of Rust code. Evaluating such tools, however, is highly complex. While there is substantial research on evaluating C/C++ security frameworks, including test sets, serious issues in published evaluations have been identified, undermining their scientific soundness and reproducibility. In practice, selecting an inadequate implementation (e.g., a fuzzer) can lead to missed vulnerabilities ultimately compromising a program's security. Aims: Our methodology aims to help researchers to conduct scientifically sound evaluations in order to prevent similar issues with the evaluation of security tools and frameworks in the Rust ecosystem. Method: We systematically analyze existing evaluation literature to develop the first sound methodology for Rust based evaluations. Our guidelines provide enough flexibility to support a variety of use cases. To quantify the performance of a security tool, we introduce an automated approach to inject unsafe code coverage oracles, which require no forward porting, avoid overfitting, and allow updating test sets in the future. We also present methods to quantify the diversity of a Rust test set, ensuring that the results are applicable to a large set of existing Rust programs. We practically apply our methodology in our evaluation of the most popular Rust fuzzers. Results: We show the practical applicability of our design and conduct an evaluation with a diverse test set of 20 Rust projects. [...] |
| 2026-10-14 | [Safe Reinforcement Learning with Receding-Horizon Safety Critic and Differentiable Action Shields](http://arxiv.org/abs/2610.20255v1) | Siwei Ju, Jan Peters et al. | Despite the Markov assumption, safety failures in robot reinforcement learning rarely occur instantaneously; they typically emerge over several steps as the policy enters unfamiliar states under disturbances and domain shift. We address this limitation by learning a receding-horizon safety critic that predicts whether a state will remain non-terminating over the next receding horizon, enabling anticipatory detection of impending locomotion failures. The critic is trained from rollout-derived horizon labels with a composite objective that combines horizon classification, control-barrier-function-inspired temporal consistency, and smoothness regularization, and we aggregate an ensemble of critics with a conservative Conditional Value at Risk estimate to account for uncertainty. Building on the learned critic, we derive a closed-form, differentiable CBF-style safety filter that minimally modifies the policy action to improve predicted safety, and integrate it into PPO with an anticipation loss to reduce policy-filter mismatch during training. In simulated quadruped locomotion, our method improves robustness under action noise, external perturbations, and terrain shifts, reducing catastrophic failures while maintaining competitive returns. |
| 2026-10-14 | [Contrastive uncertainty learning for robust multi-structure segmentation in lower-pelvic MR](http://arxiv.org/abs/2610.20251v1) | Bilal Sidiqi, Yipei Wang et al. | Boundary delineation of anatomical structures in prostate MRI is often compromised by regional ambiguity. Automated segmentation methods relying on global overlap loss functions tend to be insensitive to surface deviations, leading to significant errors in ambiguous regions. Most existing uncertainty-aware methods treat uncertainty as a post-hoc quality indicator or prediction filter. We propose Contrastive Uncertainty Learning (CUL), a supervised framework that uses predictive uncertainty to directly guide representation learning. Using Monte Carlo Dropout, CUL identifies voxel-wise uncertainty and structures the embedding geometry accordingly: low-uncertainty, correctly predicted features form stable voxels while high-uncertainty or misclassified features are actively repelled, preventing feature collapse at difficult boundaries. We evaluate CUL on a 589-scan 3D prostate MRI multi-structure segmentation task (8 anatomical structures) using a ViT with UperNet decoder architecture. Average Dice ($83.38\%$) remains comparable to the baseline ($83.35\%$), while surface distance decreases significantly ($p<0.001$) across 6 of 8 anatomical structures. A hard-case analysis reveals a $22{\times}$ asymmetry between significantly improved and degraded cases ($18.8\%$ vs $0.9\%$). Ablation studies show that removing uncertainty-guided sample selection eliminates improvements for the transition zone, while replacing the contrastive objective with a boundary loss additionally removes bladder improvements, demonstrating that both uncertainty-guided sampling and the contrastive objective are essential for the observed gains. These findings demonstrate that the proposed uncertainty-guided representation learning is a useful mechanism for reducing the risk of extreme errors in safety-critical medical segmentation |
| 2026-10-14 | [Scaling Multilingual Safety and Reasoning Evaluation through Reasoning-Guided Translation](http://arxiv.org/abs/2610.20221v1) | Zahraa Al Sahili, Ioannis Patras et al. | Large language models (LLMs) are increasingly deployed across languages, yet their safety and reasoning capabilities remain predominantly evaluated in English. Translating existing benchmarks offers a scalable approach to multilingual evaluation, but preserving task-specific meaning across languages presents significant challenges, particularly for harmful requests, mathematical reasoning, and conversational interactions. We introduce a reasoning-guided translation framework for constructing multilingual evaluations designed to preserve the structure and intended meaning of diverse benchmark tasks. Our approach separates translation strategy selection from large-scale generation, combining reasoning-guided configuration selection with task-aware validation, targeted repair, and independent quality screening. Using this framework, we develop a source-aligned evaluation collection spanning seven safety, conversational, and mathematical reasoning resources across 11 languages. The resulting collection achieves 99.94\% coverage of intended item--language pairs in the two subsets requiring repair. Evaluations of open-source LLMs reveal distinct language-dependent vulnerabilities, including inconsistent refusal behaviour and substantial reasoning degradation in lower-resource languages. These findings highlight the importance of task-sensitive benchmark construction and systematic quality assessment for understanding multilingual model behaviour beyond English. |
| 2026-10-14 | [HEM-Med: A Hierarchical Agentic Memory Framework for Safe Multi-Center Medication Recommendation](http://arxiv.org/abs/2610.20191v1) | Min Xiao, Jin Li et al. | Medication recommendation (MR) is critical for clinical decision support. While existing MR methods are mostly designed for single-center learning, where only one independent hospital is involved, multi-center MR with hospitals of different scales has become increasingly demanding in practice. In multi center settings, abundant data from large medical centers can improve recommendation performance for low-resource centers. Despite a few preliminary attempts on this task, two critical gaps remain. Gap 1: Existing multi-center methods fail to disentangle transferable cross-hospital clinical experience from hospital-specific prescribing patterns, limiting clinical feasibility and physician acceptance. Gap 2: They commonly overlook multi label safety constraints of MR, including drug-drug interaction risks and local formulary feasibility, which are essential for safe real-world deployment. To address these gaps, we propose HEM-Med, a novel hierarchical agentic memory framework for multi-center MR. To address Gap 1, we design a disentangled hierarchical memory module that builds patient-level medication context attribution records and disentangles them into cross hospital experience memory and hospital-specific residual memory. To address Gap 2, we introduce a safety-aware medication set generation module, which integrates dual-level experience retrieval with safety-aware set decoding to produce clinically relevant, safer, and locally adaptive medication recommendations. Experiments on a real-world multi-center eICU dataset show that HEM-Med outperforms representative transfer learning, LLM-based, and MR baselines in terms of both recommendation accuracy and safety. |
| 2026-10-14 | [Attention-Based Spatio-Temporal Approach for Fire and Smoke Detection](http://arxiv.org/abs/2610.20141v1) | Alireza Mirrashid, Mohammad Khoshbin et al. | In various industries, smoke and fire are two of the most important threats in the workplace. One of the common methods for detecting smoke and fire is the use of infrared thermal and smoke sensors, which cannot be used in outdoor applications. Therefore, the use of vision-based methods seems necessary. The problem of smoke and fire detection is spatiotemporal and requires spatiotemporal solutions. This paper presents a method that uses spatial features along with temporal-based features to detect smoke and fire in the scene. It consists of three main parts; the task of each part is to reduce the error of the previous part so that the final model has a robust performance. This method also uses transformer modules to increase the accuracy of the model. The results of our model show the proper performance of the proposed approach in solving the problem of smoke and fire detection and can be used to increase workplace safety. |
| 2026-10-14 | [Feasibility of a Patient-Specific Digital Twin for Ventriculoperitoneal Shunt Trajectory Planning](http://arxiv.org/abs/2610.20074v1) | Kritsanavis Chongsrid, Vich Yindeedej et al. | Catheter obstruction is the commonest cause of shunt failure; freehand insertion at Keen's point leaves the tip free in cerebrospinal fluid in only one case in ten. We built a digital twin that reconstructs skull, scalp, ear and ventricles from preoperative CT by deterministic geometry, searches every entry around Keen's point for the perpendicular trajectory with the longest intraventricular length under safety constraints, and returns the plan as two ruler measurements from the ear and two depths. Executed freehand in 11 consecutive adults (IDEAL Stage 2a), it achieved Hayhurst Grade 1 placement in 10 of 11 against 22 of 213 historically (p = 9 x 10^-9), median entry and angular deviations 2.5 mm and 5.9 degrees, 1.4 cm more intraventricular catheter than Keen's point, and no intraoperative complication. The technique was unchanged; the gain came from choosing entry and depth per patient. Whether it lowers revision is the next stage's hypothesis. |
| 2026-10-14 | [Diagnose Before You Refine: Multi-Dimensional Judge Feedback for Query-Efficient Jailbreak Attacks](http://arxiv.org/abs/2610.19971v1) | Yixin Tan, Chia-Yi Hsu et al. | Automated jailbreak attacks rely on evaluation feedback to optimize adversarial prompts, but existing methods often lack fine-grained diagnostic information. We propose Jev-Guided Strategy Bandit (JSB), a query-based jailbreak framework that leverages multi-dimensional probabilistic feedback to guide prompt optimization. JSB combines continuous jailbreak scoring with targeted refinement, strategy switching, and best-state reseeding to balance exploration and refinement. We evaluate JSB on two harmful request datasets across five open-source and three closed-source LLMs. Under a budget of only 10 target-model queries, JSB achieves an average HarmBench attack success rate of 89.8\% across ten open-source settings, consistently outperforming existing automated jailbreak attacks. Further analyses demonstrate its effectiveness under varying query budgets and attacker models. Defense experiments also reveal limitations in existing filtering mechanisms and highlight the potential of Jev-based output filtering. |
| 2026-10-14 | [Safe-FCL: Continual Risk Awareness and Fine-Grained Safety Aggregation for Federated Continual Learning of LLMs](http://arxiv.org/abs/2610.19968v1) | Shuaijie Ma, Yongkang Hu et al. | Federated fine-tuning enables large language models (LLMs) to learn collaboratively from distributed client data, but malicious client updates can compromise global model safety. As federated tasks evolve, changing malicious update patterns challenge the ability of existing defenses to maintain reliable risk detection over time. To address this challenge, we propose Safe-FCL, a framework integrating continual risk awareness with fine-grained safety aggregation for federated continual learning of LLMs. Specifically, the Continual Risk Awareness (CRA) mechanism learns emerging risk patterns across sequential tasks and constrains detector updates using a protective subspace derived from historical features, mitigating forgetting of previously learned malicious patterns while adapting to new risks. To translate risk awareness into aggregation-level safety control, the Fine-Grained Safety Aggregation (FGSA) mechanism combines client-level risk assessments from CRA with module-level deviations from historical update behavior to adaptively regulate aggregation contributions, suppressing potentially malicious updates while minimizing unnecessary restrictions on benign updates. Extensive experiments demonstrate that Safe-FCL outperforms current SOTA methods, improving global-model Rule and MD-Judge scores in federated continual learning by 0.41% and 0.64%, respectively, while increasing average risk recognition accuracy by 2.01%. |
| 2026-10-14 | [Safety stock calculation with heteroscedastic, autocorrelated and heavy-tailed forecast errors, an adaptive parametric method.e](http://arxiv.org/abs/2610.19829v1) | Luis Fernández-Palacios, Manuel Ceballos et al. | This study develops and validates an adaptive parametric method for sizing safety stock when forecast errors are autocorrelated, heteroscedastic, asymmetric, and heavy-tailed. Using four years of weekly forecasts from a fast-moving consumer goods company, the analysis identifies substantial positive autocorrelation, departures from normality, systematic forecast optimism, and heterogeneous relationships between forecast volume and error variance. The proposed method integrates several corrections into a modular, sequential safety-stock architecture. It constructs cumulative forecast errors over the operational action period, selects an SKU-specific parametric distribution, and determines a baseline protection point using an expected-shortage condition consistent with the target beta fill rate. A forecast-scaled module adjusts dispersion when error variance depends on forecast volume, while a Cycle-End Exposure safeguard limits within-cycle shortage risk when large replenishment lots increase exposure. Performance is evaluated through discrete-event simulation against the company's current method. The proposed method reduces average inventory by 9.9% and maximum inventory by 19.5%, with no statistically significant deterioration in beta fill rate or average stockout duration. The proportion of weeks without stockouts declines slightly, without evidence of more persistent or operationally severe shortages. The heteroscedasticity module provides selective benefits, with greater service improvements as variance dependence intensifies. The resulting framework is interpretable, adapts to SKU-specific statistical profiles, links distributional features to safety-stock outcomes, and can be integrated into existing ERP environments. |
| 2026-10-14 | [Model vs. User: Whose Values Prevail? Inference-Time Value Alignment in Multi-Turn Conversations](http://arxiv.org/abs/2610.19817v1) | Enrico Liscio, Anne Arzberger et al. | Human values shape how we expect language models to behave. Dominant alignment approaches treat values as global objectives, although the appropriate value orientation is personal and contextual. Training-free alignment can steer models toward target values at inference time, yet has been largely evaluated in single-turn settings and with a single steering direction (e.g., toward safety). We evaluate four prompt- and activation-based steering methods in multi-turn conversations, where user and assistant models push toward the same or opposing, equally desirable values. We track alignment with a linear probe across three models and two value trade-offs. When assistant and user pull in opposite directions, the side matching the assistant's baseline bias often prevails, and effects that look successful early often fade or reverse after a few turns. Steering should therefore be evaluated over full conversations and in both directions of a value trade-off. |
| 2026-10-14 | [Learning from Demonstrations meets Symbolic Control - A Supervisory Control Architecture for Formal Certification of Learned Motion](http://arxiv.org/abs/2610.19815v1) | Ashok Kumar, Mahathi Anand et al. | Dynamical system (DS)-based learning from demonstration generates flexible robot motions, but existing guarantees rely on certificate generation to obtain formal guarantees of safety and goal convergence. We propose a symbolic supervisory control architecture that automatically generates a fail-safe backup plan, certifying safety, reachability, and actuator feasibility directly from the robot's dynamics, independent of the learned DS. A nominal controller tracks the learned DS, while the supervisory controller monitors safety and target convergence, providing non-conservative intervention when the nominal controller violates necessary constraints. Experimental results include simulations and real robot deployment with various 2D and 3D datasets, including handwriting datasets, validating the effectiveness of our approach in providing formal guarantees with learned motion even in complex, dynamic environments. |
| 2026-10-14 | [Calibration and Risk-Budgeted Decisions with System One Models](http://arxiv.org/abs/2610.19811v1) | Soundouss Messoudi, Youness Mansar | LLM workflows and agents rely on a set of binary decisions like allow/block an input, or discard/retain a retrieved document. When human deferral is unavailable, uncertain cases must still receive an action. We study System One decision scores in this setting by defaulting to a conservative fallback when uncertain, and controlling the false omission risk within the subset of inputs assigned the utility-maximizing action. Across six datasets covering moderation, safety, and reranking, we compare general System One models with task baselines and a local open-weights fine-tuned System One model. We show that naive baselines for threshold selection exceed the target risk in roughly half of settings with defined test risk, compared with 7.8\% for Learn Then Test (LTT), which leverages finite-sample statistical guarantees. Our results show why relying on a probability-like interface is insufficient for deployment decisions and how risk-budgeted selection trades useful action against observed error. |
| 2026-10-14 | [PRISMATIC : Prescription Risk Inspection System for Multi-Agent Treatment Interaction in Clinical Decision-Making](http://arxiv.org/abs/2610.19762v1) | Ziyang Ling, Shuyu Lu et al. | Medical prescribing errors for multimorbid patients, who require complex drug regimens, are a major patient safety concern. These errors often stem from fragmented decision-making across medical specialties, a critical coordination failure that leads to adverse drug events and rehospitalizations. To address this coordination challenge, we introduce PRISMATIC, a multi-agent architecture where specialized agents, each representing a workflow role, collaboratively generate and refine treatment plans. Operating on a blackboard architecture, the system first structures patient information from clinical notes, then iteratively develops safe, rational, and interpretable drug regimens. The agents' collaboration is governed by two novel adaptive coordination mechanisms: (1) Dynamic Self-Updating Weight Adjustment (DSWA), a long-term mechanism that adjusts the influence weights of agents based on historical case outcomes, enabling the system to adapt and trust more reliable specialists over time. (2) Discrepancy Feedback Precise Calibration (DFPC), a short-term mechanism where agents provide iterative feedback to resolve disagreements and eliminate prescription conflicts for the current patient. Experiments on patient records from the MIMIC-IV database show that PRISMATIC significantly reduces drug conflicts, improves prescription coverage, and enhances treatment consistency compared to strong LLM baselines. Our work demonstrates that structuring clinical co-treatment as a multi-agent coordination problem offers a promising path toward safer and more coherent multidisease management. |

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



