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
| 2026-08-28 | [xTRUCE: A Provably Safe Arbiter for Multi-xApp Conflict Mitigation in Agentic O-RAN](http://arxiv.org/abs/2608.28532v1) | Le Xia, Rose Qingyang Hu et al. | The open radio access network (O-RAN) is evolving toward agentic operation, where large language model (LLM)-driven xApps/rApps generate control proposals under operator intents. However, such proposals may be conflicting, infeasible, or hallucinated, and no existing system jointly provides proposal-independent safety, priority-aware reconciliation, and traceable feedback. To this end, we propose a provably safe arbiter, namely xTRUCE, in the near-real-time (Near-RT) RAN intelligent controller for mitigating multi-xApp conflicts in gNB control. We first develop a structured xApp proposal interface and a three-layer constraint hierarchy that places physical limits and operator-defined rules above relaxable performance targets, alongside a dual-timescale control action space. A two-stage arbitration mechanism then minimizes target shortfalls in the operator-priority order to finalize safe E2 actions within the Near-RT latency budget, while returning conflict certificates to xApps and the operator for renegotiation. Finally, we implement xTRUCE in a multi-cell O-RAN use case, and evaluate its multi-process prototype through simulations with live API-backed LLM xApps and over-the-air experiments on OpenAirInterface/FlexRIC-based O-RAN stacks. Results show that xTRUCE ensures gNB control safety with $100\%$ protected services despite severe proposal hallucinations, achieves priority-consistent performance satisfaction under overload, efficiently guides LLM intent renegotiation via certificates, and keeps a delay-safe E2 control loop. |
| 2026-08-28 | [When Robots Mishear Us: Mapping the Safety Risks of Voice-Controlled Embodied AI](http://arxiv.org/abs/2608.28518v1) | Sihan Jia, Oliver Lemon | We investigate whether automatic speech recognition (ASR) errors in user input can lead to unsafe outputs from Embodied AI (EAI) models. We find that ASR errors can lead to harmful instructions being accepted and executed by EAI models, thereby reducing safety. We simulate ASR errors and combine them with existing safety benchmarks (SafeAgentBench and POEX) to evaluate how different errors affect embodied AI safety. We find that some of them preserve semantic structure but increase harmful ambiguity, while others weaken the model refusal behaviour and allow unsafe plans to be generated and executed. We show that in some cases automatic correction of ASR errors can reduce the risk, but this is not always effective. Overall, we show that ASR errors lead to significant safety risks for embodied AI. |
| 2026-08-28 | [LLM-Based Agents for Software and Systems Security: Approaches, Applications, and Assessment](http://arxiv.org/abs/2608.28490v1) | Jingjing Nie, Jiawei Guo et al. | Software and systems security workflows are typically procedural: analysts inspect heterogeneous artifacts, form hypotheses, invoke tools, interpret outputs, and revise plans. Large language model (LLM)-based agents, which can plan, use tools, retain state, and revise actions across multi-step workflows, are being rapidly adopted to automate this work. Given the consequences of delegating security decisions to autonomous systems, understanding how such agents are built, used, and assessed is crucial. Yet to this date, there remains a lack of systematic understanding of what has been done and how far we are in this field: the term "agent" is applied inconsistently, applications differ sharply in risk, and assessment protocols are often incomparable. To gain a comprehensive and coherent view of this area hence inform relevant future research, this paper provides a systematic literature review of the (1) technical approaches, including agent architecture, perception, memory, reasoning and planning, action space, orchestration, and self-improvement, (2) applications, with respect to the security tasks served, and (3) assessment, including the datasets, outcome and trajectory metrics, safety measures, and baselines considered, over the peer-reviewed literature spanning the emergence of this area (2023--2026). Our synthesis reveals a field that has built agents able to act but not yet agents whose authority is bounded or whose behavior is auditable. In addition to knowledge systematization, we also extend our insights into the limitations of and challenges faced by current approach, application, and assessment designs, which shed light on potentially promising future research directions. |
| 2026-08-28 | [Significance-Driven Semantic Communication](http://arxiv.org/abs/2608.28441v1) | Christian McDowell, Andrea Panebianco et al. | In this paper, we study a significance-driven cross- layer semantic communication design problem. Based on sta- tistical decision theory, we introduce an information-theoretic measure of per-sample data significance that quantifies the task-specific value of each individual observation. Using this metric, we formulate a cross-layer optimization problem that simultaneously optimizes (i) physical-layer semantic encoding and inference and (ii) MAC-layer resource allocation, with the objective of maximizing semantic spectrum efficiency, defined as the semantic value delivered per unit bandwidth per unit time. At the physical layer, we develop Meta-Learning Variational Information Bottleneck (Meta-VIB), a new semantic transceiver that employs a meta-learned hypernetwork to compress high- dimensional observations into semantically significant latents, enabling instantaneous adaptation to dynamic channel conditions and varying symbol budgets without online retraining. At the MAC layer, we model channel allocation as a Multi-Action Restless Multi-Armed Bandit (MA-RMAB) and adopt the Q- Maximization algorithm, which dynamically allocates channel resources to sensors based on their semantic value of information. Experimental results on a real-world pedestrian safety dataset demonstrate that our joint design achieves substantial gains in semantic spectrum efficiency over baselines, reaching up to 1000 times gain at an average SNR of 0 dB and 40 times gain at an average SNR of 5 dB. |
| 2026-08-28 | [CultureConverse: A Multilingual Multi-turn Simulation Harness for Culturally Grounded Assistance in East and Southeast Asia](http://arxiv.org/abs/2608.28405v1) | Bryan Chen Zhengyu Tan, Weihua Zheng et al. | Current cultural evaluations for large language models (LLMs) often reduce culture to single-turn factual recall via MCQs, failing to capture a common use case: users seeking practical help over multiple turns in culturally grounded scenarios. We introduce CultureConverse, a scalable, multilingual simulation and evaluation harness for culturally grounded assistant dialogue that covers 10 East and Southeast Asian regions, 58 subgroup identities, and 7 domains. Each simulated and evaluated episode produces a scored interaction where the assistant assists the user and infers cultural constraints from partial information. The resulting CultureConverse-DS dataset contains 14,610 benchmark (evaluation) episodes and 274,295 oracle-guided (gold-mode) dialogues. In our benchmark evaluation of 18 models, GPT-5 mini achieves the highest assistance quality. Human annotation experiments suggest that our evaluation framework is a sufficient proxy for human judgment. Performance gains from fine-tuning on 27,860 high-quality CultureConverse-DS samples improve in-domain assistance and transfer out-of-domain to cultural MCQ and safety classification benchmarks. We release the harness, both splits, and judge prompts to support interactive evaluation of cultural competency. |
| 2026-08-28 | [AGENT-O: A Semantic Agent Card Framework for Interoperable and Governed Healthcare AI Agents](http://arxiv.org/abs/2608.28345v1) | Pengze Li, Cui Tao | AGENT-O is a modular ontology framework that defines a semantic Agent Card for representing health-oriented AI agent systems and supports assessment of reporting completeness in scientific publications. AGENT-O was developed as an OWL 2/RDF ontology covering runtime, models, workflow, tools, clinical use, evaluation, provenance, governance, and reporting assessment. Evaluation included ontology inventory, OWL-RL reasoning, three SHACL suites, 12 SPARQL competency queries, three cases, and model-assisted reporting-completeness assessment of 279 papers across five dimensions. The ontology contained 1,962 RDF triples and 1,922 Protege axioms, with 252 active classes, 198 active object properties, and 51 datatype properties. All SHACL suites conformed on example graphs, all competency queries returned prespecified evidence, and all 279 papers were scored. Incomplete reporting was highest for runtime/architecture (84.6%), governance/safety (82.8%), and provenance/reproducibility (78.1%), compared with evaluation (25.8%) and benchmark-process alignment (29.8%). AGENT-O supported semantic Agent Card representation and reporting assessment while revealing an evaluation-specification gap: evaluation and benchmark procedures were reported more consistently than runtime architecture, governance, and reproducibility. AGENT-O provides a reusable ontology, semantic Agent Card profile, and reporting-completeness workflow for structured reporting and gap identification, but does not assess agent quality or deployment readiness. |
| 2026-08-28 | [PanelShield: Verifiable Closed-Loop Safe Planning for Robotic Industrial Panel Operation](http://arxiv.org/abs/2608.28305v1) | Guipeng Xin, Jiahe Xu et al. | Industrial panel operation is knowledge-intensive and safety-critical. Beyond control recognition and action generation, execution must satisfy constraints in operation manuals and safety regulations. While foundation-model-based planners show strong semantic capability, they typically lack computable, localizable, and reproducible mechanisms for violation detection and repair. To address this, we propose PanelShield, a verifiable closed-loop safety planning framework for manual-guided industrial panel operation. The framework generates parameterized action primitive sequences from task-relevant manual evidence and applies dual formal verification with LTL and a Safety FSM to enforce cross-step temporal correctness and local transition legality. When violations occur, it outputs a structured counterexample with the earliest violating step and cause, enabling targeted repair and re-verification. We build a multi-level long-horizon planning benchmark covering three representative industrial device panels, and evaluate the framework in simulation and real-world robotic experiments. Results show that PanelShield improves complex safety-constrained task performance over foundation-model-only planning baselines while reducing the violation rate to 2.7%, with 4.1 s total latency. Real-world experiments demonstrate end-toend feasibility. Overall, PanelShield offers a verifiable approach to robotic panel operation that balances flexibility, safety, and auditability. |
| 2026-08-28 | [MaCoPlanner: LLM-Assisted Manual-Compiled Task Planning with Proactive Safety Verification for Robotic Industrial Panel Operation](http://arxiv.org/abs/2608.28300v1) | Guipeng Xin, Jiahe Xua et al. | Robotic industrial panel operation requires not only accurate control localization but also compliance with operating procedures, safety rules, and device-state constraints distributed across heterogeneous manuals. This study presents MaCoPlanner, a task-planning framework built on knowledge compiled from equipment manuals that converts equipment manuals into a typed intermediate representation, retrieves task- and state-relevant evidence, and uses it to support plan generation. Before actuation, candidate plans are symbolically rolled out and checked against procedural and state-transition constraints; detected violations are localized and returned for targeted repair, while unresolved plans are rejected. A separate execution interface grounds verified symbolic actions to physical controls and updates the device state. Under an independent evaluation oracle, MaCoPlanner achieves a final violation rate of 2.7%, and 26.3% of the runs in the repair analysis are rejected after exhausting the refinement budget. Compared with Raw-Manual, task success increases from 62.8% to 84.4% on Level-2 tasks and from 25.9% to 43.2% on Level-3 tasks. Experiments on a controller-panel simulator without an attached industrial load further demonstrate integrated execution feasibility under representative interaction conditions, without claiming industrial deployment readiness. |
| 2026-08-28 | [REINS: Refusal-Enhanced Inhibitory Steering with Sparse Autoencoder Features](http://arxiv.org/abs/2608.28233v1) | Kai-Xuan Ding, Hao-Xiang Xu et al. | Steering with Sparse Autoencoders (SAEs) offers a lightweight inference-time path for adapting the behavior of large language models without retraining. By exposing sparse and interpretable features, SAE steering provides a promising interface for safety control that guides harmful continuations toward refusal. However, we observe that complex wrappers can still undermine existing SAE steering methods on harmful prompts. To evaluate this failure mode systematically, we construct Generalized Undercover Instruction Safety Evaluation (GUISE), a dataset of harmful prompts with complex wrappers. Existing single direction SAE steering methods do not reliably produce refusals on harmful prompts, suggesting that refusal enhancement alone can be too weak when the harmful continuation path remains active. This motivates us to propose Refusal-Enhanced INhibitory Steering (REINS), which suppresses harmful continuation features and enhances safe refusal features in the same SAE feature space. Experiments on GUISE and other datasets show that prior methods either intervene too weakly or achieve only apparent safety through collapse, while REINS substantially reduces harmful responses, markedly improves safe refusals and largely preserves general capabilities. |
| 2026-08-28 | [Speculative Probing: LLM Monitoring at Speculative-Decoding Cost](http://arxiv.org/abs/2608.28099v1) | Collin Zhang, Tingwei Zhang et al. | Real-time classification during language model inference is valuable for safety filtering, behavioral analysis, and model monitoring, but current approaches force a trade-off between accuracy and efficiency. Hidden-state probes are fast but limited: they are either not context-aware: operating on a single vector and cannot model interactions across positions; or they are very costly: having dedicated classifier models (Llama Guard, Qwen Guard, LLM-as-judge) or performing computation on hidden states for all tokens and then pooling the results (MultiMax). This shows an intrinsic trade-off between efficiency and accuracy.   However, we find that the speculative-decoding module in recent LLMs can be repurposed for efficient high-quality classification. By appending a trained soft prompt at the end of the target sequence, we can repurpose the speculative-decoding module into a sequence classifier. At inference time in a speculative-decoding pipeline, the KV cache is already in GPU memory, so classification adds negligible overhead. We evaluate on four classification tasks across four models (Qwen3.5-4B, 9B, 27B, MiniCPM4.1-8B). Our small probes consistently outperform zero-shot GPT-5.4-mini and, on multilingual prompt safety, match or beat specialized 8B safety classifiers (Qwen3Guard-Gen-8B, Llama-Guard-3-8B) without running a full LLM. |
| 2026-08-28 | [Managing Inherent Risk: On the Conceptualization of Risk in Defense Systems](http://arxiv.org/abs/2608.28093v1) | NIklas Braun, Leon J. Brettin et al. | Certain defense systems are, by nature, deployed in a civilian environment in order to serve a defensive function for that environment. However, the risk involved with their deployment and operation poses a challenge for the public acceptance of these systems. Compared to safety engineering for civilian systems, the risk constellation is quite different. While reducing risk of safety-critical systems is generally desirable, defense systems are required to cause harm in order to be useful. In both cases, not all risk can be eliminated. The complex risk constellation of defense systems has implications for the systems' designs. We compare the concepts of risk in existing standards from both the civilian and the military domains. An extended constellation of risks needs to be considered, including risk caused by external threats to physical security. By including this risk constellation in public communication about defense systems, we aim to stimulate a productive debate. |
| 2026-08-28 | [A Controlled Audit of Architectural Complexity in Uncertainty-Aware Multi-Organ Ultrasound Classification](http://arxiv.org/abs/2608.28063v1) | Yang Song, Pengbo Sun et al. | Multi-organ ultrasound classifiers increasingly combine attention, mixture-of-experts routing, uncertainty gating, and evidential deep learning (EDL) objectives to address heterogeneous anatomy and acquisition. Yet a plausible design rationale does not by itself establish that an added component improves the trained system. We contribute a controlled complexity-audit framework, applied to the deployment decision between the maximal evidential candidate Full-EDL and simpler alternatives. Six candidates were evaluated on the primary dataset and three in an internal replication, using ten matched seeds, frozen image-level partitions, capacity- and optimisation-aware comparisons, symmetric temperature scaling, paired decision rules, and a separate out-of-distribution (OOD) veto. Retaining Full-EDL did not establish a reliable macro-F1 gain on either dataset, while the simplified alternatives remained inconclusive under the non-inferiority margin. Simple cross-entropy with temperature scaling (Simple-CE+TS) met the calibrated negative log-likelihood criterion on both datasets and showed favourable selective-risk ordering. The raw calibration advantage of evidential training disappeared after temperature scaling and did not recur on the second dataset. The gate had negligible observable influence at the audited checkpoints, and deleting the Full-only chain revealed no stable task or calibrated-loss benefit. Simple-CE nevertheless triggered the OOD veto against the fetal probe but not the lung probe, precluding an unconditional OOD-safety claim. We therefore selected Simple-CE+TS for the evaluated in-distribution objective while retaining Full-EDL as the maximal reference. Components should earn retention through functional and retraining-based evidence, and calibration and distribution-shift reliability should be evaluated separately. |
| 2026-08-28 | [Explainable Uncertainty Estimation for Reliable Medical AI](http://arxiv.org/abs/2608.28052v1) | Li Rong Wang, Jamie Duell et al. | Artificial intelligence has strong potential to support clinical decision-making, yet its adoption in healthcare remains limited due to a lack of trust. Uncertainty estimation can signal unreliable predictions, and explainable AI (XAI) can clarify how predictions are made but existing methods treat them separately, providing no feature-level insight into why a prediction is uncertain or which tests to prioritize to reduce it. To address this gap, we propose explainable uncertainty estimation, which unifies uncertainty estimation and XAI to both quantify uncertainty and explain feature-level contributions. We introduce the Expected Gradients Reconstruction Uncertainty Estimate (egRUE), which incorporates prediction explanations into its uncertainty computation and decomposes uncertainty into feature-wise contributions. We prove theoretical properties of egRUE and show through experiments that it improves reliability and interpretability compared to existing methods. A user study with medical experts further demonstrates that egRUE's explanations improve calibrated trust over uncertainty scores alone, increasing confidence in correct predictions and reducing confidence in incorrect ones. By combining prediction uncertainty with feature-level explanations, egRUE strengthens decision-making support in safety-critical healthcare settings, clarifying both when predictions may be unreliable and which features drive that uncertainty. |
| 2026-08-28 | [GAN-Based Semantic Communication for Image Transmission in IoV](http://arxiv.org/abs/2608.27989v1) | Ruixing Ren, Shan Chen et al. | For cooperative perception in the internet of vehicles, this paper proposes a generative adversarial network-based semantic communication framework to address the efficiency and fidelity bottlenecks of traditional communication systems in visual data transmission under limited bandwidth and dynamic channel conditions. At the transmitter, the framework adopts a pyramid attention network to extract semantic label maps and introduces a semantic priority preservation mechanism. It assigns differentiated weights to distinct semantic categories based on driving safety, guiding bit allocation and loss function design. At the receiver, an image reconstruction module integrating a coarse to-fine multi-resolution generator and multi-scale discriminator is designed. Combined with the temporal consistency branch, spatial pyramid pooling and class-aware convolutional layers, it achieves high-fidelity reconstruction of high-quality images from corrupted semantic labels. The model is trained with combined adversarial, feature matching and perceptual losses, effectively improving semantic consistency and visual realism of generated images. Experimental results on the Cityscapes dataset show that the proposed method outperforms existing counterparts in both semantic segmentation accuracy and reconstructed image quality, and maintains stable reconstruction performance under AWGN and Rayleigh channels. |
| 2026-08-28 | [CHISEL-ing Back Source Code with AI-enabled Iterative Recovery](http://arxiv.org/abs/2608.27981v1) | Varun Kohli, N Raghava et al. | Decompilation aims to recover high-level, compilable, and semantically equivalent code from binaries. Traditional decompilers produce pseudo-C that is difficult to read and does not compile, while the recent LLM-assisted approaches generate readable, but semantically incorrect code. LLM-aided iterative recovery is an emerging branch of research, but prior works rely on supplied test suites for semantic recovery. In this work, we present CHISEL, a test suite-free framework to iteratively recover source code from Ghidra-derived pseudo-C. CHISEL uses simple yet effective feedback from a compiler (static analysis) and a coverage-guided fuzzer (differential analysis), augmented by rich observables for grounded divergence detection and feedback, cross-iteration divergence memory, and best candidate retention. We systematically evaluate CHISEL for compilation and semantic recovery, feedback oracle soundness, and iteration overhead on 120 ExeBench functions compiled for the x86-64 architecture, across four optimizations (O0-O3), in both stripped and unstripped variants, using the open-weight Gemma4:31b LLM. CHISEL, with all recommended features, achieves an average of 96.1% re-compilability and 79.8% re-executability rates at an average of 2.1 iterations. Significantly, CHISEL recovers 26% of first-generation execution errors. At the same time, CHISEL feedback oracle falsely accepts only 9.4% candidates. Lastly, CHISEL performs significantly better than two recent prior work on LLM-assisted decompilation. |
| 2026-08-28 | [When Teacher Guidance Misleads: Reward-Aligned On-Policy Distillation](http://arxiv.org/abs/2608.27960v1) | Siyuan Gan, Yuhan Li et al. | On-policy distillation (OPD) has recently emerged as a popular post-training paradigm for large language models (LLMs), providing an efficient way to transfer the knowledge and capabilities of teacher models into student models. However, teacher guidance on student-generated prefixes is not always reliable. Training should optimize the model to generate responses that are more likely to be correct, or equivalently, to get higher outcome rewards. But during OPD, the teacher model may provide guidance that discourages the student from moving toward correct trajectories or moves the student toward incorrect ones, which is misaligned with outcome reward. Such misaligned guidance is unreliable, as it would mislead the optimization process and ultimately degrade model performance. To mitigate misaligned teacher guidance, we propose Reward-Aligned On-Policy Distillation (RA-OPD). The key insight is to keep only trajectories whose induced updates move the student toward correct trajectories or discourage the student from moving toward incorrect ones. Specifically, for each sampled trajectory, RA-OPD checks whether its trajectory-level distillation return is consistent with its outcome reward and then filters out the misaligned trajectories. RA-OPD selects more reliable trajectories to improve student model performance without requiring additional computational cost. We evaluate RA-OPD on math and code benchmarks using models from the Qwen3 family and the DeepSeek-R1 family. Across seven math benchmarks and three code benchmarks, RA-OPD significantly outperforms standard OPD and other tested OPD variants. |
| 2026-08-28 | [Cross-Session Decomposition Attacks: Scaling Risk and Intent-Aligned Retrieval Defense](http://arxiv.org/abs/2608.27945v1) | Disen Liao, Yihan Wang et al. | Scaling laws are usually read as a capability story: lower language-modeling loss yields more useful models. We study a safety consequence of this mechanism in \emph{cross-session decomposition attacks}, where benign-looking subqueries are asked across independent interactions and later recomposed toward a forbidden objective. We formalize this setting as \emph{compositional safety risk} and prove a conditional risk-transfer bound: when the reference environment already contains dispersed evidence for a risky reconstruction, the gap between deployed composed risk and reference composed risk is controlled by the model's excess loss on allowed subqueries. Synthetic withholding experiments show that wider transformers assign lower loss to held-out instructions that never appear verbatim in training but are recoverable from injected supporting facts. A 600-intent pretrained-LLM evaluation shows that larger Qwen3 and Gemma3 family members can yield greater harmful-capability uplift under a fixed decomposition-composition pipeline. As a defense, IntentAlign-MiniLM, our 22M-parameter intent-aligned retriever, outperforms much larger embedding models on held-out intent retrieval and yields the best learned-retriever harmful recall across tested guardrails. Code is available in \href{https://github.com/liaodisen/Cross-Session-Decomposition-Attacks}{our GitHub repository}. |
| 2026-08-28 | [A Deep Learning-Based Stacking Ensemble Framework for Turbofan Engine Remaining Useful Life Prediction](http://arxiv.org/abs/2608.27940v1) | Limon Bin Hossain, Md. Salehin Seyam et al. | This study proposes a two-level stacking ensemble framework for Remaining Useful Life (RUL) prediction of turbofan engines, evaluated on the NASA C-MAPSS benchmark using the FD001 and FD003 subsets. The framework integrates four heterogeneous deep learning base learners: Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), CNN-LSTM, and CNN-GRU, whose out-of-fold predictions are combined by an XGBoost meta-learner to capture complex degradation patterns while mitigating individual model biases. Comprehensive experiments demonstrate that the stacking ensemble achieves superior predictive performance, with Root Mean Square Error (RMSE) of 9.989 and 8.613, Mean Absolute Error (MAE) of 7.081 and 5.195, and R-squared values of 0.899 and 0.906 for FD001 and FD003, respectively. Compared to the best-reported baseline (TCAT: RMSE 11.12 and 11.02), the proposed method achieves RMSE reductions of 10.2 percent and 21.8 percent for FD001 and FD003, respectively. Feature correlation analysis, residual diagnostics, and training convergence curves validate the model's robustness. These findings underscore the efficacy of stacking ensemble methods for prognostics and health management in safety-critical aerospace applications. |
| 2026-08-28 | [Backup Control Barrier Function Synthesis using Sum-of-Squares Reachability](http://arxiv.org/abs/2608.27916v1) | Jungbae Chun, Shima Sadat Mousavi et al. | Backup control barrier functions (bCBFs) enforce safety for input-constrained nonlinear systems using a pre-certified backup set and controller, but their performance depends strongly on this prescribed pair. This letter develops a constructive method for synthesizing a less conservative backup pair via finite horizon sum-of-squares (SOS) backward reachability. Starting from an initial backup set, we compute an SOS-certified finite horizon backward reachable set and controller that satisfy safety and input constraints while steering trajectories to the original backup set. We then provide conditions under which this certified set becomes a valid backup set for a piecewise backup controller. The resulting backup pair is integrated into the bCBF framework to certify larger safe sets. |
| 2026-08-28 | [From Uncertainty to Clinical Risk: Severity-Aware Conformal Planning for Interactive Medical Diagnosis](http://arxiv.org/abs/2608.27847v1) | Yue Zhou, Haiyang Zhou et al. | Interactive medical diagnosis dynamically acquires patient information through multiple rounds of questioning, supporting accurate, efficient, and safe clinical decisions under incomplete evidence. Existing methods commonly guide information acquisition with predictive uncertainty or label ambiguity, but overlook the asymmetric clinical risk of missing severe diseases and lack unified long-horizon planning over whether to continue asking questions or commit to a diagnosis. To address these limitations, we propose Severity-Aware Conformal Clinical Planning, which formulates interactive diagnosis as a risk-sensitive sequential decision problem. The framework maintains complementary diagnostic, safety, and masked-evidence beliefs; calibrates turn-specific diagnostic prediction sets and severity-weighted differential-diagnosis risk on held-out diagnostic trajectories; and introduces the calibrated clinical risk into Monte Carlo Tree Search to jointly evaluate long-horizon Ask and Commit trajectories. Experiments on DDXPlus and MediQ show that our method achieves more accurate diagnoses with fewer questions across multiple large language models, while improving differential-diagnosis quality and reducing high-risk errors in severe cases. These findings validate the value of using clinical risk, rather than predictive uncertainty alone, as a planning signal and demonstrate the effectiveness of the proposed framework for information acquisition and risk-aware diagnostic decision making. They also motivate future work on clinical-risk-oriented interactive diagnosis and information-acquisition methods. |

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



