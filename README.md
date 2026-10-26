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
| 2026-10-23 | [When Human Corrections Travel: Safety in Federated VLA Adaptation](http://arxiv.org/abs/2610.32888v1) | Vigneshwaran Subbaraju, Obaidullah Zaland et al. | Vision-language-action (VLA) policies can adapt to local human intervention, while federated learning can combine distributed robot experience without pooling raw records. We examine their prospective intersection: sharing adaptations derived from diverse human corrections across tasks, environments, people, and robot embodiments. Corrections differ in modality, safety semantics(urgency, behavioral scope, etc.), and privacy sensitivity. A shared update that improves mean task success may suppress a rare safety intervention or produce an unsafe response on another robot. We argue for three requirements: preserve the context and uncertainty of multimodal corrections when deciding what to share; assess candidate updates in each receiving robot's closed loop before deployment; and evaluate leakage alongside retention of safety-critical evidence. Immediate, legible defensive behavior remains necessary during human interaction. We offer counterexamples and a falsifiable evaluation agenda, not experimental results. |
| 2026-10-23 | [Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery](http://arxiv.org/abs/2610.32866v1) | Marek Masiak, Andrzej Szablewski et al. | Fine-tuning on datasets of unknown origin can create a backdoor in the trained model when the data includes poisoned examples. Difficult to localise and remove, backdoors can persist through safety training. While parameter-efficient fine-tuning methods capture the full parameter changes introduced during training, they are not easily interpretable. We present the full method of TopKLoRA, extending the preliminary proposal of Masiak et al. (2025), an interpretable-by-design LoRA variant in which only the $k$ largest adapter latents are active at each token. Unlike SAEs, TopKLoRA is trained purely on downstream loss, where optimisation pressure incentivises learning functionally causal latents. We demonstrate this sparse LoRA adapter in a sleeper-agent backdoor case study where we create model organisms by fine-tuning gemma-2-2b, Qwen2.5-1.5B, and Qwen2.5-7B on a poisoned instruction-following dataset. For each organism, we identify and ablate a circuit that is both necessary and sufficient to implement the backdoor. Furthermore, our new fine-tuning method produces sparse circuits that are up to an order of magnitude smaller than a dense LoRA baseline, allowing surgical ablation of the malicious capability without sacrificing instruction-following performance. Finally, we propose a technique to construct model organisms with localised circuits using TopKLoRA latents and gradient-routing, which we use to validate the findings of our circuit search algorithm. We consider these findings as a step toward interpretable-by-design architectures that provide faithful units of analysis without compromising model performance. We share our code: https://github.com/interpretable-finetuning/TopKLoRA. |
| 2026-10-23 | [Quantum Cadence: Post-Quantum Consensus with Multiple Concurrent Proposers](http://arxiv.org/abs/2610.32803v1) | Mohammad Mussadiq Jalalzai, Tobias Klenze et al. | The consensus protocols of most high-performance blockchains use quorum certificates, which bundle the signatures of a quorum of validators. Signature schemes such as BLS can aggregate the signatures of a whole quorum into a single short signature, which keeps these certificates small. A large-scale quantum computer would break these schemes, and no post-quantum scheme with comparable aggregate size and aggregation time is known. Without aggregation, a certificate holds one large post-quantum signature per member of the quorum, so it becomes infeasible for each validator to broadcast certificates, as modern consensus protocols often require.   We present Quantum Cadence, a post-quantum consensus protocol with multiple concurrent proposers that avoids certificates altogether: every validator counts quorums among the messages it has received itself. Its safety and liveness do not even depend on signatures, although we add signatures to obtain accountability. Quantum Cadence builds on Cadence and replaces its slot consensus, Chorus, with Quantum Chorus, which keeps the optimal fast-path latency of Chorus: dissemination followed by two rounds of votes. When its fast path fails, Quantum Chorus runs Coda, our signature-free Byzantine agreement for multiple proposers. We also outline how the Conductor, which schedules the slots of Cadence, can be adapted to Quantum Cadence. In simulations with 200 validators, Quantum Chorus has a latency comparable to that of Chorus and scales better with the number of proposers. |
| 2026-10-23 | [Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks](http://arxiv.org/abs/2610.32765v1) | Yalin E. Sagduyu, Tugba Erpek et al. | This paper studies interactions between clean-label poisoning attack and defense in vehicular networks using federated learning (FL). Basic Safety Message (BSM) classification for anomaly detection is modeled as a distributed learning problem over a vehicular network with dynamic topology. As a critical threat to FL, a feature-space clean-label attack is introduced, wherein benign samples are perturbed toward malicious feature extremes, degrading class separability while preserving labels. To counter this, a feature-consistency relabeling defense is proposed to detect and relabel suspicious samples. A game-theoretic framework is developed to capture attack-defense interactions and characterize equilibrium strategies based on attack effectiveness and cost parameters. Results show that the attack scales effectively with compromised nodes, while the proposed defense keeps attack success low without degrading clean accuracy and shifts equilibrium regions toward attack-averse regimes. |
| 2026-10-23 | [Semantic Representation for Agentic Self-Play in Automated Laboratories](http://arxiv.org/abs/2610.32755v1) | Yves Delley, John Coller et al. | Navigating complex, high-dimensional biological search spaces necessitates scalable physical and computational intelligence to explore beyond local optima through iterative self-play across ideation, experimentation, and automation. Translating experimental designs into reliable physical execution remains bottlenecked by static laboratory automation and unmodeled physical variability. To address this, we introduce VERified Infrastructure for Discovery Laboratories, or VERID, a foundational infrastructure designed to bridge the gap between the high-level reasoning of artificial intelligence (AI) agents and complex, error-prone physical realities of wet labs. VERID is a persistent, high-fidelity semantic digital twin of the laboratory workcell that provides four core services: queryable introspection for agentic reasoning, time-compressed simulation for protocol validation, comprehensive trace persistence for a complete scientific record, and a framework for safe execution.   We present three case studies using aqueous dye and deionized water mixtures to validate the optical and fluidic foundations of VERID. These studies demonstrate how VERID enables pre-emptive detection of critical errors in experimental protocols, stateful mid-run error recovery and post-hoc identification of subtle physical failures. The simulation functions as a discrete accounting engine, allowing agents to safely and inexpensively explore experimental designs in silico and refine experimental strategies through iterative self-play. VERID also captures structured multimodal experimental data for downstream analysis, knowledge mining and agent training. We argue that this semantic layer is an essential prerequisite for deploying autonomous agents in the biochemical and biomedical sciences, ensuring data integrity, operational safety, and paving the way for tackling previously intractable grand challenges in biology. |
| 2026-10-23 | [Cross-Scenario Controller Parameter Transformation from Real-World Driving Data through Flow Matching](http://arxiv.org/abs/2610.32719v1) | Christian Zink, Maik Pfefferkorn et al. | Advanced Driver Assistance Systems have become increasingly widespread, aiming to support drivers, reduce workload, and improve traffic safety. However, many systems are rarely used because their behavior feels unnatural and does not match individual driving behavior. The growing availability of large-scale, real-world human driving data enables more human-like and driver-individual assistance functions. This paper formulates the adaptation of Adaptive Cruise Control parameters as a driver-individual scenario-to-scenario transport problem, transforming controller parameters from one driving context to another. Using flow matching, we train a generative model to predict the target-scenario behavior of previously unseen drivers from only a few demonstrations in a source scenario. We compare the approach with a linear parameter-shift baseline and evaluate flow-matching variants incorporating driver-consistent coupling, driver embeddings, and classifier-free guidance based on driving profiles extracted from trajectory data using a variational autoencoder. |
| 2026-10-23 | [The Standard Deviation in VICReg Buys a Safety Margin](http://arxiv.org/abs/2610.32630v1) | Hawar Dizayee | VICReg prevents collapse with a hinge on the standard deviation of each embedding dimension, and its authors argue from the gradient that a hinge on the variance would fail. To our knowledge, this claim has not been tested experimentally. We test it in a controlled setting (ResNet-18, CIFAR-10 and CIFAR-100), changing only the hinge and sweeping its weight $μ$ over five orders of magnitude. At the default weight ($μ= 1$), the two hinges reach the same linear-probe accuracy on CIFAR-10 (about 89.5%). Both collapse below a threshold weight, but the standard-deviation hinge's threshold is 35-130$\times$ lower. CIFAR-100 shows a consistent gap. In training, the variance hinge's gradient fades as the feature variance shrinks, while the standard-deviation hinge's does not until that variance is very small. The standard-deviation hinge therefore buys a safety margin: it keeps training at weights far below the point where the variance hinge collapses. |
| 2026-10-23 | [ALLMTS: Automated LLM-based Medical Triage System](http://arxiv.org/abs/2610.32469v1) | Michał Remigiusz Janiszewski, Artur Stopa et al. | Telephone triage is a safety critical task: from a brief intake note, a nurse must decide whether a caller needs an ambulance, an emergency department, an appointment, or self care at home. Queried directly, large language models (LLMs) are unreliable: they over-escalate routine cases and cannot justify their recommendations against an auditable clinical standard. We present the Automated LLM-based Medical Triage System (ALLMTS), which grounds every disposition in the same commercially maintained telehealth guidelines nurses use. Rather than asking an LLM to triage, ALLMTS makes it execute the guideline: it selects the appropriate guideline and evaluates its triage assessment questions. The disposition follows deterministically from the first positive answer, using the same traversal rule a nurse follows. We evaluate ALLMTS on 566 synthetic, expert-validated scenarios spanning 14 dispositions and designed to test specific guideline questions. ALLMTS achieves 0.95 accuracy, compared with 0.47 for a raw-LLM baseline. On a curated 50-scenario subset released as an open benchmark, ALLMTS reaches 0.98 accuracy, on par with the 0.944 correct disposition rate that a five nurse expert panel recorded on those same scenarios. Critically, on that subset the baselines have over-triage rates of up to 0.50 (0.44 on the full dataset), whereas ALLMTS reduces the over-triage rate to at most 0.02 while maintaining low under-triage. These results support coupling LLM language understanding with executable clinical guidelines for triage from intake notes. |
| 2026-10-23 | [Graphs of Trajectory Bundles for Nonlinear Motion Planning](http://arxiv.org/abs/2610.32365v1) | Joris Verhagen, Jon Arrizabalaga et al. | We present the Graphs of Trajectory Bundles (GTB) method for motion planning of systems with nonlinear and black-box dynamics, combining sampling-based motion planning with trajectory optimization and graph search. Based on a graph-of-convex-sets decomposition of the workspace, GTB iteratively constructs a graph of trajectory bundles that captures the local dynamics of the system at any vertex. By solving a shortest-path problem on this graph, we find maximally dynamically feasible iterates that satisfy the system's dynamics in the limit and any constraints such as collision avoidance or additional safety or specification constraints by construction of the underlying graph. As a result, GTB can handle highly nonlinear, unstable, and even black-box dynamics with complex collision and specification constraints, making it a powerful tool for motion planning of complex systems. |
| 2026-10-23 | [Detection and Validation of Hidden Subgroup Disparities in Clinical AI](http://arxiv.org/abs/2610.32362v1) | Alceu Bissoto, Trung-Dung Hoang et al. | Machine learning (ML) models used in clinical workflows often perform systematically differently across patient groups, raising concerns about equity and patient safety. Traditional subgroup analyses can expose such disparities only when rich metadata (e.g., patient demographics) is available and captures the sources of performance variation, conditions rarely met in practice. Algorithmic subgroup discovery methods offer a solution by identifying coherent groups from learned feature representations, but because true underlying stratifications are unknown, these methods are hard to evaluate. As a result, subgroup discovery has not been systematically applied or assessed for performance monitoring. Our main objective is to evaluate algorithmic subgroup discovery for performance monitoring, assessing the discovered subgroups' performance gaps and semantic coherence. Investigating cardiomegaly classification in a dataset of 223,228 chest X-rays, we showed that the discovered subgroups exhibited larger performance gaps (accuracies of 0.61-0.90) than traditional metadata-based analysis (0.80-0.94) and were frequently interpretable using patient, acquisition, and model metadata. As a proof of concept, we also showed that annotations from learned subgroups can be used to mitigate the biases they expose. We conclude that subgroup discovery is a practical and effective tool to enhance the performance validation and monitoring of ML systems in healthcare. |
| 2026-10-23 | [FOCUS: A Lightweight Mamba-2 Model for EEG-Based Cognitive Workload Classification](http://arxiv.org/abs/2610.32361v1) | Ioannis Kyprakis, Stefanos Gkikas et al. | Cognitive workload estimation from physiological signals is increasingly relevant in high-risk environments, where continuous and unobtrusive monitoring can inform adaptive systems and safety interventions. This paper presents FOCUS (Frequency-band Optimized Cognitive-load Ultralight SSM), a compact Mamba-2-based model for EEG-based binary workload classification under constrained data conditions. Rather than learning frequency sensitivity from data, FOCUS encodes preprocessed multi-channel EEG using a fixed FFT decomposition that covers the five canonical EEG bands. The proposed model combines depthwise-separable convolution and a weight-shared encoder-bottleneck Mamba-2 pair, with 0.04 million parameters and 0.11 GFLOPs. Evaluated in a pilot study with 11 participants under leave-one-subject-out cross-validation across three cognitively distinct tasks, FOCUS achieves 74.75% accuracy on the full dataset at 0.08 ms GPU model-forward latency. In the weight-sharing ablation, the shared configuration achieves 4.55 percentage points higher accuracy on the full dataset than the non-shared variant while reducing the SSM parameter budget; however, this difference does not reach statistical significance under a two-sided Wilcoxon signed-rank test ($p=0.083$). Overall, the selected FOCUS configuration provides a favorable trade-off between classification performance and computational cost within this controlled pilot setting. |
| 2026-10-23 | [Empirically Calibrated Differential Games for Simulating Driver Interactions](http://arxiv.org/abs/2610.32276v1) | Amirmohammad Khakpour, Nachuan Li et al. | Game-theoretic models can resolve vehicle interactions without hand-written priority rules, but whether they describe human drivers once calibrated has rarely been tested. This study formulates a general-sum differential game of vehicle interactions at urban intersections, calibrates it to observed driving, and evaluates it on data not used for calibration. Each vehicle minimizes a cost that combines speed tracking, adherence to a map-derived lane route, control effort, gap keeping based on the intelligent driver model, a predictive conflict term, and a contact barrier, while perceiving the other vehicles with a delay. The game is solved with a feedback iterative linear-quadratic algorithm, and ten behavioral parameters per vehicle are calibrated by Bayesian optimization on trajectories recorded at four intersections in Washington, D.C. The calibrated game reproduces the trajectories of 819 vehicles with a mean position error of 1.00 m. When only each vehicle's route is known, as in a simulator, it is the most accurate of the compared models on 96 held-out vehicles (7.42 m), and ten-fold cross-validation shows that it is significantly more accurate than calibrated SUMO (by 3.61 m), an inverse reinforcement learning model (by 0.93 m), and independent and cooperative variants of the same model (by 0.33 and 0.32 m). The solutions, computed in 2.1 s for a typical interaction, are verified approximate equilibria. Finally, the conflict and steering weights, which the data do not identify, can be set from safety and comfort targets; together with a stiffer contact barrier, this removes the collisions that otherwise occur in 43% of tight closed-loop conflicts. |
| 2026-10-23 | [Empirical concentration for chains with infinite memory](http://arxiv.org/abs/2610.32265v1) | Pierre Alquier, Paul Doukhan | Concentration inequalities for time series usually involve unknown quantities measuring dependence. This prevents their direct use as confidence bounds, or as criteria for learning algorithms. Recent works proved empirical concentration inequalities for stationary Markov chains, where the unknown quantities are estimated. We extend this approach beyond the Markov case, by considering stationary chains with infinite memory. We first establish a bounded-differences inequality on bounded Polish metric spaces under a summable Wasserstein contraction condition. We then estimate its dependence coefficients from a single trajectory, using either transition counts when the state space is finite, or derivatives of kernel conditional-density estimates for a continuous state space. We give general error bounds and illustrate their dependence on the memory tail. Finally, a fixed multiplicative safety factor turns these estimators into empirical concentration bounds. While some regularity assumptions are required to make the bounds empirical, our procedure remains fully adaptive in the sense that it does not require the knowledge of the constants involved in these assumptions. The empirical confidence bounds derived from our results are asymptotically as tight as the theoretical ones. |
| 2026-10-23 | [Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection](http://arxiv.org/abs/2610.32255v1) | Donghwan Kim, Youngwoong Jin et al. | Out-of-distribution (OOD) detection is recognized as a critical problem in AI safety. SCALE is an effective post-hoc OOD detection method in the activation-shaping family that requires no training data and rescales penultimate activations using a percentile-based magnitude ratio. However, this ratio is ill-defined when negative activations are present, since positive and negative contributions cancel within the underlying sum, causing SCALE to degenerate on backbones such as ViT. We address this with an Absolute Fix that computes the ratio over absolute-valued activations rather than discarding negative activations through ReLU. Building on this corrected scale, we propose Anti-coherence regularization, which penalizes the scaling factor by the fraction of a sample's max-logit contributions to the max logit that are sign-mismatched and thereby suppresses unreliable evidence for the predicted class. Despite requiring no access to training data or training labels, our method ties or outperforms activation-shaping methods that rely on training data in terms of AUROC and FPR@95 on CIFAR-10 and ImageNet under the OpenOOD v1.5 protocol. |
| 2026-10-23 | [Autonomous Cyber Weapons as AI-Enabled Digital Harms: A Platform Accountability Framework](http://arxiv.org/abs/2610.32238v1) | Mansi Ganga Nayak, Samarjeet Malik et al. | Governments govern AI-generated online harms from commercial platforms (EU DSA, AI Act, UK Online Safety Act), yet the most severe class remains ungoverned: autonomous cyber weapons (ACWs) deployed by nation-states against civilian infrastructure. We address this gap with three contributions: a digital-harm taxonomy (six ACW modes $\times$ five harm categories); the Meaningful Human Control (MHC) Model, a governance-compliance function with a closed-form IHL threshold and a proved weight-sensitivity bound; and a platform accountability architecture with treaty language grounded in ARSIWA state-responsibility doctrine, not Rome Statute individual responsibility. Applied to eleven documented state-attributed operations and one hypothetical defensive AI system, every documented case falls in an accountability vacuum; the threshold is target-dependent, and against wholly civilian targets no system studied clears it, including the hypothetical. Three further incidents are correctly excluded on autonomy/attribution grounds, a discriminant-validity check. The weights and threshold parameters are illustrative, not fitted. |
| 2026-10-23 | [Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems](http://arxiv.org/abs/2610.32208v1) | Christian A. Rogers, Qian Lin et al. | Vision-Language-Action (VLA) models have emerged as powerful general-purpose robot policies, but their successes and failures remain difficult to diagnose and understand. In this work, we show that sparse autoencoder (SAE) representations of OpenVLA activations contain outcome-predictive information and can be used to predict task failure before the robot reaches an unrecoverable state. Moreover, to provide insights into what common behaviors may contribute to these outcomes, we identify individual SAE features whose strong activations are associated with specific stages of success or failure, and uncover recurring behavioral patterns by localizing temporal intervals where these features exhibit significant activation changes. We further develop a VLM-based interpretation pipeline to generate behavioral descriptions of these patterns and validate their correspondence to feature activations on held-out intervals. |
| 2026-10-23 | [Evidence-Grounded SLM Behavioral Prediction Architecture for Future Assistance Use](http://arxiv.org/abs/2610.32193v1) | Adeyemi Adeseye, Aisvarya Adeseye et al. | Users often exhibit changes in interaction behavior before explicitly seeking assistance, enabling systems to anticipate future assistance use. This is especially important in high-stakes environments, where delayed assistance can have operational, financial, or safety consequences. However, conventional Large Language Models (LLMs) are often unsuitable for continuous on-device behavioral prediction because of their computational and memory demands. Behavioral patterns are also highly individual and device-specific, motivating local processing that preserves privacy and data sovereignty. Small Language Models (SLMs) offer a promising alternative, but their ability to reason reliably over low-level sequential interaction events remains underexplored. This study introduces Evidence-Grounded Local Behavioral Prediction (ELBP), a four-layer architecture combining deterministic interaction processing, dual-view behavioral representation, SLM-based prediction, and evidence verification with abstention. Three instruction-tuned models of 3B parameters or fewer were evaluated using raw, structured, dual-view, and complete ELBP configurations. The best-performing model, Qwen2.5-3B with ELBP, achieved a full-set F1-score of 0.80 and PR-AUC of 0.84, representing a 15.9\% relative improvement over raw-event prompting and 17.6\% over XGBoost, while full-set ECE decreased by 43.2\% relative to raw-event prompting. With evidence verification and abstention, selective F1 increased to 0.84 at 91\% coverage and evidence support reached 96\%. These results show that structured behavioral representation and evidence-verified selective prediction can improve the reliability of behavioral forecasting in SLMs. |
| 2026-10-23 | [Depth-Dependent Safety Geometry in Vision-Language Models: Layer-Wise Localization and Bidirectional Refusal Control](http://arxiv.org/abs/2610.32182v1) | Muhammad Zeeshan Karamat, Syed Talal Hassan et al. | Vision-Language Models (VLMs) are widely deployed, yet the internal mechanisms behind their safety behavior remain poorly understood. We present a lightweight inference-time framework for localizing and manipulating safety-sensitive representations in frozen VLMs, without retraining or modifying parameters. On Qwen3-VL-4B, layer-wise activation divergence analysis reveals non-uniform safety geometry across depth. Causal interventions show that early vision layers support strong bidirectional refusal control through activation steering along a learned safety direction, with unsafe rejection spanning 6%-96%, whereas later layers respond more to activation magnitude scaling and exhibit a substantial safety-selectivity trade-off. Deeper layers remain well separated along the safety direction yet are largely insensitive to steering, distinguishing representational separability from causal controllability. InternVL2.5-4B shows qualitatively similar depth-dependent activation geometry despite differences in layer location and magnitude. Our results expose a white-box, representation-level refusal-bypass attack surface in VLM vision encoders and motivate inference-time stabilization of safety-sensitive representations. |
| 2026-10-23 | [Detecting Insecure Compositions of Clean Agent Skills via Capability-Flow Graph Analysis](http://arxiv.org/abs/2610.32155v1) | Yihao Li, Chuanpu Fu et al. | AI agents increasingly extend their capabilities by invoking third-party skills, where each skill encapsulates instructions, scripts, and resources for agents to execute on demand. This paper reveals that deploying many benign skills together introduces compositional security risks: multiple individually clean skills can compose into safety-violation chains, because the agent passes an upstream skill's output to a downstream skill's input across invocations. The risk arises not from any single skill, but from the agent's runtime orchestration, and thus cannot be captured by existing malicious skill detection methods. To this end, we develop KnotScryer to detect insecure skill compositions by treating skills as typed capability flows. In particular, we abstract each skill into a capability summary, and construct the Skill Capability-Flow Graph (SCFG), where edges denote cross-skill information flow. Moreover, we design a search policy that tracks protected or untrusted data along handoff edges to locate paths reaching insecure operations. Finally, a two-layer verification method replays the insecure information flow for reachability analysis and applies a task-anchored LLM chain audit to confirm the semantic plausibility of the identified violation chains. We deploy KnotScryer to analyze 13,598 clean skills from ClawHub, and validate the detected chains across three agent frameworks and three LLMs. KnotScryer identifies 612 insecure compositions, which are consistently exploited across agent frameworks. Meanwhile, under GLM-5.1, it outperforms tag-match and graph-only baselines by improving the exploitation rate from 16.5%-18.2% to 85.6%, while reducing the search space by 85.3%. Furthermore, 13.7% of the detected chains consist entirely of skills labeled safe by a single-skill scanner, showing that KnotScryer effectively identifies insecure compositions that per-skill auditing cannot. |
| 2026-10-23 | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](http://arxiv.org/abs/2610.32113v1) | Hanwen Zhang, Enyi Jiang et al. | Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. This separates risk readability from effective control: intervention timing must be evaluated through its effect on subsequent generation. On the same 8B model, REACT raises the rate of fully safe CoT-answer outputs on StrongREJECT-273 from 29.06% to 82.05%, while keeping benign refusal and usefulness near baseline. An anonymized implementation is available at https://anonymous.4open.science/r/REACT_official2-333B. |

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



