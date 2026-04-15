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
| 2026-04-14 | [LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](http://arxiv.org/abs/2604.12994v1) | Syed Md Mukit Rashid, Abdullah Al Ishtiaq et al. | Logical vulnerabilities in software stem from flaws in program logic rather than memory safety, which can lead to critical security failures. Although existing automated program repair techniques primarily focus on repairing memory corruption vulnerabilities, they struggle with logical vulnerabilities because of their limited semantic understanding of the vulnerable code and its expected behavior. On the other hand, recent successes of large language models (LLMs) in understanding and repairing code are promising. However, no framework currently exists to analyze the capabilities and limitations of such techniques for logical vulnerabilities. This paper aims to systematically evaluate both traditional and LLM-based repair approaches for addressing real-world logical vulnerabilities. To facilitate our assessment, we created the first ever dataset, LogicDS, of 86 logical vulnerabilities with assigned CVEs reflecting tangible security impact. We also developed a systematic framework, LogicEval, to evaluate patches for logical vulnerabilities. Evaluations suggest that compilation and testing failures are primarily driven by prompt sensitivity, loss of code context, and difficulty in patch localization. |
| 2026-04-14 | [Parallax: Why AI Agents That Think Must Never Act](http://arxiv.org/abs/2604.12986v1) | Joel Fokou | Autonomous AI agents are rapidly transitioning from experimental tools to operational infrastructure, with projections that 80% of enterprise applications will embed AI copilots by the end of 2026. As agents gain the ability to execute real-world actions (reading files, running commands, making network requests, modifying databases), a fundamental security gap has emerged. The dominant approach to agent safety relies on prompt-level guardrails: natural language instructions that operate at the same abstraction level as the threats they attempt to mitigate. This paper argues that prompt-based safety is architecturally insufficient for agents with execution capability and introduces Parallax, a paradigm for safe autonomous AI execution grounded in four principles: Cognitive-Executive Separation, which structurally prevents the reasoning system from executing actions; Adversarial Validation with Graduated Determinism, which interposes an independent, multi-tiered validator between reasoning and execution; Information Flow Control, which propagates data sensitivity labels through agent workflows to detect context-dependent threats; and Reversible Execution, which captures pre-destructive state to enable rollback when validation fails. We present OpenParallax, an open-source reference implementation in Go, and evaluate it using Assume-Compromise Evaluation, a methodology that bypasses the reasoning system entirely to test the architectural boundary under full agent compromise. Across 280 adversarial test cases in nine attack categories, Parallax blocks 98.9% of attacks with zero false positives under its default configuration, and 100% of attacks under its maximum-security configuration. When the reasoning system is compromised, prompt-level guardrails provide zero protection because they exist only within the compromised system; Parallax's architectural boundary holds regardless. |
| 2026-04-14 | [Probabilistic Feature Imputation and Uncertainty-Aware Multimodal Federated Aggregation](http://arxiv.org/abs/2604.12970v1) | Nafis Fuad Shahid, Maroof Ahmed et al. | Multimodal federated learning enables privacy-preserving collaborative model training across healthcare institutions. However, a fundamental challenge arises from modality heterogeneity: many clinical sites possess only a subset of modalities due to resource constraints or workflow variations. Existing approaches address this through feature imputation networks that synthesize missing modality representations, yet these methods produce point estimates without reliability measures, forcing downstream classifiers to treat all imputed features as equally trustworthy. In safety-critical medical applications, this limitation poses significant risks. We propose the Probabilistic Feature Imputation Network (P-FIN), which outputs calibrated uncertainty estimates alongside imputed features. This uncertainty is leveraged at two levels: (1) locally, through sigmoid gating that attenuates unreliable feature dimensions before classification, and (2) globally, through Fed-UQ-Avg, an aggregation strategy that prioritizes updates from clients with reliable imputation. Experiments on federated chest X-ray classification using CheXpert, NIH Open-I, and PadChest demonstrate consistent improvements over deterministic baselines, with +5.36% AUC gain in the most challenging configuration. |
| 2026-04-14 | [Output-Feedback Safe Control of Discrete-Time Stochastic Systems with Chance Constraints](http://arxiv.org/abs/2604.12956v1) | Jianing Zhao, Zhuoting Cai et al. | In this paper, we investigate safety-critical control problem of discrete-time stochastic systems with incomplete information, where safety constraints must be enforced using state estimates obtained from noisy measurements. We develop an output-feedback control barrier function (CBF) framework based on an expectation-based discrete-time barrier condition that explicitly incorporates estimation uncertainty through the evolving belief over the state. To enable real-time implementation, we derive deterministic sufficient conditions that conservatively enforce the expectation-based CBF by bounding the expectation with computable functions of the belief statistics using Jensen inequalities. The resulting safety filter is formulated as a tractable optimization problem compatible with standard online controllers. Numerical simulations demonstrate that the proposed output-feedback approach achieves fast online computation while providing reliable safety performance in the presence of process noise and measurement uncertainty. |
| 2026-04-14 | [AISafetyBenchExplorer: A Metric-Aware Catalogue of AI Safety Benchmarks Reveals Fragmented Measurement and Weak Benchmark Governance](http://arxiv.org/abs/2604.12875v1) | Abiodun A. Solanke | The rapid expansion of large language model (LLM) safety evaluation has produced a substantial benchmark ecosystem, but not a correspondingly coherent measurement ecosystem. We present AISafetyBenchExplorer, a structured catalogue of 195 AI safety benchmarks released between 2018 and 2026, organized through a multi-sheet schema that records benchmark-level metadata, metric-level definitions, benchmark-paper metadata, and repository activity. This design enables meta-analysis not only of what benchmarks exist, but also of how safety is operationalized, aggregated, and judged across the literature. Using the updated catalogue, we identify a central structural problem: benchmark proliferation has outpaced measurement standardization. The current landscape is dominated by medium-complexity benchmarks (94/195), while only 7 benchmarks occupy the Popular tier. The workbook further reports strong concentration around English-only evaluation (165/195), evaluation-only resources (170/195), stale GitHub repositories (137/195), stale Hugging Face datasets (96/195), and heavy reliance on arXiv preprints among benchmarks with known venue metadata. At the metric level, the catalogue shows that familiar labels such as accuracy, F1 score, safety score, and aggregate benchmark scores often conceal materially different judges, aggregation rules, and threat models. We argue that the field's main failure mode is fragmentation rather than scarcity. Researchers now have many benchmark artifacts, but they often lack a shared measurement language, a principled basis for benchmark selection, and durable stewardship norms for post publication maintenance. AISafetyBenchExplorer addresses this gap by providing a traceable benchmark catalogue, a controlled metadata schema, and a complexity taxonomy that together support more rigorous benchmark discovery, comparison, and meta-evaluation. |
| 2026-04-14 | [Understanding and Improving Continuous Adversarial Training for LLMs via In-context Learning Theory](http://arxiv.org/abs/2604.12817v1) | Shaopeng Fu, Di Wang | Adversarial training (AT) is an effective defense for large language models (LLMs) against jailbreak attacks, but performing AT on LLMs is costly. To improve the efficiency of AT for LLMs, recent studies propose continuous AT (CAT) that searches for adversarial inputs within the continuous embedding space of LLMs during AT. While CAT has achieved empirical success, its underlying mechanism, i.e., why adversarial perturbations in the embedding space can help LLMs defend against jailbreak prompts synthesized in the input token space, remains unknown. This paper presents the first theoretical analysis of CAT on LLMs based on in-context learning (ICL) theory. For linear transformers trained with adversarial examples from the embedding space on in-context linear regression tasks, we prove a robust generalization bound that has a negative correlation with the perturbation radius in the embedding space. This clearly explains why CAT can defend against jailbreak prompts from the LLM's token space. Further, the robust bound shows that the robustness of an adversarially trained LLM is closely related to the singular values of its embedding matrix. Based on this, we propose to improve LLM CAT by introducing an additional regularization term, which depends on singular values of the LLM's embedding matrix, into the objective function of CAT. Experiments on real-world LLMs demonstrate that our method can help LLMs achieve a better jailbreak robustness-utility tradeoff. The code is available at https://github.com/fshp971/continuous-adv-icl. |
| 2026-04-14 | [Fragile Reconstruction: Adversarial Vulnerability of Reconstruction-Based Detectors for Diffusion-Generated Images](http://arxiv.org/abs/2604.12781v1) | Haoyang Jiang, Mingyang Yi et al. | Recently, detecting AI-generated images produced by diffusion-based models has attracted increasing attention due to their potential threat to safety. Among existing approaches, reconstruction-based methods have emerged as a prominent paradigm for this task. However, we find that such methods exhibit severe security vulnerabilities to adversarial perturbations; that is, by adding imperceptible adversarial perturbations to input images, the detection accuracy of classifiers collapses to near zero. To verify this threat, we present a systematic evaluation of the adversarial robustness of three representative detectors across four diverse generative backbone models. First, we construct adversarial attacks in white-box scenarios, which degrade the performance of all well-trained detectors. Moreover, we find that these attacks demonstrate transferability; specifically, attacks crafted against one detector can be transferred to others, indicating that adversarial attacks on detectors can also be constructed in a black-box setting. Finally, we assess common countermeasures and find that standard defense methods against adversarial attacks provide limited mitigation. We attribute these failures to the low signal-to-noise ratio (SNR) of attacked samples as perceived by the detectors. Overall, our results reveal fundamental security limitations of reconstruction-based detectors and highlight the need to rethink existing detection strategies. |
| 2026-04-14 | [GF-Score: Certified Class-Conditional Robustness Evaluation with Fairness Guarantees](http://arxiv.org/abs/2604.12757v1) | Arya Shah, Kaveri Visavadiya et al. | Adversarial robustness is essential for deploying neural networks in safety-critical applications, yet standard evaluation methods either require expensive adversarial attacks or report only a single aggregate score that obscures how robustness is distributed across classes. We introduce the \emph{GF-Score} (GREAT-Fairness Score), a framework that decomposes the certified GREAT Score into per-class robustness profiles and quantifies their disparity through four metrics grounded in welfare economics: the Robustness Disparity Index (RDI), the Normalized Robustness Gini Coefficient (NRGC), Worst-Case Class Robustness (WCR), and a Fairness-Penalized GREAT Score (FP-GREAT). The framework further eliminates the original method's dependence on adversarial attacks through a self-calibration procedure that tunes the temperature parameter using only clean accuracy correlations. Evaluating 22 models from RobustBench across CIFAR-10 and ImageNet, we find that the decomposition is exact, that per-class scores reveal consistent vulnerability patterns (e.g., ``cat'' is the weakest class in 76\% of CIFAR-10 models), and that more robust models tend to exhibit greater class-level disparity. These results establish a practical, attack-free auditing pipeline for diagnosing where certified robustness guarantees fail to protect all classes equally. We release our code on \href{https://github.com/aryashah2k/gf-score}{GitHub}. |
| 2026-04-14 | [Reliability-Guided Depth Fusion for Glare-Resilient Navigation Costmaps](http://arxiv.org/abs/2604.12753v1) | Shang-En Tsai, Wei-Cheng Sun | Specular glare on reflective floors and glass surfaces frequently corrupts RGB-D depth measurements, producing holes and spikes that accumulate as persistent phantom obstacles in occupancy-grid costmaps. This paper proposes a glare-resilient costmap construction method based on explicit depth-reliability modeling. A lightweight Depth Reliability Map (DRM) estimator predicts per-pixel measurement trustworthiness under specular interference, and a Reliability-Guided Fusion (RGF) mechanism uses this signal to modulate occupancy updates before corrupted measurements are accumulated into the map. Experiments on a real mobile robotic platform equipped with an Intel RealSense D435 and a Jetson Orin Nano show that the proposed method substantially reduces false obstacle insertion and improves free-space preservation under real reflective-floor and glass-surface conditions, while introducing only modest computational overhead. These results indicate that treating glare as a measurement-reliability problem provides a practical and lightweight solution for improving costmap correctness and navigation robustness in safety-critical indoor environments. |
| 2026-04-14 | [Short Version of VERIFAI2026 Paper -- Learning Infused Formal Reasoning: Contract Synthesis, Artefact Reuse and Semantic Foundations](http://arxiv.org/abs/2604.12747v1) | Arshad Beg, Diarmuid O'Donoghue et al. | Artificial intelligence systems have achieved remarkable capability in natural language processing, perception and decision-making tasks. However, their behaviour often remains opaque and difficult to verify, limiting their applicability in safety-critical systems. Formal methods provide mathematically rigorous mechanisms for specifying and verifying system behaviour, yet the creation and maintenance of formal specifications remains labour intensive and difficult to scale. This paper outlines a research vision called Learning-Infused Formal Reasoning (LIFR), which integrates machine learning techniques with formal verification workflows. The framework focuses on three complementary research directions: automated contract synthesis from natural language requirements, semantic reuse of verification artifacts using graph matching and learning-based embeddings, and mathematically grounded semantic foundations based on the Unifying Theories of Programming (UTP) and the Theory of Institutions. Together these research threads aim to transform verification from isolated correctness proofs into a cumulative knowledge-driven process where specifications, contracts and proofs can be synthesised, aligned and reused across systems. |
| 2026-04-14 | [Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](http://arxiv.org/abs/2604.12719v1) | Adam T. Müller, Tobias Rögelein et al. | The deployment of deep neural networks in safety-critical systems necessitates reliable and efficient uncertainty quantification (UQ). A practical and widespread strategy for UQ is repurposing stochastic regularizers as scalable approximate Bayesian inference methods, such as Monte Carlo Dropout (MCD) and MC-DropBlock (MCDB). However, this paradigm remains under-explored for Stochastic Depth (SD), a regularizer integral to the residual-based backbones of most modern architectures. While prior work demonstrated its empirical promise for segmentation, a formal theoretical connection to Bayesian variational inference and a benchmark on complex, multi-task problems like object detection are missing. In this paper, we first provide theoretical insights connecting Monte Carlo Stochastic Depth (MCSD) to principled approximate variational inference. We then present the first comprehensive empirical benchmark of MCSD against MCD and MCDB on state-of-the-art detectors (YOLO, RT-DETR) using the COCO and COCO-O datasets. Our results position MCSD as a robust and computationally efficient method that achieves highly competitive predictive accuracy (mAP), notably yielding slight improvements in calibration (ECE) and uncertainty ranking (AUARC) compared to MCD. We thus establish MCSD as a theoretically-grounded and empirically-validated tool for efficient Bayesian approximation in modern deep learning. |
| 2026-04-14 | [Risk-Calibrated Learning: Minimizing Fatal Errors in Medical AI](http://arxiv.org/abs/2604.12693v1) | Abolfazl Mohammadi-Seif, Ricardo Baeza-Yates | Deep learning models often achieve expert-level accuracy in medical image classification but suffer from a critical flaw: semantic incoherence. These high-confidence mistakes that are semantically incoherent (e.g., classifying a malignant tumor as benign) fundamentally differ from acceptable errors which stem from visual ambiguity. Unlike safe, fine-grained disagreements, these fatal failures erode clinical trust. To address this, we propose Risk-Calibrated Learning, a technique that explicitly distinguishes between visual ambiguity (fine-grained errors) and catastrophic structural errors. By embedding a confusion-aware clinical severity matrix M into the optimization landscape, our method suppresses critical errors (false negatives) without requiring complex architectural changes. We validate our approach in four different imaging modalities: Brain Tumor MRI, ISIC 2018 (Dermoscopy), BreaKHis (Breast Histopathology), and SICAPv2 (Prostate Histopathology). Extensive experiments demonstrate that our Risk-Calibrated Loss consistently reduces the Critical Error Rate (CER) for all four datasets, achieving relative safety improvements ranging from 20.0% (on breast histopathology) to 92.4% (on prostate histopathology) compared to state-of-the-art baselines such as Focal Loss. These results confirm that our method offers a superior safety-accuracy trade-off across both CNN and Transformer architectures. |
| 2026-04-14 | [Every Picture Tells a Dangerous Story: Memory-Augmented Multi-Agent Jailbreak Attacks on VLMs](http://arxiv.org/abs/2604.12616v1) | Jianhao Chen, Haoyang Chen et al. | The rapid evolution of Vision-Language Models (VLMs) has catalyzed unprecedented capabilities in artificial intelligence; however, this continuous modal expansion has inadvertently exposed a vastly broadened and unconstrained adversarial attack surface. Current multimodal jailbreak strategies primarily focus on surface-level pixel perturbations and typographic attacks or harmful images; however, they fail to engage with the complex semantic structures intrinsic to visual data. This leaves the vast semantic attack surface of original, natural images largely unscrutinized. Driven by the need to expose these deep-seated semantic vulnerabilities, we introduce \textbf{MemJack}, a \textbf{MEM}ory-augmented multi-agent \textbf{JA}ilbreak atta\textbf{CK} framework that explicitly leverages visual semantics to orchestrate automated jailbreak attacks. MemJack employs coordinated multi-agent cooperation to dynamically map visual entities to malicious intents, generate adversarial prompts via multi-angle visual-semantic camouflage, and utilize an Iterative Nullspace Projection (INLP) geometric filter to bypass premature latent space refusals. By accumulating and transferring successful strategies through a persistent Multimodal Experience Memory, MemJack maintains highly coherent extended multi-turn jailbreak attack interactions across different images, thereby improving the attack success rate (ASR) on new images. Extensive empirical evaluations across full, unmodified COCO val2017 images demonstrate that MemJack achieves a 71.48\% ASR against Qwen3-VL-Plus, scaling to 90\% under extended budgets. Furthermore, to catalyze future defensive alignment research, we will release \textbf{MemJack-Bench}, a comprehensive dataset comprising over 113,000 interactive multimodal jailbreak attack trajectories, establishing a vital foundation for developing inherently robust VLMs. |
| 2026-04-14 | [IDEA: An Interpretable and Editable Decision-Making Framework for LLMs via Verbal-to-Numeric Calibration](http://arxiv.org/abs/2604.12573v1) | Yanji He, Yuxin Jiang et al. | Large Language Models are increasingly deployed for decision-making, yet their adoption in high-stakes domains remains limited by miscalibrated probabilities, unfaithful explanations, and inability to incorporate expert knowledge precisely. We propose IDEA, a framework that extracts LLM decision knowledge into an interpretable parametric model over semantically meaningful factors. Through joint learning of verbal-to-numerical mappings and decision parameters via EM, correlated sampling that preserves factor dependencies, and direct parameter editing with mathematical guarantees, IDEA produces calibrated probabilities while enabling quantitative human-AI collaboration. Experiments across five datasets show IDEA with Qwen-3-32B (78.6%) outperforms DeepSeek R1 (68.1%) and GPT-5.2 (77.9%), achieving perfect factor exclusion and exact calibration -- precision unattainable through prompting alone. The implementation is publicly available at https://github.com/leonbig/IDEA. |
| 2026-04-14 | [Goal-oriented safe active learning for predictive control using Bayesian recurrent neural networks](http://arxiv.org/abs/2604.12542v1) | Laura Boca de Giuli, Alessio La Bella et al. | A key challenge in learning-based model predictive control (MPC) is to collect informative data online for model adaptation while ensuring safety and without penalising control performance. In this paper, we propose an online model adaptation scheme embedded within an MPC framework in which the last-layer parameters of a recurrent neural network are recursively updated via Bayesian learning. This is achieved by means of a goal-oriented safe active learning algorithm that alternates between an exploration phase, where the MPC actively explores system dynamics to collect informative data for model adaptation while still pursuing the main control objective, and a goal-reaching phase, where it focuses exclusively on the main control objective. The algorithm is complemented with theoretical guarantees of (i) recursive feasibility, (ii) safety, (iii) termination of exploration in finite time, and (iv) close-to-optimal performance. Simulation results on a benchmark energy system demonstrate that the proposed framework achieves economic performance comparable to that of an MPC with full system knowledge, while progressively improving model accuracy and respecting operational safety constraints with high probability. |
| 2026-04-14 | [A Heterogeneous Dual-Network Framework for Emergency Delivery UAVs: Communication Assurance and Path Planning Coordination](http://arxiv.org/abs/2604.12501v1) | Ping Huang, Bin Duo et al. | Natural disasters often damage ground infrastructure, making unmanned aerial vehicles (UAVs) essential for emergency supply delivery. Yet safe operation in complex post-disaster environments requires reliable command-and-control (C2) links; link instability can cause loss of control, delay rescue, and trigger severe secondary harm. To provide continuous three-dimensional (3D) C2 coverage during dynamic missions, we propose a Heterogeneous Dual-Network Framework (HDNF) for safe and reliable emergency delivery. HDNF tightly couples an Emergency Communication Support Network (ECSN), formed by hovering UAV base stations, with a Delivery Path Network (DPN), formed by fast-moving delivery UAVs. The ECSN dynamically safeguards mission-critical flight corridors, while the DPN aligns trajectories with reliable coverage regions. We formulate a joint optimization problem over task assignment, 3D UAV-BS deployment, and DPN path planning to maximize end-to-end C2 reliability while minimizing UAV flight energy consumption and base-station deployment cost. To solve this computationally intractable NP-hard problem, we develop a layered strategy with three components: (i) a multi-layer C2 service model that overcomes 2D-metric limitations and aligns UAV-BS deployment with mission-critical 3D phases; (ii) a 3D coverage-aware multi-agent reinforcement learning algorithm that addresses the high-dimensional search space and improves both training efficiency and topology resilience; and (iii) a 3D communication-aware A* planner that jointly optimizes C2 quality and flight energy, mitigating trajectory--coverage mismatch and improving routing safety. Extensive simulations show that HDNF markedly improves C2 reliability, eliminates outages in critical phases, and sustains high task success rates while reducing hardware deployment cost. |
| 2026-04-14 | [Safety Training Modulates Harmful Misalignment Under On-Policy RL, But Direction Depends on Environment Design](http://arxiv.org/abs/2604.12500v1) | Leon Eshuijs, Shihan Wang et al. | Specification gaming under Reinforcement Learning (RL) is known to cause LLMs to develop sycophantic, manipulative, or deceptive behavior, yet the conditions under which this occurs remain unclear. We train 11 instruction-tuned LLMs (0.5B--14B) with on-policy RL across 3 environments and find that model size acts as a safety buffer in some environments but enables greater harmful exploitation in others. Controlled ablations trace this reversal to environment-specific features such as role framing and implicit gameability cues. We further show that most safety benchmarks do not predict RL-induced misalignment, except in the case of Sycophancy scores when the exploit relies on inferring the user's preference. Finally, we find that on-policy RL preserves a safety buffer inherent in the model's own generation distribution, one that is bypassed during off-policy settings. |
| 2026-04-14 | [HazardArena: Evaluating Semantic Safety in Vision-Language-Action Models](http://arxiv.org/abs/2604.12447v1) | Zixing Chen, Yifeng Gao et al. | Vision-Language-Action (VLA) models inherit rich world knowledge from vision-language backbones and acquire executable skills via action demonstrations. However, existing evaluations largely focus on action execution success, leaving action policies loosely coupled with visual-linguistic semantics. This decoupling exposes a systematic vulnerability whereby correct action execution may induce unsafe outcomes under semantic risk. To expose this vulnerability, we introduce HazardArena, a benchmark designed to evaluate semantic safety in VLAs under controlled yet risk-bearing contexts. HazardArena is constructed from safe/unsafe twin scenarios that share matched objects, layouts, and action requirements, differing only in the semantic context that determines whether an action is unsafe. We find that VLA models trained exclusively on safe scenarios often fail to behave safely when evaluated in their corresponding unsafe counterparts. HazardArena includes over 2,000 assets and 40 risk-sensitive tasks spanning 7 real-world risk categories grounded in established robotic safety standards. To mitigate this vulnerability, we propose a training-free Safety Option Layer that constrains action execution using semantic attributes or a vision-language judge, substantially reducing unsafe behaviors with minimal impact on task performance. We hope that HazardArena highlights the need to rethink how semantic safety is evaluated and enforced in VLAs as they scale toward real-world deployment. |
| 2026-04-14 | [RACF: A Resilient Autonomous Car Framework with Object Distance Correction](http://arxiv.org/abs/2604.12418v1) | Chieh Tsai, Hossein Rastgoftar et al. | Autonomous vehicles are increasingly deployed in safety-critical applications, where sensing failures or cyberphysical attacks can lead to unsafe operations resulting in human loss and/or severe physical damages. Reliable real-time perception is therefore critically important for their safe operations and acceptability. For example, vision-based distance estimation is vulnerable to environmental degradation and adversarial perturbations, and existing defenses are often reactive and too slow to promptly mitigate their impacts on safe operations. We present a Resilient Autonomous Car Framework (RACF) that incorporates an Object Distance Correction Algorithm (ODCA) to improve perception-layer robustness through redundancy and diversity across a depth camera, LiDAR, and physics-based kinematics. Within this framework, when obstacle distance estimation produced by depth camera is inconsistent, a cross-sensor gate activates the correction algorithm to fix the detected inconsistency. We have experiment with the proposed resilient car framework and evaluate its performance on a testbed implemented using the Quanser QCar 2 platform. The presented framework achieved up to 35% RMSE reduction under strong corruption and improves stop compliance and braking latency, while operating in real time. These results demonstrate a practical and lightweight approach to resilient perception for safety-critical autonomous driving |
| 2026-04-14 | [Preventing Safety Drift in Large Language Models via Coupled Weight and Activation Constraints](http://arxiv.org/abs/2604.12384v1) | Songping Peng, Zhiheng Zhang et al. | Safety alignment in Large Language Models (LLMs) remains highly fragile during fine-tuning, where even benign adaptation can degrade pre-trained refusal behaviors and enable harmful responses. Existing defenses typically constrain either weights or activations in isolation, without considering their coupled effects on safety. In this paper, we first theoretically demonstrate that constraining either weights or activations alone is insufficient for safety preservation. To robustly preserve safety alignment, we propose Coupled Weight and Activation Constraints (CWAC), a novel approach that simultaneously enforces a precomputed safety subspace on weight updates and applies targeted regularization to safety-critical features identified by sparse autoencoders. Extensive experiments across four widely used LLMs and diverse downstream tasks show that CWAC consistently achieves the lowest harmful scores with minimal impact on fine-tuning accuracy, substantially outperforming strong baselines even under high harmful data ratios. |

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



