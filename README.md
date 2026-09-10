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
| 2026-09-09 | [Forgetting Only What Matters: Layer-Selective Unlearning toward Robust LLMs](http://arxiv.org/abs/2609.10439v1) | Ravi Ranjan, Olivera Kotevska et al. | Large Language Models (LLMs) can memorize and reproduce sensitive, copyrighted, or otherwise undesirable training content, creating privacy, safety, and regulatory concerns. Machine unlearning offers a practical alternative to full retraining, but many existing methods apply broad or fixed parameter updates that can degrade utility and remain brittle under deployment changes such as post-training quantization, where forgotten knowledge may partially re-emerge. We propose Forgetting Only What Matters via Unlearning Layers (FOM-UL), a layer-level unlearning framework that selects transformer layers using a forget-to-retain significance score. This score identifies layers with high influence on the forget set and low sensitivity to the retain set, allowing FOM-UL to concentrate updates where they are most effective while leaving most of the model unchanged. This targeted update strategy improves the forgetting-utility trade-off and provides an empirical path toward quantization-resilient unlearning by reducing the chance that small, diffuse updates are erased by low-bit rounding. Across TOFU, KnowUnDo, and MUSE-style evaluations, FOM-UL reduces residual memorization compared with strong GA, NPO, KLD, SURE, ReLearn, and LUNAR-based baselines while preserving retain-set utility close to the vanilla model. Under 8-bit and 4-bit post-training quantization, FOM-UL maintains stronger memorization suppression and utility preservation than competing methods, and adversarial prompt evaluations show lower recovery of forgotten content. Overall, FOM-UL provides an efficient unlearning strategy that improves targeted forgetting, utility preservation, and deployment robustness without claiming formal guarantees of erasure. |
| 2026-09-09 | [Data-Driven Risk Fields for Safer End-to-End Autonomous Driving](http://arxiv.org/abs/2609.10377v1) | Yuanxin Tian, Zhiyuan Liu et al. | Safety is a fundamental requirement for autonomous driving, yet existing end-to-end driving models still lack explicit risk-aware learning capacities. Existing rule-based risk models provide interpretable safety priors, yet their absolute risk scores depend on handcrafted functions, coefficients, and thresholds. Learning-based risk representations reduce part of this manual design, but their supervision often relies on occupancy-derived labels or heuristic cost values, which may not capture ego-conditioned planning risk. In this paper, we propose DRiF, a data-driven risk-field framework for safer end-to-end autonomous driving. DRiF learns a shared BEV feature with static map segmentation, dynamic risk prediction, and vehicle planning. For dynamic risk learning, DRiF converts rule-based safety priors into pairwise risk labels, and trains the risk field to preserve relative risk ordering instead of regressing handcrafted absolute scores. Experiments on Bench2Drive show that DRiF achieves competitive overall performance, with consistent improvements in driving score, success rate, and collision-related metrics. These results establish relative risk supervision as an effective way to connect explicit safety structure with end-to-end planning. The data and code will be publicly available. |
| 2026-09-09 | [Future-Aware Flow Planning for Safe UAV Target Following](http://arxiv.org/abs/2609.10166v1) | Boning Feng, Haoran Zhang et al. | UAV target following in cluttered environments is inherently predictive: current-state followers can lag behind turns, choose blocked corridors, or trade tracking for unsafe near-horizon motion. We propose a future-aware flow planning framework for state-informed UAV target following. Predicted target futures guide clean UAV trajectory generation as horizon-aligned residual signals, while risk-scored executable-prefix repair is embedded inside the sampling loop. On fixed ID/OOD receding-horizon benchmarks, the planner improves the intended safety--tracking trade-off rather than dominating every metric: it matches zero measured ID collision rate with the highest ID safe-tracking time, and gives the lowest OOD macro collision rate and final tracking error among the displayed methods, while Future-MPC remains smoother and stronger on some thresholded OOD success metrics under its hand-designed objective. Ablations show that future adaptation improves candidate generation before safety repair, and simulator-facing stress tests probe interface, sensing, and controller-execution effects. These results support horizon-aligned future adaptation and embedded prefix repair as complementary ingredients for safe UAV target following under the tested simulation conditions. |
| 2026-09-09 | [Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning](http://arxiv.org/abs/2609.10142v1) | Jing Guan, Yachao Yang et al. | Large language models remain fragile against malicious fine-tuning, motivating training-time defenses against harmful persona drift. Preventative Steering injects undesirable-trait persona vectors during fine-tuning and removes them at evaluation time, yet the mechanism behind its lasting protection remains unclear. Analyzing its temporal optimization dynamics, we find that the defense emerges from an early compensatory adaptation phase followed by a steady-state phase where the corrective signal decays; in parameter space, attention output projections emerge as the dominant residual-write route for defensive updates. Through Intervention Delta Preservation (IDP) and IDP Continuation experiments, we further show that preserving or reinjecting the weight offset fails to maintain protection, indicating that preventative steering relies on active adaptation rather than a static defense. Motivated by this finding, we propose Progressive Intensity Scheduling (PIS), which starts with a moderate injection strength and increases it after static-strength alignment begins to decay. Across the evaluated Qwen2.5 and Gemma-3 models, PIS improves safety robustness over static-strength steering while reducing harmful trait expression. |
| 2026-09-09 | [Guaranteeing Faithful Evidence Extraction in Speculative Retrieval-Augmented Generation](http://arxiv.org/abs/2609.10046v1) | Quentin Signé, Mohand Boughanem et al. | Large Language Models (LLMs) are increasingly used as interfaces for information retrieval, but they remain prone to hallucinations and faithfulness errors, in which the generated answers diverge from the retrieved evidence. While Retrieval-Augmented Generation (RAG) and recent hybrid or semi-extractive approaches mitigate this issue, they do not guarantee that quoted or extracted spans are verbatim from the retrieved context. This limitation can have severe consequences in safety-critical domains, where answers must exactly match certified documentation.   We introduce Constrained Hybrid Decoding (CHyD), a novel faithfulness-first paradigm for speculative RAG. While traditional speculative decoding is optimized for inference speed, CHyD repurposes this architecture to ensure faithful verbatim evidence extraction when the extraction mode is correctly triggered. Our approach enforces hard decoding constraints that restrict generation to continuous spans present in the retrieved documents. This design provides a robust but straightforward guarantee: any explicitly quoted span in the output appears verbatim in the provided context.   We evaluate our method across state-of-the-art LLMs on diverse abstractive, extractive, and semi-extractive QA benchmarks, including technical datasets motivated by aircraft maintenance. Results show that existing hybrid methods frequently hallucinate quoted spans, with exact extraction accuracy dropping below 40% in technical domains. In contrast, our approach achieves near-perfect extraction faithfulness regardless of the model used. Although enforcing hard constraints introduces a trade-off with fluency-oriented metrics, our method improves exact answer correctness and remains competitive overall, highlighting its suitability for safety-critical information retrieval applications. |
| 2026-09-09 | [From Few-Shot Segmentation to Clinician-in-the-Loop Medical Image Analysis](http://arxiv.org/abs/2609.10001v1) | Yazhou Zhu | Few-shot medical image segmentation (FSMIS) seeks to delineate unseen structures from a small support set, but its standard formulation fixes task-defining evidence before inference. This assumption is fragile when query cases exhibit acquisition shift, atypical pathology, ambiguous boundaries, or poor image quality. Prototype learning, cross-domain matching, interactive segmentation, uncertainty estimation, test-time adaptation, and promptable foundation models address parts of this problem, yet have not been jointly evaluated under a common model of expert attention and clinical risk. This Perspective reframes FSMIS as a sequential clinician-model decision problem with a static support budget $K$ and a distinct interaction budget $B$. At each step, a system accepts the current segmentation, requests feedback, or defers to full expert review. Queries vary in location and modality and are selected by response-conditioned net expected value of information; clinician-provided feedback informs bounded adaptation only after prespecified provenance, consistency, and safety gates. The framework separates distributional atypicality from predicted clinical failure and treats clinician responses as informative but fallible observations. We synthesize the transition from few-shot and cross-domain segmentation to interactive and selective adaptation, delineate the integration gap, and define four research directions with falsifiable hypotheses. Evaluation spans external-domain calibration, quality-effort trade-offs, reader studies, and prospective workflow assessment. The central claim is not that interaction alone resolves domain shift, but that scarce expert attention should be allocated only when it is expected to reduce clinically relevant risk. |
| 2026-09-09 | [HaWMPO: Hallucination-Aware World Model-based Policy Optimization for Generalist Robot Policy](http://arxiv.org/abs/2609.09941v1) | Zengjue Chen, Peidong Liu et al. | Generalist robot policies have demonstrated strong generalization across robotic manipulation tasks, yet their success rates remain limited in com- plex long-horizon scenarios. Recent methods improve Visual-Language-Action (VLA) policies through online reinforcement learning on real robots, but such training relies on costly physical interactions, suffers from low sample efficiency, and may introduce hardware and safety risks. World models offer a promising alternative by enabling policy optimization with imagined rollouts. However, long-horizon rollouts generated by world models often suffer from prediction hal- lucinations, producing biased state transitions that can mislead policy learning. To address this issue, we propose Hallucination-aware World Model-based Pol- icy Optimization (HaWMPO), a closed-loop reinforcement learning pipeline for VLA policy post-training with world models. Specifically, HaWMPO introduces an action-conditioned hallucination-aware model to estimate the reliability of gen- erated image sequences, and incorporates hallucination scores into group relative policy optimization through a Reward-Soft mechanism, suppressing unreliable ac- tion chunks during training. On the LIBERO benchmark, HaWMPO achieves the best average success rate, with gains of 15.0% over the base model and 2.8% over the strongest baseline; real-world experiments on a G1 robot further validate its effectiveness, raising the average success rate on two manipulation tasks from 67.5% to 80.0%. |
| 2026-09-09 | [TempTPI: Informer-Based trajectory prediction for maritime vessels](http://arxiv.org/abs/2609.09840v1) | Kevin Ferneding, Veronika Lietavcova et al. | Accurate long-term trajectory prediction for maritime vessels is essential for safety and logistical efficiency. While deep learning models, particularly Transformers, have shown promise in processing Automatic Identification System (AIS) data, they often struggle with the quadratic computational complexity of self-attention and the loss of accuracy over extended forecasting horizons. This study proposes TempTPI, a novel prediction framework that integrates an Informer-based encoder with a multi-channel temporal encoding mechanism. The Informer architecture leverages a ProbSparse self-attention mechanism to reduce computational overhead and focus on the most significant dependencies, while the temporal encoder utilizes Fourier-like frequency expansions to capture cyclic patterns (hourly, daily, and seasonal) in vessel behavior. We evaluate our model against the state-of-the-art TPTrans architecture using AIS data from Danish waters. Experimental results demonstrate that TempTPI consistently outperforms existing methods across prediction windows of 1 to 5 hours. Notably, at a 5-hour horizon, the proposed model achieves a 55% improvement in Mean Squared Error (MSE), offering a robust solution for long-range maritime situational awareness. |
| 2026-09-09 | [CS-Guard: Benchmarking LLM Guardrails for Code Generation Security](http://arxiv.org/abs/2609.09798v1) | Jinyang Li, Mingyu Guo et al. | Large language models (LLMs) have been ex- ploited to generate malware, but the effective- ness of guardrails for code generation secu- rity remains unclear. We introduce CS-Guard, the first benchmark to systematically evalu- ate guardrails for code generation security. It covers 1) text-to-code generation with 1000 high-quality malware-generation prompts, 7 jailbreak attacks, and a novel fictional scenario attack (FSA) that embeds malicious intent in a legitimate fictional software-development sce- nario; and 2) code-to-code generation with 331 code prompts spanning code infilling, code completion, and code translation. We empiri- cally evaluate 9 guardrails across seven LLMs. We find that current guardrails perform poorly against malicious code-generation re- quests: for text-to-code, the average attack success rate (ASR) after jailbreaks reaches about 50% for many guardrails; for code-to- code, average ASR approaches 100% on base LLMs and remains high across many guardrails (14.4% to nearly 100%). Our FSA also achieves ASR close to 100% across many guardrails, raising major reliability concerns for real-world software development. To sup- port future research, CS-Guard uses a modular three-layer guardrail taxonomy that lets devel- opers register guardrails for evaluation. We release the benchmark and data to enable fur- ther community evaluation. |
| 2026-09-09 | [How Fragile Is Safety Alignment at Frontier Scale? A Single-Direction Attack on a 320B MoE](http://arxiv.org/abs/2609.09793v1) | Yi Shi, Tanyu Chen et al. | Directional ablation removes an aligned language model's ability to refuse by projecting a single "refusal direction" out of the weights that write the residual stream. It needs no gradient-based training and no optimization, only a few hundred contrastive prompts, which makes it the canonical white-box attack on open-weight alignment. However, it has been established only on dense models up to roughly 70B parameters. We study whether it survives the shift to frontier mixture-of-experts (MoE) models whose residual streams are no longer a single tensor and whose weights ship quantized. We apply it to GLM-5.3-Flash (320B parameters, 288 routed experts, a four-wide hyper-connection residual, block-FP8). The attack survives the architecture, but what it reaches is no longer where a reader of the original recipe would look for it. Editing the attention, dense and routed-expert writers on their own removes 0.039, 0.016 and 0.148 of refusal respectively; editing all three together removes 0.776. As a result, 74% of the effect exists only under the joint intervention. The part the conventional recipe reaches by module-name matching accounts for 0.066 of that 0.776, which is why it fails silently on an MoE. The effect does not follow from removing just any direction: ablating a random direction orthogonal to it leaves refusal unchanged. A category-concentrated residue survives every edit we tried: subspaces fitted on violence, sexual content and hate leave measurable refusal at every rank from 1 to 12. We report the method, the 41-89 percentage-point reductions it achieves across seven harmful benchmarks with no detected change in capability, and the boundary where it stops. |
| 2026-09-09 | [Procedural Memory Under Change: Reuse and Interference in Controlled Web Tasks](http://arxiv.org/abs/2609.09774v1) | Yanze Cao | Procedural memory lets language agents reuse successful routines, but reuse presumes that a stored routine remains applicable. We study what happens when that presumption is deliberately violated. The study combines a retrospective, human-assisted interface-adaptation case from BrowserGym TimeWarp with controlled frozen-memory comparisons on synthetic shopping decisions. During the documented WebShop V1-V6 development path, interface-specific code was adapted while the separately stored high-level procedure was not reported to change; this phase does not constitute an autonomous memory-agent evaluation. In the controlled phase, an early pilot produced one task on which two memory conditions selected a more expensive item while the no-memory condition selected the reference minimum. Follow-up probes did not establish a recurring row-order or identity-binding pattern. We then tested four forms of mismatch: changed quantities, a different evidence representation, a conflict between local and global optimization, and distributed promotion evidence, across 32 formal cells. Each cell used one temperature-0 generation with the same local qwen3:8b configuration and no adaptive retry. Across these pairs, none of the predefined diagnostic interference signatures appeared on the tasks for which they were defined when current-task evidence was explicit and sufficient. The result identifies a tested region of non-interference: a procedural memory can be mismatched without becoming behaviorally disruptive. It does not establish general safety or a mechanism. The remaining question is which additional conditions turn applicability mismatch into observable, memory-caused error. |
| 2026-09-09 | [Can Artificial Intelligence Support Healthcare and Mental Health Through Early Cyberbullying Detection ? The Impact of Emotion-Aware AI on Proactive Online Safety](http://arxiv.org/abs/2609.09735v1) | Hamed Jelodar, Amir Firouzi et al. | Healthcare systems, mental health, and public well-being are increasingly affected by cyberbullying and harmful online interactions. This paper presents CareGuard, an early-warning framework designed to support healthcare-driven mental health protection and proactive online safety through the detection of cyberbullying-related content using advanced natural language processing techniques. CareGuard integrates zero-shot semantic labeling with fine-tuned transformer-based models, including BERT, DistilBERT, and RoBERTa, to enable robust and context-aware classification across sensitive cyberbullying categories. To improve efficiency and reduce unnecessary computation in healthcare-oriented monitoring settings, the framework incorporates an emotion-aware filtering mechanism alongside cosine similarity-based semantic screening, allowing the system to focus on semantically relevant and emotionally salient content. Experimental results on benchmark datasets demonstrate that CareGuard effectively balances detection accuracy and computational efficiency, highlighting its potential for scalable deployment in healthcare systems, mental health monitoring, and online safety applications. |
| 2026-09-09 | [CT-SAFR: Safe and Interpretable Chain-of-Thought Reasoning for Autonomous Robots: A Multi-Layered Verification Framework for Trustworthy AI-Driven Robotic Decision Making](http://arxiv.org/abs/2609.09692v1) | Cagri Temel | Chain-of-Thought (CoT) prompting enables LLMs to perform explicit, step-by-step reasoning, creating opportunities for sophisticated autonomous robots. However, recent research reveals that reasoning models verbalize their actual decision processes only 25-39% of the time, with faithfulness degrading 44% on complex tasks. This paper presents CT-SAFR (Chain-of-Thought Safety and Faithfulness for Robotics), a multi-layered verification framework achieving 94.2% hallucination detection (n = 500, 95% CI: 91.8-95.9%) with sub-500ms latency. Through a warehouse robot case study, this work demonstrates 87% reduction in unsafe reasoning outputs (p < 0.001) and provides recommendations for responsible deployment of reasoning-capable autonomous robots. |
| 2026-09-09 | [Safe to Stop? Risk-Constrained Stopping for Sequential Clinical Diagnosis Agents](http://arxiv.org/abs/2609.09678v1) | Yuexin Wu, Vasile Rus | Clinical diagnosis agents must decide not only what test to request next, but also when to diagnose or defer. Existing agent benchmarks largely evaluate accuracy after fixed or unconstrained interaction, leaving autonomous stopping reliability implicit. We present Cros, a risk-constrained stopping layer combining state-wise error ranking, policy design on disjoint development splits, and LTT-style exact tests of selective diagnostic error and minimum autonomous coverage for complete sequential policies. Its finite-sample guarantee requires the candidate family, testing rule, and any randomization to be frozen before calibration labels are accessed. On a 1,834-episode MIMIC-derived abdominal-pain benchmark, the full ranker achieves exploratory state-error AUROC 0.853, compared with 0.715 for maximum class probability and 0.552 for the backbone's native stop score. On the previously viewed 367-episode evaluation split, analytically averaging over the frozen Cros weights yields 16.9% selective error at 78.8% coverage, cost 5.57, and 0.68 tests, versus 30.8% error at 100% coverage, cost 8.14, and 1.53 tests under native stopping. Forced continuation is non-monotone: error is 28.3% with HPI alone and 34.3% after full workup. However, the uniform-weight mixture ablation is cheaper on this viewed split despite missing the locked development margins, and Cros nominally satisfies the joint criterion in only 6 of 20 development resplits. Because evaluation labels were inspected during earlier development, these findings provide exploratory feasibility and audit evidence, not a confirmatory safety certificate. |
| 2026-09-09 | [Reducing Prescription Errors Through Information Intervention: A Field Experiment in Healthcare Operations](http://arxiv.org/abs/2609.09673v1) | Xiaodan Shao, Vivek Choudhary et al. | Drug-drug interaction (DDI) errors pose serious risks to patient safety. Existing decision-support systems often require physicians to respond to alerts, disrupting workflows and contributing to high override rates. We examine whether a non-mandatory information intervention can reduce DDI errors and foster learning. Using a randomized field experiment with India's largest electronic medical record platform, we analyze 2.81 million prescriptions from 1,700 physicians using a difference-in-differences design. Treatment physicians received real-time information highlighting DDI errors without being required to respond, while control physicians received no such information. The intervention reduced DDI errors by 8.6%, corresponding to an estimated US$4.8 million in annual hospitalization cost savings and approximately 134 lives potentially saved. We identify two mechanisms: reactive correction, whereby physicians remove errors after they are flagged, and proactive learning, whereby they avoid errors before alerts occur. While early reductions are driven primarily by correction, physicians increasingly avoid errors over time. They also become less likely to repeat previously flagged errors and reduce new errors, suggesting that learning generalizes beyond specific drug pairs. The effects are consistent across physician types and do not compromise productivity or care quality. Our findings show that non-mandatory information interventions can improve patient safety through both immediate error correction and persistent, generalizable learning. |
| 2026-09-09 | [A Risk-Sensitive and Uncertainty-Aware Decision-Making and Control Framework for Safe and Robust Autonomous Driving](http://arxiv.org/abs/2609.09650v1) | Zhuoren Li, Ran Yu et al. | Reinforcement learning (RL) has demonstrated considerable potential for autonomous driving decision-making. However, its deployment in urban autonomous driving, particularly at highly interactive unsignalized intersections, remains challenging, as learned policies may struggle to maintain both safety and robust decision-making in complex traffic situations. Conventional safety-filtering approaches typically employ fixed conservative constraints, which may improve safety at the cost of excessive intervention and degraded traffic efficiency. To address these limitations, we propose a Risk-sensitive and Uncertainty-aware Decision-making and Control (RUDC) framework for safe and robust autonomous driving. RUDC couples risk-sensitive distributional RL with ensemble-based policy uncertainty quantification, jointly accounting for tail risks in return distributions and uncertainty in learned policies. An uncertainty-aware high-order control barrier function (HOCBF)-based safety correction mechanism adaptively adjusts constraint strictness according to policy uncertainty, while a learnable residual predictor compensates for CBF model mismatches and discretization errors. Extensive simulations at unsignalized intersections demonstrate that RUDC achieves a favorable balance among safety, efficiency, and robustness, outperforming representative safe RL baselines under both nominal and challenging OOD and long-tail scenarios while satisfying real-time requirements. |
| 2026-09-09 | [UnsafeChecker: Finding Soundness Bugs in Rust Safe Abstractions](http://arxiv.org/abs/2609.09641v1) | Xizhe Yin, Yaokun Zhang et al. | Rust guarantees memory safety without garbage collection through a strict ownership and borrowing system. However, for low-level systems programming, many widely used libraries rely on the unsafe keyword. These libraries encapsulate raw-pointer operations behind safe APIs to form safe abstractions. A single mistake in this internal unsafe code can break its safety contract, rendering the abstraction unsound and allowing safe clients to trigger undefined behavior. Detecting these potential soundness violations is challenging. Existing static analysis tools for C/C++ ignore Rust-specific safety contracts, while current Rust tools lack the deep semantic modeling required to track the contexts that raw pointers erase.   To address this gap, we present UnsafeChecker, a compiler-integrated static analysis framework for detecting potential soundness violations in Rust safe abstractions. UnsafeChecker analyzes Rust MIR using a flow-sensitive abstract interpretation that maintains a shared state with three components: ownership, object validity, and layout. Each warning rule consumes the subset of facts needed for the corresponding Rust safety obligation. UnsafeChecker reports both instruction-level undefined behavior and boundary-level contract violations that may escape through safe APIs. We evaluate UnsafeChecker on a benchmark of 46 RustSec vulnerabilities, which contain 53 ground-truth bugs. UnsafeChecker outperforms several state-of-the-art tools, detecting 32 CVEs and covering 36 bugs (67.9% recall) with 51.6% alert-level precision. Furthermore, in a large-scale scan of real-world crates on crates.io, UnsafeChecker uncovered 114 previously unknown bugs across 83 crates, with 45 confirmed and 27 already fixed by maintainers. |
| 2026-09-09 | [Arbitrary Cipher Attacks Against Large Language Models Do Not Require Fine-Tuning](http://arxiv.org/abs/2609.09553v1) | Thomas Rivasseau | Large language model safety and security research is preoccupied with, among other things, detecting and preventing jailbreak attacks: alignment bypasses that allow an adversarial user to elicit unwanted or harmful outputs from models. Arbitrary cipher, or covert communication, attacks are one such type of jailbreak and have previously been demonstrated against the fine-tuning APIs of commercial models. In these attacks, target models are trained on a corpus of encrypted harmful questions and responses and subsequently respond to harmful requests through the learned encryption scheme. In this paper, we show that newer frontier models do not require fine-tuning to acquire cipher-based communication skills. Instead, they can learn these skills through prompting and, when necessary, through in-context learning. Furthermore, model alignment is significantly weakened or entirely bypassed when communication occurs through the learned cipher. To the best of our knowledge, this constitutes a novel attack vector against commercial black-box large language models. We demonstrate successful jailbreaks against frontier models developed by Anthropic, Google, and OpenAI. Our attack bypasses commercial harmfulness classifiers because harmful content is encrypted and therefore appears as nonsensical text or gibberish. |
| 2026-09-08 | [Scalable Oversight for AI in Mental Health: Lessons from 350,000 AI Coaching Conversations between Therapy Sessions](http://arxiv.org/abs/2609.09533v1) | Matthew A. Scult, John L. Havlik et al. | Clinician review of every AI output is often proposed as a safeguard in mental healthcare, but vigilance research suggests this approach fails at scale and may paradoxically reduce safety. Drawing on our experience deploying an AI coaching tool across 350,000+ conversations between therapy sessions, we describe how we arrived at a three-layer human-on-the-loop oversight framework combining preventive design, real-time monitoring, and continuous clinician evaluation. We show how specific findings from clinical review drove iterative improvements, and offer practical recommendations for mental health professionals evaluating AI systems. |
| 2026-09-08 | [OmniEye: Efficient Multimodal Forensic Video Intelligence for Law-Enforcement Body-Worn Cameras](http://arxiv.org/abs/2609.09460v1) | Mamadou K. Keita, Angela Srbinovska et al. | We introduce OmniEye, a multimodal video intelligence system for law-enforcement training and review (source code available on request to verified law-enforcement and public-safety agencies). OmniEye ingests body-worn camera footage and perceives every 30-second window jointly across video and audio with one multimodal foundation model. It then stores the model's structured output in an embedded SQLite database with BM25 full-text search. Officers can question the footage through an agent that writes structured queries, retrieves candidate windows, and re-perceives them with the model before it may cite them. The whole system runs on one 16 GB GPU with a 4-bit quantization-aware-trained model, and it also scales to full bf16 precision on a multi-GPU cluster. |

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



