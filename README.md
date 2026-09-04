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
| 2026-09-03 | [Toward Frontier-Quality Declarative UI Generation at Small-Model Cost](http://arxiv.org/abs/2609.04184v1) | Yingxiang Yang, Weihang Xiao et al. | Declarative UI protocols such as A2UI let applications generate interactive UIs by selecting pre-built components from a catalog and binding their props to application data, rather than emitting frontend code from scratch. This contract is attractive for production systems because of safety and consistency. An open question is: can low-latency and low-cost small models achieve the required quality for A2UI-based UI generation? To answer this, we systematically study three controllable design choices for catalog-conditioned A2UI generation: supervised fine-tuning (SFT) data construction method, model size, and component-catalog size. Across two React/TypeScript domains and four base checkpoints spanning two model families (Qwen 3.5 0.8B/2B/4B; SmolLM 3B), we find: (i) a 4B fine-tuned student recovers ~98% of teacher semantic quality and ~97% of teacher visual quality at more than an order of magnitude lower cost than frontier API calls; (ii) both augmented strategies (Perturbed-catalog and Constrained-GT) Pareto-dominate the unaugmented Full-catalog baseline, while specializing on different axes; (iii) even small models can handle and benefit from relatively large component catalog size. We distill these results into practitioner-facing trade-offs and deployment recommendations across the three design choices. |
| 2026-09-03 | [Representational alignment yields generalizable safety in language models](http://arxiv.org/abs/2609.04022v1) | Lingyu Li, Yan Teng et al. | Aligning large language models (LLMs) is essential for their safe deployment. Current alignment methods mainly optimize observable responses, yet models remain vulnerable when the same harmful intent is recast in unfamiliar or adversarial forms that humans can easily recognize. Prototype theory offers an account of this adaptability. Human concepts are represented around central cases, and new instances are categorized according to their graded typicality relative to these prototypes. Here we show that such categorization of moral concepts is weakly preserved in current LLMs. Across 23 LLMs, models often failed to distinguish opposed moral categories or preserve fine-grained typicality within each category. These deficits persist across parameter sizes and alignment stages. We developed representational similarity optimization, which directly aligns the latent representations in LLMs with the categorization expressed in human moral judgements, without supervising generated responses. In matched experiments using the same 251,334 moral annotations, standard behavioral alignment learned the intended moral judgements at the response level while leaving the categorization structure largely unchanged and increasing vulnerability across adversarial evaluations. Reorganizing moral categorization produced more modest gains in explicit judgements but consistently improved adversarial robustness across model scales on diverse benchmarks and attack strategies. Our findings provide functional support for the view that prototype-based categorization contributes to behavioral adaptability. They also show that transferring this representational principle to LLMs yields generalizable safety under adversarial conditions. |
| 2026-09-03 | [FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models](http://arxiv.org/abs/2609.04021v1) | Yalun Wu, Junfeng Fang et al. | Evaluating large language models (LLMs) in safety-critical, physics-governed environments requires more than accuracy-based metrics, because predictions that are numerically close to the ground truth can still violate operational constraints, combine fields in physically inconsistent ways, or fail to produce usable structured outputs. Existing evaluation protocols do not measure these failure modes reliably. We propose FLY-EVAL++, an evidence-driven evaluation protocol that combines deterministic verification of protocol compliance, physical feasibility, and safety constraints with fixed rubric-guided aggregation into interpretable multi-dimensional scores. We instantiate FLY-EVAL++ for Flight Trajectory and Attitude Prediction (FTAP) by extending the PilotBench setting with history-conditioned and multi-step prediction tasks. Across 66 LLMs, safety compliance is the most discriminative dimension of model behavior: models with comparable predictive performance differ by more than 28 points in safety score, and we observe recurrent failures including safety violations under physically plausible predictions and instability in multi-step rollouts. These results show that evaluation in safety-critical domains should measure constraint satisfaction and structured validity explicitly rather than rely on accuracy-centric reporting alone. |
| 2026-09-03 | [Barnacle: Adaptive Multi-Leader Scheduling for DAG-Based Consensus](http://arxiv.org/abs/2609.03978v1) | Zeno De Angeli, Alexandru Ianov Vitanov et al. | In DAG-based consensus, all validators propose blocks concurrently, and designated leader blocks drive transaction commit. Having multiple leader slots per round cuts queuing latency, yet production deployments run a single leader because of head-of-line blocking: a slow leader stalls the pipeline for at least one leader timeout, and for several waves when its slot must wait for the fallback indirect decision rule. This risk grows with the leader count. We introduce Barnacle, an add-on that adapts the leader count at run time. Every interval, it measures on the agreed committed DAG the fraction of slots decided as commit by the direct rule, and drives the leader count with additive increase, multiplicative decrease. The measurement requires no extra messages and no cryptography, and is deterministic. Barnacle is generic over DAG protocols; we instantiate it on four protocols spanning the Byzantine (3f + 1, 5f + 1), crash-only (2c + 1), and mixed (5f + 3c + 1) fault models, with proven safety and liveness. Results show Barnacle matches the best static leader count in every regime: in a healthy network its latency is 6-13% lower than a single leader's, and under degradation it matches a single leader while remaining 35-56% below a static high count. We are currently collaborating with the Sui team to integrate Barnacle into the Sui blockchain. |
| 2026-09-03 | [Beyond Majority Vote: Multi-Perspective Adjudication for Medical Hallucination Detection](http://arxiv.org/abs/2609.03953v1) | Joe Cecil, Marjorie Freedman | Understanding the frequency of factual errors in chatbot-generated text and evaluating systems that detect these errors is critical for determining chatbot safety. Yet factual-error detection is often treated as a single-pass, single-annotator labeling problem. In long-form chatbot responses, factual errors can be subtle and embedded within mostly correct text.   We develop a multi-perspective annotation study of medically relevant chatbot responses, combining first-pass annotation, LLM-as-a-Judge (LaJ) candidate discovery, and two forms of adjudication: medical-expert and evidence-based fact-checking. First-pass annotators frequently miss factual errors later validated by adjudicators. LaJ improves candidate discovery, but is insufficient on its own: It misses factual errors that annotators catch. We also find disagreement among adjudicators, suggesting that adjudication over multiple candidate sources can improve benchmark completeness, but does not eliminate the need to apply judgment and expertise. Applied to an existing benchmark, this technique reveals a similar pattern of missing annotations. Together, these results suggest that in the settings examined here, single-pass hallucination benchmarks may achieve scale at the cost of undercounting factual errors. Multi-pass adjudication can improve coverage, but inferences drawn from the benchmarks are still sensitive to the judgment, expertise, and evidence used to determine error presence. |
| 2026-09-03 | [Value-Preserving Architectures for Agentic AI Systems](http://arxiv.org/abs/2609.03920v1) | Alessandro Pesare, Tommaso Dolci et al. | The emergence of agentic AI and LLM-based multi-agent systems (MAS) presents unprecedented opportunities for automating complex tasks, while simultaneously raising critical concerns about the preservation of fundamental human-centered values, such as privacy, fairness, and safety. Although software engineering has traditionally focused on functional correctness, the adoption of LLMs and AI agents into complex socio-technical systems has intensified the need for responsible software engineering and robust value alignment. In MAS, architectural design decisions, such as coordination mechanisms, communication protocols, and system topologies, play a central role in shaping system behavior and the outcomes they produce. This paper argues that architectural choices influence not only the functionality and performance of MAS but can also promote value-oriented system behavior. Therefore, we investigate how different architectural designs support different human-centered values, discussing the following value-preserving architectural patterns: (i) a privacy-aware architecture with a federated topology, (ii) a distributed architecture to promote pluralism and diversity, and (iii) a guard-agent architecture to detect and mitigate unfairness. Finally, we introduce representative use cases to illustrate the proposed architectures in real-world scenarios. By linking architectural design with human-centered values, this work lays the foundation for a unified set of architectural patterns and guidelines towards the design of trustworthy MAS. |
| 2026-09-03 | [Beyond Shallow Alignment: How Post-Training Methods Determine Refusal Circuits And Steering Robustness](http://arxiv.org/abs/2609.03887v1) | Hoang Cuong Nguyen, Mark Dras et al. | How do the methods used to train language models to refuse harmful requests shape how that refusal actually works inside the model? We compare three post-training methods - supervised fine-tuning, reasoning-augmented fine-tuning (training on reasoning chains that justify a safety decision), and preference optimization (ORPO) - across three architecturally distinct models (Llama-3.1-8B, Gemma-2-9B, Qwen3-8B). We find that training method, not just data, reshapes how refusal is computed internally: reasoning-augmented training consistently produces a distinct kind of refusal computation, visible across all three models, while architecture independently shapes internal structure and how reliably refusal can be steered. Most importantly, no method we study achieves all three properties we would want from safe alignment at once: refusal that isn't concentrated in a few fragile components, safety gains that don't cost general capability, and safety behavior correctable through small, targeted edits. We caution against treating current post-training methods as a solved, reliable defense, especially for security-critical use. Code and models are available in https://github.com/hoangcuongnguyen2001/Beyond-Shallow-Alignment. |
| 2026-09-03 | [IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks](http://arxiv.org/abs/2609.03781v1) | Saikat Mondal,  Mamta et al. | Large language models (LLMs) are increasingly used in multilingual settings, yet their safety is still evaluated primarily in English. This limits our understanding of how alignment failures manifest in low-resource and culturally diverse languages. We introduce IndicSafeEval, a persuasion-based jailbreak evaluation framework for Indian languages. Our benchmark combines ten safety critical content categories with six human-like persuasive strategies across four different Indian languages, such as Hindi, Bengali, Marathi and Punjabi, resulting in 7,200 adversarial prompts. We conduct a systematic black-box evaluation of several open-source LLMs to examine how their safety behaviour varies across languages, persuasion strategies, and risk categories. Our analysis shows that the model does not behave equally safely across all languages and prompt styles. Instead, safety performance depends strongly on both the languages used and the way a request is phrased using persuasive cues. We further observe that different risk categories exhibit different levels of vulnerability, with some types of harmful content being significantly more susceptible to persuasion-based jailbreaks than others. These findings reveal important limitations of current safety evaluations, which are largely English-centric, and underscore the need for multilingual and persuasion-aware benchmarking frameworks to more accurately assess real-world LLM safety. Our implementation is available at https://github.com/MonSaikat/IndicSafeEval. Warning: this paper contains example data that may be offensive or harmful. |
| 2026-09-03 | [Rethinking World Models for Safety-Critical Embodied Systems](http://arxiv.org/abs/2609.03774v1) | Kailang Ma, Heye Huang et al. | World models have progressed from compact latent dynamics to generative, controllable, and interactive simulators of embodied environments. However, high predictive likelihood and visual fidelity do not necessarily ensure that a model preserves the evidence required for safe decision-making. This perspective identifies three structural mismatches in current world modeling: likelihood versus risk, prediction versus intervention, and finite-horizon prediction versus accumulated consequences. We propose the Risk-Informed World Model (RIWM) as a decision-centric research direction for safety-critical embodied systems. RIWM organizes world modeling around consequences, intervention, epistemic uncertainty, and recoverability, and integrates four interdependent capabilities: decision-relevant representation, counterfactual reasoning, safety-critical episodic memory, and runtime safety assurance. It distinguishes physical, social, and operational consequences while using epistemic uncertainty to qualify the evidence supporting action. We further discuss open challenges in identifying consequential futures, validating counterfactual reasoning, maintaining revisable safety memories, translating learned consequences into executable constraints, and determining when evidence is sufficient to act. This perspective argues that future world models should move beyond predicting likely futures toward identifying which futures matter, revising judgments through experience, and recognizing when to act, revise, sense, defer, or abstain. |
| 2026-09-03 | [Virtual Testing of Automated Driving Systems through Credible Simulations](http://arxiv.org/abs/2609.03760v1) | Riccardo Dona, Espedito Rusciano et al. | Simulation is increasingly used to support safety-related decision-making in road transport, particularly for the assessment and approval of automated driving systems (ADS). The complexity of ADS behavior and size of their operational design domains make exclusive reliance on physical testing impractical, leading to extensive use of virtual testing (VT) during the approval phase. This shift raises critical questions regarding the credibility of modelling and simulation (M&S) results used to support road safety decisions. Current VT accreditation approaches in the ADS domain typically rely on validation-only practices, which have been shown to scale poorly when applied to complex, multi-tool simulation environments. To address this limitation, this paper proposes a risk-based framework for assessing the credibility of simulation toolchains used in ADS safety evaluation, drawing inspiration from established practices in other safety-critical domains, notably NASA's STD-7009 for models and simulations. The framework extends traditional verification and validation (V&V) by explicitly linking credibility requirements to the intended use of simulation outputs and to the safety criticality of the decisions they support within the approval process. It provides a lifecycle-oriented assessment scheme integrating toolchain management, modelling assumptions and limitations, verification, validation, and sensitivity analysis. Credibility acceptance thresholds are defined proportionally, allowing differentiated requirements depending on whether simulation is used for exploratory safety analysis, partial decision support, or as a substitute for physical testing. While demonstrated for ADS, the proposed approach is directly applicable to road safety and simulation studies where VT plays a central role in safety assessment and regulatory decision-making. |
| 2026-09-03 | [Proactive Service Agents: A Unified Decision Framework, Methods, and Evaluation](http://arxiv.org/abs/2609.03727v1) | Yan Tang, Tingyu Cao et al. | Large language model agents can plan, invoke tools, and modify external states, yet most systems still take an explicit user instruction as a fixed starting point. Proactive service moves the decision upstream: an agent must infer service opportunities from incomplete environmental and user signals, choose among remaining silent, asking, assisting, and acting, and account for interruption, misunderstanding, overreach, and privacy costs. This survey gives an operational definition centered on initiative and formulates the problem as a partially observable sequential decision process constrained by authorization and risk. The formulation represents timing, content, and delivery within one structured action, while making explicit the option value of waiting, the decision value of questions, and feedback-induced state changes. On this basis, we organize existing methods along one decision pipeline (state and need estimation, intervention gating, action construction, and feedback adaptation) and describe prescribed, predictive, model based, and return optimizing mechanisms as nonexclusive policy-construction components. We further normalize decision units and three-axis evidence descriptors across streaming dialogue, screen, video, software-engineering, and human-agent collaboration resources, and formalize metrics for triggering, timing, calibration, user burden, safety, and policy value. The synthesis shows why offline classification performance alone does not predict deployment benefit and why long-term memory is not a defining condition of proactivity. Reliable proactive service instead requires calibrated incremental intervention value, verifiable authorization, recoverable execution, and counterfactual evidence. |
| 2026-09-03 | [Can LLMs Extract Architectural Design Decisions from Source Code Commits? - A Preliminary Exploratory Study](http://arxiv.org/abs/2609.03721v1) | Amey Karan, Rudra Dhar et al. | Context: Architectural Design Decisions (ADDs) capture the rationale behind the structure and evolution of software systems but are rarely documented explicitly, and are often hidden inside source code commits. Recovering them is important for Architectural Knowledge Management (AKM). Problem: Extracting ADDs from commits is challenging due to their implicit and unstructured nature. Large Language Models (LLMs) have shown strong capabilities in understanding code and text, yet their effectiveness for this task remains underexplored. Study: We present a preliminary study using four LLMs (Gemini 3 Pro, DeepSeek R1, Kimi K2, Qwen3) with zeroshot and fewshot prompting on 30 developer-written ADDs from open-source projects. We score outputs with ROUGE-L, BLEU, METEOR, and BERTScore, and one author manually reviews the Gemini outputs. Results: All models reach a BERT-F1 above 0.81, and fewshot prompting improves alignment (Gemini BERT-F1: 0.828 to 0.847). However, the generated ADDs are often too long, implementation-focused, and miss the rationale behind the decision. This highlights opportunities for architecture-aware LLM systems and automated AKM. |
| 2026-09-03 | [Predictive Zonotope Reduction: Precise Runtime Monitoring under Uncertainty](http://arxiv.org/abs/2609.03699v1) | Vladimir Krsmanovic, Florian Kohn et al. | Robots operating in physical environments make control decisions based on uncertain sensor measurements, which can lead to unsafe or suboptimal actions. Runtime monitors that check their behavior against safety specifications must represent this uncertainty soundly. Zonotopes are a widely used representation, but continuously incorporating new measurements grows their order unboundedly, so monitors must periodically apply an over-approximating reduction. The choice of the reduction method substantially affects the zonotope's precision, yet existing approaches typically utilize a fixed method throughout the run, even though the optimal choice depends on the current state. This paper presents a Predictive Zonotope Reduction (PZR) approach, which frames reducer selection as an optimal control problem and solves it using beam-search model predictive control. Policy distillation into a small neural policy further provides substantially higher execution speed than model predictive control while maintaining improved performance, enabling uncertainty-aware runtime monitoring on resource-constrained real-time systems. We implement our approach in the RLola runtime monitoring framework and evaluate it on a 5-degree-of-freedom robotic arm simulated in MuJoCo, with sensor uncertainty modeled according to ISO 5725. Experiments on a Raspberry Pi 5 show that dynamic reduction significantly lowers false-positive rates in monitoring compared with static reduction strategies. |
| 2026-09-03 | [AlcaTRAz - Anchored Tree-Rule Defense Against Jailbreaks](http://arxiv.org/abs/2609.03693v1) | Jakub Reš, Petr Kaška et al. | Large language models (LLMs) are vulnerable to jailbreak attacks that bypass safety alignment through carefully crafted prompts. Many existing defenses require access to model weights or internals, making them difficult to apply to black-box deployments. We propose AlcaTRAz (Anchored Tree-Rule defense Against jailbreaks), a prompt-level defense based on rule trees that operates exclusively on the input text and requires no modification or retraining of the target model. The method automatically learns a transferable transformation rule that inserts controlled character-level perturbations at selected positions, thereby disrupting structural regularities exploited by jailbreak attacks while largely preserving the model's utility on benign queries. We evaluate the proposed method across 33 open-weight models, 22 jailbreak attack types, and a benchmark of short, single-turn benign questions, comparing against three representative prompt-level baselines (Llama Guard, RA-LLM, Goal Prioritization). Among the compared defenses, AlcaTRAz achieves the best composite security and functionality score in 73.4 % of model-attack combinations and shifts the aggregate score from a modal value of 10 (maximal-severity response to the malicious request) in the undefended setting to a modal value of 2 (near-refusal) after defense, while keeping the mean benign score within 0.27 points of the undefended baseline (8.35 vs. 8.62 on a 0-10 scale). AlcaTRAz substantially reduces but does not eliminate jailbreak success: a high-severity tail remains, and we do not consider adaptive attackers, so we position it as one layer within a defense-in-depth strategy rather than a standalone guarantee. |
| 2026-09-03 | [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](http://arxiv.org/abs/2609.03677v1) | Julian Truetsch, Felix Hauser et al. | Understanding the composition of large-scale autonomous driving datasets is essential for safety, robustness, and reliable operation across domains. For example, domain shift between locations could lead to the operating environment being misaligned with the training data, resulting in potentially dangerous performance degradation. Yet, existing data analysis pipelines largely rely on metadata, predefined labels, or manual inspection, which provide limited semantic insight or do not scale. This paper studies set difference captioning: given two subsets of images, the goal is to produce a natural-language hypothesis describing differences between the target and reference set. Building on a two-stage formulation, we adapt the method to autonomous driving by focusing on object-centric patches derived from object detection, which simplifies aggregation and enables attribution of differences to specific object instances or categories. To evaluate this setting in-domain, we introduce a new benchmark, AD-Diff Bench. Low-concentration experiments assess the suitability of set-difference-captioning approaches to sparse, real-world differences. We restrict our experiments to open-weight models to support reproducibility and ease of deployment. The proposed benchmark and analysis provide a step towards practical, human-interpretable dataset introspection for autonomous driving datasets. Our implementation and benchmark dataset are available at https://github.com/KIT-MRT/AD-Diff |
| 2026-09-03 | [EraseSAE: Surgical Concept Erasure in Text-to-Video Diffusion Models via Sparse Autoencoders](http://arxiv.org/abs/2609.03629v1) | Xinghao Wang, Dong Li et al. | Recent advances in text-to-video (T2V) diffusion models have demonstrated remarkable generative capabilities, yet their reliance on loosely curated training data raises pressing safety and copyright concerns. Concept erasure offers a principled remedy by removing unwanted semantics from pretrained models while preserving remaining concepts. However, existing approaches typically operate at a coarse granularity misaligned with the fine-grained, distributed nature of concept representations, leading to incomplete removal or degraded generation quality. We argue that surgical erasure fundamentally requires intervention at the level of monosemantic features, where each unit encodes a single interpretable concept. To this end, we propose EraseSAE, a novel framework that leverages sparse autoencoders to achieve surgical concept erasure in DiT-based T2V diffusion models via a principled decompose-attribute-erase pipeline. We first introduce the Partitioned Convolutional Sparse Autoencoder, which decomposes dense spatiotemporal activations into disentangled, interpretable sparse features while preserving spatiotemporal coherence. A contrastive attribution mechanism then contrasts activations from paired prompts to isolate concept-specific feature kernels. At inference, timestep-resolved spatiotemporal masks derived from the identified kernels confine erasure to regions where the target concept is active, leaving unrelated content intact. Extensive experiments across diverse diffusion models and concept erasure tasks demonstrate that EraseSAE achieves precise and robust concept removal with minimal quality degradation, substantially outperforming state-of-the-art methods. The code is available at https://github.com/HiDream-ai/EraseSAE. |
| 2026-09-03 | [SV-WAM: An Efficient Surround-View World-Action Model for End-to-End Autonomous Driving](http://arxiv.org/abs/2609.03602v1) | Jinyang Wang, Shiwei Li et al. | World models (WMs) have demonstrated strong potential for end-to-end autonomous driving by learning predictive representations of future scene dynamics. However, generating future videos during inference introduces substantial computational overhead, leading many recent driving WMs to adopt a single front camera as input for efficient deployment. This design restricts spatial coverage in safety-critical maneuvers such as lane changes, merges, and turns. To address this limitation, we propose SV-WAM, a surround-view world-action model (WAM) that preserves full six-camera observations while maintaining efficient inference. SV-WAM leverages future-video prediction as dense training supervision for action learning within a shared generative model, rather than as an inference-time output. At the core of this design is an action-centered causal mask that prevents action tokens from attending to future-video tokens during joint action-video denoising. Consequently, the video branch can be discarded at deployment, enabling efficient action-only planning. Furthermore, we introduce a differentiable drivable-area compliance regularizer that penalizes vehicle-footprint corners approaching or crossing drivable boundaries, improving planning safety and boundary awareness. Extensive experiments on the closed-loop NAVSIMv2 benchmark and the open-loop nuScenes benchmark demonstrate that SV-WAM achieves state-of-the-art planning performance with low inference latency and competitive zero-shot transfer capability. |
| 2026-09-03 | [SafeRI: Recognition and Intervention for Token-Level Safety Intervention in Large Vision Language Models](http://arxiv.org/abs/2609.03544v1) | Caoyuan Ma, Tian Gu et al. | Existing safety alignment methods for vision-language models usually modify the model behavior globally: once the safety parameters are trained or loaded, they participate in both unsafe and already-safe generations. This always-on intervention can unnecessarily perturb the model's original reasoning path and degrade general multimodal capabilities. We argue that safety alignment should be an on-demand intervention rather than a permanent modification to every decoding trajectory. To this end, we propose a streaming recognition and gated LoRA framework for intrinsic VLM safety. During autoregressive generation, a lightweight recognizer estimates whether the current pre-token generation state is safe or unsafe. Its output updates the LoRA gate for the following decoding step; otherwise, generation follows the frozen-backbone policy. The LoRA module is trained from unsafe prefixes, transition statements, and safe continuations, so that it learns to redirect unsafe generations back to safe responses after activation. Experiments across multiple safety and general-purpose benchmarks demonstrate the effectiveness of our method in post-alignment settings. |
| 2026-09-03 | [When Retrieval Helps: Selective Retrieval for Single-Turn Mental-Health QA](http://arxiv.org/abs/2609.03454v1) | Hyunseo Oh, Chong-Kwon Kim et al. | Retrieval-augmented generation (RAG) can improve the specificity and grounding of large language model responses, but its effect is not uniformly beneficial in single-turn mental-health question answering, where user queries often combine emotional distress, treatment concerns, and safety-sensitive needs. We study when retrieval helps or hurts mental-health QA, and whether a lightweight selective retrieval policy can better control this trade-off. We operationalize retrieval need using three draft-conditioned utility dimensions: psychoeducational need, coping need, and response specificity, together with a rule-based safety trigger. Following psychotherapy-grounded RAG systems such as coTherapist, we construct a compact and controllable guideline corpus comprising coping-strategy, psychoeducational, and safety resources. We fine-tune an instruction-tuned generator on MentalChat16K using QLoRA and compare Closed-book, Always Retrieval, and Selective Retrieval settings on CounselBench-Eval and CounselBench-Adv. Experiments show that retrieval is not uniformly beneficial in this domain. Always Retrieval improves specificity but lowers overall quality and introduces additional safety-sensitive failures. Selective Retrieval preserves closed-book behavior for low-need cases while avoiding the additional degradation caused by unconditional retrieval, supporting the view that retrieval activation is a safety-sensitive control decision. |
| 2026-09-03 | [It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories](http://arxiv.org/abs/2609.03436v1) | Yigit Utku Bulut | Reasoning traces of large language models are widely read as containing "breakthrough" moments and early-legible fates. Both readings rest on measurements missing a counterfactual control at the level of the claim; we supply both controls. First, a restart-controlled truncation probe separates when a solution fits the continuation budget from when a prefix carries value that fresh computation cannot buy, comparing per-anchor continuation solve rates against from-scratch restart curves at matched total generated-token budget. Applied to 178 problem-model cells (89 MATH problems x two small open models, an outcome-blind but difficulty-targeted cohort), exactly 1 of 178 cells survives as prefix-limited; restart dose-response separates a compute-starved model from a capability-limited one; and wherever the matched budget lies inside the restart grid, continuing the model's own prefix beats restarting (9 of 9) -- predominantly compute compression rather than expanded reachability. Second, a pre-registered, difficulty-controlled test finds no detectable outcome information in early-window internal signals beyond a problem-difficulty baseline, and two generation-free analyses of public corpora show why this control is needed: a trace-blind difficulty proxy reaches AUROC 0.873 on 192K DeepSeek-R1 generations -- inside the published probe range -- and a closely matched reconstruction of the closest published early-window positive recovers a comparable pooled result (0.849) while within problem it is statistically indistinguishable from chance at all ten anchors (0.496 at t=4); a post-hoc within-targeted probe finds only a small average residual, concentrated in three low-failure problems. High pooled probe AUROCs cannot by themselves establish within-attempt information; a question-only baseline or within-problem evaluation is required. |

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



