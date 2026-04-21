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
| 2026-04-20 | [Symbolic Synthesis for LTLf+ Obligations](http://arxiv.org/abs/2604.18532v1) | Giuseppe De Giacomo, Christian Hagemeier et al. | We study synthesis for obligation properties expressed in LTLfp, the extension of LTLf to infinite traces. Obligation properties are positive Boolean combinations of safety and guarantee (co-safety) properties and form the second level of the temporal hierarchy of Manna and Pnueli. Although obligation properties are expressed over infinite traces, they retain most of the simplicity of LTLf. In particular, we show that they admit a translation into symbolically represented deterministic weak automata (DWA) obtained directly from the symbolic deterministic finite automata (DFA) for the underlying LTLf properties on trace prefixes. DWA inherit many of the attractive algorithmic features of DFA, including Boolean closure and polynomial-time minimization. Moreover, we show that synthesis for LTLfp obligation properties is theoretically highly efficient - solvable in linear time once the DWA is constructed. We investigate several symbolic algorithms for solving DWA games that arise in the synthesis of obligation properties and evaluate their effectiveness experimentally. Overall, the results indicate that synthesis for LTLfp obligation properties can be performed with virtually the same effectiveness as LTLf synthesis. |
| 2026-04-20 | [LLM Safety From Within: Detecting Harmful Content with Internal Representations](http://arxiv.org/abs/2604.18519v1) | Difan Jiao, Yilun Liu et al. | Guard models are widely used to detect harmful content in user prompts and LLM responses. However, state-of-the-art guard models rely solely on terminal-layer representations and overlook the rich safety-relevant features distributed across internal layers. We present SIREN, a lightweight guard model that harnesses these internal features. By identifying safety neurons via linear probing and combining them through an adaptive layer-weighted strategy, SIREN builds a harmfulness detector from LLM internals without modifying the underlying model. Our comprehensive evaluation shows that SIREN substantially outperforms state-of-the-art open-source guard models across multiple benchmarks while using 250 times fewer trainable parameters. Moreover, SIREN exhibits superior generalization to unseen benchmarks, naturally enables real-time streaming detection, and significantly improves inference efficiency compared to generative guard models. Overall, our results highlight LLM internal states as a promising foundation for practical, high-performance harmfulness detection. |
| 2026-04-20 | [Different Paths to Harmful Compliance: Behavioral Side Effects and Mechanistic Divergence Across LLM Jailbreaks](http://arxiv.org/abs/2604.18510v1) | Md Rysul Kabir, Zoran Tiganj | Open-weight language models can be rendered unsafe through several distinct interventions, but the resulting models may differ substantially in capabilities, behavioral profile, and internal failure mode. We study behavioral and mechanistic properties of jailbroken models across three unsafe routes: harmful supervised fine-tuning (SFT), harmful reinforcement learning with verifiable rewards (RLVR), and refusal-suppressing abliteration. All three routes achieve near-ceiling harmful compliance, but they diverge once we move beyond direct harmfulness. RLVR-jailbroken models show minimal degradation and preserve explicit harm recognition in a structured self-audit: they are able to identify harmful prompts and describe how a safe LLM should respond, yet they comply with the harmful request. With RLVR, harmful behavior is strongly suppressed by a reflective safety scaffold: when a harmful prompt is prepended with an instruction to reflect on safety standards, harmful behavior drops close to the baseline. Category-specific RLVR jailbreaks generalize broadly across harmfulness domains. Models jailbroken with SFT show the largest collapse in explicit safety judgments, the highest behavioral drift, and a substantial capability loss on standard benchmarks. Abliteration is family-dependent in both self-audit and response to a reflective safety scaffold. Mechanistic and repair analyses further separate the routes: abliteration is consistent with localized refusal-feature deletion, RLVR with preserved safety geometry but retargeted policy behavior, and SFT with broader distributed drift. Targeted repair partially recovers RLVR-jailbroken models, but has little effect on SFT-jailbroken models. Together, these results show that jailbreaks can produce vastly different properties despite similar harmfulness, with models jailbroken via RLVR showing remarkable similarity to the base model. |
| 2026-04-20 | [Adversarial Humanities Benchmark: Results on Stylistic Robustness in Frontier Model Safety](http://arxiv.org/abs/2604.18487v1) | Marcello Galisai, Susanna Cifani et al. | The Adversarial Humanities Benchmark (AHB) evaluates whether model safety refusals survive a shift away from familiar harmful prompt forms. Starting from harmful tasks drawn from MLCommons AILuminate, the benchmark rewrites the same objectives through humanities-style transformations while preserving intent. This extends literature on Adversarial Poetry and Adversarial Tales from single jailbreak operators to a broader benchmark family of stylistic obfuscation and goal concealment. In the benchmark results reported here, the original attacks record 3.84% attack success rate (ASR), while transformed methods range from 36.8% to 65.0%, yielding 55.75% overall ASR across 31 frontier models. Under a European Union AI Act Code-of-Practice-inspired systemic-risk lens, Chemical, biological, radiological and nuclear (CBRN) is the highest bucket. Taken together, this lack of stylistic robustness suggests that current safety techniques suffer from weak generalization: deep understanding of 'non-maleficence' remains a central unresolved problem in frontier model safety. |
| 2026-04-20 | [Safe Control using Learned Safety Filters and Adaptive Conformal Inference](http://arxiv.org/abs/2604.18482v1) | Sacha Huriot, Ihab Tabbara et al. | Safety filters have been shown to be effective tools to ensure the safety of control systems with unsafe nominal policies. To address scalability challenges in traditional synthesis methods, learning-based approaches have been proposed for designing safety filters for systems with high-dimensional state and control spaces. However, the inevitable errors in the decisions of these models raise concerns about their reliability and the safety guarantees they offer. This paper presents Adaptive Conformal Filtering (ACoFi), a method that combines learned Hamilton-Jacobi reachability-based safety filters with adaptive conformal inference. Under ACoFi, the filter dynamically adjusts its switching criteria based on the observed errors in its predictions of the safety of actions. The range of possible safety values of the nominal policy's output is used to quantify uncertainty in safety assessment. The filter switches from the nominal policy to the learned safe one when that range suggests it might be unsafe. We show that ACoFi guarantees that the rate of incorrectly quantifying uncertainty in the predicted safety of the nominal policy is asymptotically upper bounded by a user-defined parameter. This gives a soft safety guarantee rather than a hard safety guarantee. We evaluate ACoFi in a Dubins car simulation and a Safety Gymnasium environment, empirically demonstrating that it significantly outperforms the baseline method that uses a fixed switching threshold by achieving higher learned safety values and fewer safety violations, especially in out-of-distribution scenarios. |
| 2026-04-20 | [SemLT3D: Semantic-Guided Expert Distillation for Camera-only Long-Tailed 3D Object Detection](http://arxiv.org/abs/2604.18476v1) | Hao Vo, Khoa Vo et al. | Camera-only 3D object detection has emerged as a cost-effective and scalable alternative to LiDAR for autonomous driving, yet existing methods primarily prioritize overall performance while overlooking the severe long-tail imbalance inherent in real-world datasets. In practice, many rare but safety-critical categories such as children, strollers, or emergency vehicles are heavily underrepresented, leading to biased learning and degraded performance. This challenge is further exacerbated by pronounced inter-class ambiguity (e.g., visually similar subclasses) and substantial intra-class diversity (e.g., objects varying widely in appearance, scale, pose, or context), which together hinder reliable long-tail recognition. In this work, we introduce SemLT3D, a Semantic-Guided Expert Distillation framework designed to enrich the representation space for underrepresented classes through semantic priors. SemLT3D consists of: (1) a language-guided mixture-of-experts module that routes 3D queries to specialized experts according to their semantic affinity, enabling the model to better disentangle confusing classes and specialize on tail distributions; and (2) a semantic projection distillation pipeline that aligns 3D queries with CLIP-informed 2D semantics, producing more coherent and discriminative features across diverse visual manifestations. Although motivated by long-tail imbalance, the semantically structured learning in SemLT3D also improves robustness under broader appearance variations and challenging corner cases, offering a principled step toward more reliable camera-only 3D perception. |
| 2026-04-20 | [Train Separately, Merge Together: Modular Post-Training with Mixture-of-Experts](http://arxiv.org/abs/2604.18473v1) | Jacob Morrison, Sanjay Adhikesaven et al. | Extending a fully post-trained language model with new domain capabilities is fundamentally limited by monolithic training paradigms: retraining from scratch is expensive and scales poorly, while continued training often degrades existing capabilities. We present BAR (Branch-Adapt-Route), which trains independent domain experts, each through its own mid-training, supervised finetuning, and reinforcement learning pipeline, and composes them via a Mixture-of-Experts architecture with lightweight router training. Unlike retraining approaches that mix all domains and require full reprocessing for any update (with cost scaling quadratically), BAR enables updating individual experts independently with linear cost scaling and no degradation to existing domains. At the 7B scale, with experts for math, code, tool use, and safety, BAR achieves an overall score of 49.1 (averaged across 7 evaluation categories), matching or exceeding re-training baselines (47.8 without mid-training, 50.5 with). We further show that modular training provides a structural advantage: by isolating each domain, it avoids the catastrophic forgetting that occurs when late-stage RL degrades capabilities from earlier training stages, while significantly reducing the cost and complexity of updating or adding a domain. Together, these results suggest that decoupled, expert-based training is a scalable alternative to monolithic retraining for extending language models. |
| 2026-04-20 | [Asset Harvester: Extracting 3D Assets from Autonomous Driving Logs for Simulation](http://arxiv.org/abs/2604.18468v1) | Tianshi Cao, Jiawei Ren et al. | Closed-loop simulation is a core component of autonomous vehicle (AV) development, enabling scalable testing, training, and safety validation before real-world deployment. Neural scene reconstruction converts driving logs into interactive 3D environments for simulation, but it does not produce complete 3D object assets required for agent manipulation and large-viewpoint novel-view synthesis. To address this challenge, we present Asset Harvester, an image-to-3D model and end-to-end pipeline that converts sparse, in-the-wild object observations from real driving logs into complete, simulation-ready assets. Rather than relying on a single model component, we developed a system-level design for real-world AV data that combines large-scale curation of object-centric training tuples, geometry-aware preprocessing across heterogeneous sensors, and a robust training recipe that couples sparse-view-conditioned multiview generation with 3D Gaussian lifting. Within this system, SparseViewDiT is explicitly designed to address limited-angle views and other real-world data challenges. Together with hybrid data curation, augmentation, and self-distillation, this system enables scalable conversion of sparse AV object observations into reusable 3D assets. |
| 2026-04-20 | [Using large language models for embodied planning introduces systematic safety risks](http://arxiv.org/abs/2604.18463v1) | Tao Zhang, Kaixian Qu et al. | Large language models are increasingly used as planners for robotic systems, yet how safely they plan remains an open question. To evaluate safe planning systematically, we introduce DESPITE, a benchmark of 12,279 tasks spanning physical and normative dangers with fully deterministic validation. Across 23 models, even near-perfect planning ability does not ensure safety: the best-planning model fails to produce a valid plan on only 0.4% of tasks but produces dangerous plans on 28.3%. Among 18 open-source models from 3B to 671B parameters, planning ability improves substantially with scale (0.4-99.3%) while safety awareness remains relatively flat (38-57%). We identify a multiplicative relationship between these two capacities, showing that larger models complete more tasks safely primarily through improved planning, not through better danger avoidance. Three proprietary reasoning models reach notably higher safety awareness (71-81%), while non-reasoning proprietary models and open-source reasoning models remain below 57%. As planning ability approaches saturation for frontier models, improving safety awareness becomes a central challenge for deploying language-model planners in robotic systems. |
| 2026-04-20 | [From Awareness to Intent: Mitigating Silent Driving System Failures through Prospective Situation Awareness Enhancing Interfaces](http://arxiv.org/abs/2604.18449v1) | Jiyao Wang, Song Yan et al. | Silent automation failures, where a system fails to detect a hazard without warning, pose a critical safety challenge for partially automated vehicles. While research has mostly focused on takeover requests, how to support a driver in silent failure remains underexplored. We conducted a multi-modal driving simulator study with 48 participants to investigate how different Prospective Situation Awareness Enhancement (PSAE) interfaces, delivered via augmented reality head-up display, affect takeover performance. By integrating behavioral, subjective psychological, and physiological data, our analysis suggests that situational awareness (SA) serves as an important moderating factor through which PSAE interfaces improve takeover performance. Further, we found that providing perceptual cues was most effective in enhancing SA, while communicating system intent was superior for building trust. Finally, we identified a potential correlate of SA in the neuroactivity. Overall, this paper contributes to understanding how transparency-oriented interfaces may support drivers and provides design insights into HMI design for silent failures. |
| 2026-04-20 | [Towards an Agentic LLM-based Approach to Requirement Formalization from Unstructured Specifications](http://arxiv.org/abs/2604.18228v1) | Alberto Tagliaferro, Bruno Guindani et al. | Early-stage specifications of safety-critical systems are typically expressed in natural language, making it difficult to derive formal properties suitable for verification and needed to guarantee safety. While recent Large Language Model (LLM)-based approaches can generate formal artifacts from text, they mainly focus on syntactic correctness and do not ensure semantic alignment between informal requirements and formally verifiable properties. We propose an agentic methodology that automatically extracts verification-ready properties from unstructured specifications. The modular pipeline combines requirement extraction, compatibility filtering with respect to a target formalism, and translation into formal properties. Experimental results across three scenarios show that the pipeline generates syntactically and semantically aligned formal properties with a 77.8% accuracy. By explicitly accounting for modeling and verification constraints, the approach is a paving step towards exploiting Artificial Intelligence (AI) to bridge the gap between informal descriptions and semantically meaningful formal verification. |
| 2026-04-20 | [Architectural Design Decisions in AI Agent Harnesses](http://arxiv.org/abs/2604.18071v1) | Hu Wei | AI agent systems increasingly rely on reusable non-LLM engineering infrastructure that packages tool mediation, context handling, delegation, safety control, and orchestration. Yet the architectural design decisions in this surrounding infrastructure remain understudied. This paper presents a protocol-guided, source-grounded empirical study of 70 publicly available agent-system projects, addressing three questions: which design-decision dimensions recur across projects, which co-occurrences structure those decisions, and which typical architectural patterns emerge. Methodologically, we contribute a transparent investigation procedure for analyzing heterogeneous agent-system corpora through source-code and technical-material reading. Empirically, we identify five recurring design dimensions (subagent architecture, context management, tool systems, safety mechanisms, and orchestration) and find that the corpus favors file-persistent, hybrid, and hierarchical context strategies; registry-oriented tool systems remain dominant while MCP- and plugin-oriented extensions are emerging; and intermediate isolation is common but high-assurance audit is rare. Cross-project co-occurrence analysis reveals that deeper coordination pairs with more explicit context services, stronger execution environments with more structured governance, and formalized tool-registration boundaries with broader ecosystem ambitions. We synthesize five recurring architectural patterns spanning lightweight tools, balanced CLI frameworks, multi-agent orchestrators, enterprise systems, and scenario-verticalized projects. The result provides an evidence-based account of architectural regularities in agent-system engineering, with grounded guidance for framework designers, selectors, and researchers. |
| 2026-04-20 | [Optimizing Memory Allocation in Distributed Clusters with Predictive Modeling](http://arxiv.org/abs/2604.18043v1) | Jonathan Bader, Edgar Blumenthal et al. | In modern distributed systems, efficient resource allocation is a vital aspect to maintain scalability, reduce operational costs, and ensure fast execution even across heterogeneous workloads. Predictive models for resource usage are essential tools for optimizing allocation and preventing system bottlenecks. Predictive memory allocation has asymmetric costs as a key challenge: underallocation causes failures while overallocation wastes memory.   We propose a regression method based on a LightGBM and XGBoost ensemble trained to predict high conditional quantiles. To further account for the high cost of underallocations we add a multiplicative safety factor. With our method we are able to reduce the number of under-allocated jobs from 4.17% to 2.89% and average overallocation from 148% to 44.51% on a real-world dataset of build jobs provided by SAP. We further explore the pareto frontier between optimization for underallocation and for overallocation. |
| 2026-04-20 | [CodePivot: Bootstrapping Multilingual Transpilation in LLMs via Reinforcement Learning without Parallel Corpora](http://arxiv.org/abs/2604.18027v1) | Shangyu Li, Juyong Jiang et al. | Transpilation, or code translation, aims to convert source code from one programming language (PL) to another. It is beneficial for many downstream applications, from modernizing large legacy codebases to augmenting data for low-resource PLs. Recent large language model (LLM)-based approaches have demonstrated immense potential for code translation. Among these approaches, training-based methods are particularly important because LLMs currently do not effectively adapt to domain-specific settings that suffer from a lack of knowledge without targeted training. This limitation is evident in transpilation tasks involving low-resource PLs. However, existing training-based approaches rely on a pairwise transpilation paradigm, making it impractical to support a diverse range of PLs. This limitation is particularly prominent for low-resource PLs due to a scarcity of training data. Furthermore, these methods suffer from suboptimal reinforcement learning (RL) reward formulations. To address these limitations, we propose CodePivot, a training framework that leverages Python as an intermediate representation (IR), augmented by a novel RL reward mechanism, Aggressive-Partial-Functional reward, to bootstrap the model's multilingual transpilation ability without requiring parallel corpora. Experiments involving 10 PLs show that the resulting 7B model, trained on Python-to-Others tasks, consistently improves performance across both general and low-resource PL-related transpilation tasks. It outperforms substantially larger mainstream models with hundreds of billions more parameters, such as Deepseek-R1 and Qwen3-235B-A22B-Instruct-2507, on Python-to-Others tasks and Others-to-All tasks, respectively. In addition, it outperforms its counterpart trained directly on Any-to-Any tasks on general transpilation tasks. The code and data are available at https://github.com/lishangyu-hkust/CodePivot. |
| 2026-04-20 | [Trustworthy Endoscopic Super-Resolution](http://arxiv.org/abs/2604.18001v1) | Julio Silva-Rodríguez, Ender Konukoglu | Super-resolution (SR) models are attracting growing interest for enhancing minimally invasive surgery and diagnostic videos under hardware constraints. However, valid concerns remain regarding the introduction of hallucinated structures and amplified noise, limiting their reliability in safety-critical settings. We propose a direct and practical framework to make SR systems more trustworthy by identifying where reconstructions are likely to fail. Our approach integrates a lightweight error-prediction network that operates on intermediate representations to estimate pixel-wise reconstruction error. The module is computationally efficient and low-latency, making it suitable for real-time deployment. We convert these predictions into operational failure decisions by constructing Conformal Failure Masks (CFM), which localize regions where the SR output should not be trusted. Built on conformal risk control principles, our method provides theoretical guarantees for controlling both the tolerated error limit and the miscoverage in detected failures. We evaluate our approach on image and video SR, demonstrating its effectiveness in detecting unreliable reconstructions in endoscopic and robotic surgery settings. To our knowledge, this is the first study to provide a model-agnostic, theoretically grounded approach to improving the safety of real-time endoscopic image SR. |
| 2026-04-20 | [Causally-Constrained Probabilistic Forecasting for Time-Series Anomaly Detection](http://arxiv.org/abs/2604.17998v1) | Pooyan Khosravinia, João Gama et al. | Anomaly detection in multivariate time series is a central challenge in industrial monitoring, as failures frequently arise from complex temporal dynamics and cross-sensor interactions. While recent deep learning models, including graph neural networks and Transformers, have demonstrated strong empirical performance, most approaches remain primarily correlational and offer limited support for causal interpretation and root-cause localization. This study introduces a causally-constrained probabilistic forecasting framework which is a Causally Guided Transformer (CGT) model for multivariate time-series anomaly detection, integrating an explicit time-lagged causal graph prior with deep sequence modeling. For each target variable, a dedicated forecasting block employs a hard parent mask derived from causal discovery to restrict the main prediction pathway to graph-supported causes, while a latent Gaussian head captures predictive uncertainty. To leverage residual correlational information without compromising the causal representation, a shadow auxiliary path with stop-gradient isolation and a safety-gated blending mechanism is incorporated to suppress non-causal contributions when reliability is low. Anomalies are identified using negative log-likelihood scores with adaptive streaming thresholding, and root-cause variables are determined through per-dimension probabilistic attribution and counterfactual clamping. Experiments on the ASD and SMD benchmarks indicate that the proposed method achieves state-of-the-art detection performance, with F1-scores of 96.19% on ASD and 95.32% on SMD, and enhances variable-level attribution quality. These findings suggest that causal structural priors can improve both robustness and interpretability in detecting deep anomalies in multivariate sensor systems. |
| 2026-04-20 | [Employing General-Purpose and Biomedical Large Language Models with Advanced Prompt Engineering for Pharmacoepidemiologic Study Design](http://arxiv.org/abs/2604.17988v1) | Xinyao Zhang, Nicole Sonne Heckmann et al. | Background: The potential of large language models (LLMs) to automate and support pharmacoepidemiologic study design is an emerging area of interest, yet their reliability remains insufficiently characterized. General-purpose LLMs often display inaccuracies, while the comparative performance of specialized biomedical LLMs in this domain remains unknown. Methods: This study evaluated general-purpose LLMs (GPT-4o and DeepSeek-R1) versus biomedically fine-tuned LLMs (QuantFactory/Bio-Medical-Llama-3-8B-GGUF and Irathernotsay/qwen2-1.5B-medical_qa-Finetune) using 46 protocols (2018-2024) from the HMA-EMA Catalogue and Sentinel System. Performance was assessed across relevance, logic of justification, and ontology-code agreement across multiple coding systems using Least-to-Most (LTM) and Active Prompting strategies. Results: GPT-4o and DeepSeek-R1 paired with LTM prompting achieved the highest relevance and logic of justification scores, with GPT-4o-LTM reaching a median relevance score of 4 in 8 of 9 questions for HMA-EMA protocols. Biomedical LLMs showed lower relevance overall and frequently generated insufficient justification. All LLMs demonstrated limited proficiency in ontology-code mapping, although LTM provided the most consistent improvements in reasoning stability. Conclusion: Off-the-shelf general-purpose LLMs currently offer superior support for pharmacoepidemiologic design compared to biomedical LLMs. Prompt strategy strongly influenced LLM performance. |
| 2026-04-20 | [Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](http://arxiv.org/abs/2604.17984v1) | Junyoung Yang, Kyungmin Kim et al. | Uncertainty quantification is crucial in safety-critical systems, where decisions must be made under uncertainty. In particular, we consider the problem of online uncertainty quantification, where data points arrive sequentially. Online conformal prediction is a principled online uncertainty quantification method that dynamically constructs a prediction set at each time step. While existing methods for online conformal prediction provide long-run coverage guarantees without any distributional assumptions, they typically assume a full feedback setting in which the true label is always observed. In this paper, we propose a novel learning method for online conformal prediction with partial feedback from an adaptive adversary-a more challenging setup where the true label is revealed only when it lies inside the constructed prediction set. Specifically, we formulate online conformal prediction as an adversarial bandit problem by treating each candidate prediction set as an arm. Building on an existing algorithm for adversarial bandits, our method achieves a long-run coverage guarantee by explicitly establishing its connection to the regret of the learner. Finally, we empirically demonstrate the effectiveness of our method in both independent and identically distributed (i.i.d.) and non-i.i.d. settings, showing that it successfully controls the miscoverage rate while maintaining a reasonable size of the prediction set. |
| 2026-04-20 | [TPS-CalcBench: A Benchmark and Diagnostic Evaluation Framework for LLM Analytical Calculation Competence in Hypersonic Thermal Protection System Engineering](http://arxiv.org/abs/2604.17966v1) | Jinglai Zheng, Chuhan Qiao et al. | Deploying LLMs as reasoning assistants in safety-critical aerospace engineering requires stricter evaluation criteria than general scientific benchmarks. In hypersonic thermal protection system (TPS) design, inaccurate stagnation-point heat flux or boundary-layer calculations may cause catastrophic design margin violations. Models with numerically reasonable but physically invalid answers are more dangerous than those declining to respond. Current scientific benchmarks only test abstract math and basic physics, evaluate final answers solely, ignore engineering reasoning processes, and cannot detect such critical failures. We propose TPS-CalcBench, the first diagnostic benchmark for closed-form analytical calculations in hypersonic aerodynamics and high-temperature gas dynamics that experienced TPS engineers conduct without simulations. Our contributions include domain-oriented task taxonomy with 4 difficulty levels and 8 categories from Anderson's textbook, dual-track evaluation measuring result accuracy and reasoning quality via an 8-dimension rubric and calibrated judge with human audit to identify right answer wrong reasoning issues, human-AI data pipeline producing 420 high-confidence core items and 810 noise-controlled pre-gating items from 4560 raw data, noise-sensitivity analysis measuring data quality impacts on model ranking, and three diagnostic intervention methods: DFA-TPS fine-tuning, RAG-EQ retrieval grounding and PA-CoT process-aware prompting. Tests on 13 models from 7 groups show wide performance differences (KPI 12.6-87.9), hidden formula selection defects, data-driven rank changes and effective intervention improvements, establishing a complete diagnose-evaluate-intervene framework for safety-critical engineering LLM deployment assessment. |
| 2026-04-20 | [Quantitative Verification of Constrained Occupation Time for Stochastic Discrete-time Systems](http://arxiv.org/abs/2604.17902v1) | Bai Xue, Peixin Wang et al. | This paper addresses the quantitative verification of constrained occupation time in stochastic discrete-time systems, focusing on the probability of visiting a target set at least $k$ times while maintaining safety. Such cumulative properties are essential for certifying repeated behaviors like surveillance and periodic charging. To address this, we present the first barrier certificate framework capable of certifying these behaviors. We introduce multiplicative stochastic barrier functions that encode visitation counts implicitly within the algebraic structure of a scalar barrier. By adopting a switched-system reformulation to handle safety, we derive rigorous probabilistic bounds for both finite and infinite horizons. Specifically, we show that dissipative barriers establish upper bounds ensuring the exponential decay of frequent visits, while attractive barriers provide lower bounds via submartingale analysis. The efficacy of the proposed framework is demonstrated through numerical examples. |

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



