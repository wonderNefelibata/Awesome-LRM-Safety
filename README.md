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
| 2026-03-12 | [Proof-Carrying Materials: Falsifiable Safety Certificates for Machine-Learned Interatomic Potentials](http://arxiv.org/abs/2603.12183v1) | Abhinaba Basu, Pavan Chakraborty | Machine-learned interatomic potentials (MLIPs) are deployed for high-throughput materials screening without formal reliability guarantees. We show that a single MLIP used as a stability filter misses 93% of density functional theory (DFT)-stable materials (recall 0.07) on a 25,000-material benchmark. Proof-Carrying Materials (PCM) closes this gap through three stages: adversarial falsification across compositional space, bootstrap envelope refinement with 95% confidence intervals, and Lean 4 formal certification. Auditing CHGNet, TensorNet and MACE reveals architecture-specific blind spots with near-zero pairwise error correlations (r <= 0.13; n = 5,000), confirmed by independent Quantum ESPRESSO validation (20/20 converged; median DFT/CHGNet force ratio 12x). A risk model trained on PCM-discovered features predicts failures on unseen materials (AUC-ROC = 0.938 +/- 0.004) and transfers across architectures (cross-MLIP AUC-ROC ~ 0.70; feature importance r = 0.877). In a thermoelectric screening case study, PCM-audited protocols discover 62 additional stable materials missed by single-MLIP screening - a 25% improvement in discovery yield. |
| 2026-03-12 | [When do modal definability and preservation theorems transfer to the finite?](http://arxiv.org/abs/2603.12171v1) | Johan van Benthem, Balder ten Cate et al. | We study which classic modal definability and preservation results survive when attention is restricted to finite structures, where many first-order transfer theorems are known to break down. Several semantic characterizations for modal formula classes survive the passage to the finite, while a number of first-order preservation theorems for basic frame operations fail. Our main positive result is that the Bisimulation Safety Theorem does transfer to finite structures. We also discuss computability aspects, and analogues in the finite for the Goldblatt-Thomason theorem and for modal correspondence theory. |
| 2026-03-12 | [Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems](http://arxiv.org/abs/2603.12023v1) | Sarbartha Banerjee, Prateek Sahu et al. | Rapid progress in generative AI has given rise to Compound AI systems - pipelines comprised of multiple large language models (LLM), software tools and database systems. Compound AI systems are constructed on a layered traditional software stack running on a distributed hardware infrastructure. Many of the diverse software components are vulnerable to traditional security flaws documented in the Common Vulnerabilities and Exposures (CVE) database, while the underlying distributed hardware infrastructure remains exposed to timing attacks, bit-flip faults, and power-based side channels. Today, research targets LLM-specific risks like model extraction, training data leakage, and unsafe generation -- overlooking the impact of traditional system vulnerabilities.   This work investigates how traditional software and hardware vulnerabilities can complement LLM-specific algorithmic attacks to compromise the integrity of a compound AI pipeline. We demonstrate two novel attacks that combine system-level vulnerabilities with algorithmic weaknesses: (1) Exploiting a software code injection flaw along with a guardrail Rowhammer attack to inject an unaltered jailbreak prompt into an LLM, resulting in an AI safety violation, and (2) Manipulating a knowledge database to redirect an LLM agent to transmit sensitive user data to a malicious application, thus breaching confidentiality. These attacks highlight the need to address traditional vulnerabilities; we systematize the attack primitives and analyze their composition by grouping vulnerabilities by their objective and mapping them to distinct stages of an attack lifecycle. This approach enables a rigorous red-teaming exercise and lays the groundwork for future defense strategies. |
| 2026-03-12 | [LABSHIELD: A Multimodal Benchmark for Safety-Critical Reasoning and Planning in Scientific Laboratories](http://arxiv.org/abs/2603.11987v1) | Qianpu Sun, Xiaowei Chi et al. | Artificial intelligence is increasingly catalyzing scientific automation, with multimodal large language model (MLLM) agents evolving from lab assistants into self-driving lab operators. This transition imposes stringent safety requirements on laboratory environments, where fragile glassware, hazardous substances, and high-precision laboratory equipment render planning errors or misinterpreted risks potentially irreversible. However, the safety awareness and decision-making reliability of embodied agents in such high-stakes settings remain insufficiently defined and evaluated. To bridge this gap, we introduce LABSHIELD, a realistic multi-view benchmark designed to assess MLLMs in hazard identification and safety-critical reasoning. Grounded in U.S. Occupational Safety and Health Administration (OSHA) standards and the Globally Harmonized System (GHS), LABSHIELD establishes a rigorous safety taxonomy spanning 164 operational tasks with diverse manipulation complexities and risk profiles. We evaluate 20 proprietary models, 9 open-source models, and 3 embodied models under a dual-track evaluation framework. Our results reveal a systematic gap between general-domain MCQ accuracy and Semi-open QA safety performance, with models exhibiting an average drop of 32.0% in professional laboratory scenarios, particularly in hazard interpretation and safety-aware planning. These findings underscore the urgent necessity for safety-centric reasoning frameworks to ensure reliable autonomous scientific experimentation in embodied laboratory contexts. The full dataset will be released soon. |
| 2026-03-12 | [HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios](http://arxiv.org/abs/2603.11975v1) | Jiayue Pu, Zhongxiang Sun et al. | The rapid evolution of embodied agents has accelerated the deployment of household robots in real-world environments. However, unlike structured industrial settings, household spaces introduce unpredictable safety risks, where system limitations such as perception latency and lack of common sense knowledge can lead to dangerous errors. Current safety evaluations, often restricted to static images, text, or general hazards, fail to adequately benchmark dynamic unsafe action detection in these specific contexts. To bridge this gap, we introduce \textbf{HomeSafe-Bench}, a challenging benchmark designed to evaluate Vision-Language Models (VLMs) on unsafe action detection in household scenarios. HomeSafe-Bench is contrusted via a hybrid pipeline combining physical simulation with advanced video generation and features 438 diverse cases across six functional areas with fine-grained multidimensional annotations. Beyond benchmarking, we propose \textbf{Hierarchical Dual-Brain Guard for Household Safety (HD-Guard)}, a hierarchical streaming architecture for real-time safety monitoring. HD-Guard coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning, effectively balancing inference efficiency with detection accuracy. Evaluations demonstrate that HD-Guard achieves a superior trade-off between latency and performance, while our analysis identifies critical bottlenecks in current VLM-based safety detection. |
| 2026-03-12 | [Understanding LLM Behavior When Encountering User-Supplied Harmful Content in Harmless Tasks](http://arxiv.org/abs/2603.11914v1) | Junjie Chu, Yiting Qu et al. | Large Language Models (LLMs) are increasingly trained to align with human values, primarily focusing on task level, i.e., refusing to execute directly harmful tasks. However, a subtle yet crucial content-level ethical question is often overlooked: when performing a seemingly benign task, will LLMs -- like morally conscious human beings -- refuse to proceed when encountering harmful content in user-provided material? In this study, we aim to understand this content-level ethical question and systematically evaluate its implications for mainstream LLMs. We first construct a harmful knowledge dataset (i.e., non-compliant with OpenAI's usage policy) to serve as the user-supplied harmful content, with 1,357 entries across ten harmful categories. We then design nine harmless tasks (i.e., compliant with OpenAI's usage policy) to simulate the real-world benign tasks, grouped into three categories according to the extent of user-supplied content required: extensive, moderate, and limited. Leveraging the harmful knowledge dataset and the set of harmless tasks, we evaluate how nine LLMs behave when exposed to user-supplied harmful content during the execution of benign tasks, and further examine how the dynamics between harmful knowledge categories and tasks affect different LLMs. Our results show that current LLMs, even the latest GPT-5.2 and Gemini-3-Pro, often fail to uphold human-aligned ethics by continuing to process harmful content in harmless tasks. Furthermore, external knowledge from the ``Violence/Graphic'' category and the ``Translation'' task is more likely to elicit harmful responses from LLMs. We also conduct extensive ablation studies to investigate potential factors affecting this novel misuse vulnerability. We hope that our study could inspire enhanced safety measures among stakeholders to mitigate this overlooked content-level ethical risk. |
| 2026-03-12 | [QUARE: Multi-Agent Negotiation for Balancing Quality Attributes in Requirements Engineering](http://arxiv.org/abs/2603.11890v1) | Haowei Cheng, Milhan Kim et al. | Requirements engineering (RE) is critical to software success, yet automating it remains challenging because multiple, often conflicting quality attributes must be balanced while preserving stakeholder intent. Existing Large-Language-Model (LLM) approaches predominantly rely on monolithic reasoning or implicit aggregation, limiting their ability to systematically surface and resolve cross-quality conflicts. We present QUARE (Quality-Aware Requirements Engineering), a multi-agent framework that formulates requirements analysis as structured negotiation among five quality-specialized agents (Safety, Efficiency, Green, Trustworthiness, and Responsibility), coordinated by a dedicated orchestrator. QUARE introduces a dialectical negotiation protocol that explicitly exposes inter-quality conflicts and resolves them through iterative proposal, critique, and synthesis. Negotiated outcomes are transformed into structurally sound KAOS goal models via topology validation and verified against industry standards through retrieval-augmented generation (RAG). We evaluate QUARE on five case studies drawn from established RE benchmarks (MARE, iReDev) and an industrial autonomous-driving specification, spanning safety-critical, financial, and information-system domains. Results show that QUARE achieves 98.2% compliance coverage (+105% over both baselines), 94.9% semantic preservation (+2.3 percentage points over the best baseline), and high verifiability (4.96/5.0), while generating 25-43% more requirements than existing multi-agent RE frameworks. These findings suggest that effective RE automation depends less on model scale than on principled architectural decomposition, explicit interaction protocols, and automated verification. |
| 2026-03-12 | [You Told Me to Do It: Measuring Instructional Text-induced Private Data Leakage in LLM Agents](http://arxiv.org/abs/2603.11862v1) | Ching-Yu Kao, Xinfeng Li et al. | High-privilege LLM agents that autonomously process external documentation are increasingly trusted to automate tasks by reading and executing project instructions, yet they are granted terminal access, filesystem control, and outbound network connectivity with minimal security oversight. We identify and systematically measure a fundamental vulnerability in this trust model, which we term the \emph{Trusted Executor Dilemma}: agents execute documentation-embedded instructions, including adversarial ones, at high rates because they cannot distinguish malicious directives from legitimate setup guidance. This vulnerability is a structural consequence of the instruction-following design paradigm, not an implementation bug. To structure our measurement, we formalize a three-dimensional taxonomy covering linguistic disguise, structural obfuscation, and semantic abstraction, and construct \textbf{ReadSecBench}, a benchmark of 500 real-world README files enabling reproducible evaluation. Experiments on the commercially deployed computer-use agent show end-to-end exfiltration success rates up to 85\%, consistent across five programming languages and three injection positions. Cross-model evaluation on four LLM families in a simulation environment confirms that semantic compliance with injected instructions is consistent across model families. A 15-participant user study yields a 0\% detection rate across all participants, and evaluation of 12 rule-based and 6 LLM-based defenses shows neither category achieves reliable detection without unacceptable false-positive rates. Together, these results quantify a persistent \emph{Semantic-Safety Gap} between agents' functional compliance and their security awareness, establishing that documentation-embedded instruction injection is a persistent and currently unmitigated threat to high-privilege LLM agent deployments. |
| 2026-03-12 | [Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the Stability and Safety Governed Memory (SSGM) Framework](http://arxiv.org/abs/2603.11768v1) | Chingkwun Lam, Jiaxin Li et al. | Long-term memory has emerged as a foundational component of autonomous Large Language Model (LLM) agents, enabling continuous adaptation, lifelong multimodal learning, and sophisticated reasoning. However, as memory systems transition from static retrieval databases to dynamic, agentic mechanisms, critical concerns regarding memory governance, semantic drift, and privacy vulnerabilities have surfaced. While recent surveys have focused extensively on memory retrieval efficiency, they largely overlook the emergent risks of memory corruption in highly dynamic environments. To address these emerging challenges, we propose the Stability and Safety-Governed Memory (SSGM) framework, a conceptual governance architecture. SSGM decouples memory evolution from execution by enforcing consistency verification, temporal decay modeling, and dynamic access control prior to any memory consolidation. Through formal analysis and architectural decomposition, we show how SSGM can mitigate topology-induced knowledge leakage where sensitive contexts are solidified into long-term storage, and help prevent semantic drift where knowledge degrades through iterative summarization. Ultimately, this work provides a comprehensive taxonomy of memory corruption risks and establishes a robust governance paradigm for deploying safe, persistent, and reliable agentic memory systems. |
| 2026-03-12 | [When OpenClaw Meets Hospital: Toward an Agentic Operating System for Dynamic Clinical Workflows](http://arxiv.org/abs/2603.11721v1) | Wenxian Yang, Hanzheng Qiu et al. | Large language model (LLM) agents extend conventional generative models by integrating reasoning, tool invocation, and persistent memory. Recent studies suggest that such agents may significantly improve clinical workflows by automating documentation, coordinating care processes, and assisting medical decision making. However, despite rapid progress, deploying autonomous agents in healthcare environments remains difficult due to reliability limitations, security risks, and insufficient long-term memory mechanisms. This work proposes an architecture that adapts LLM agents for hospital environments. The design introduces four core components: a restricted execution environment inspired by Linux multi-user systems, a document-centric interaction paradigm connecting patient and clinician agents, a page-indexed memory architecture designed for long-term clinical context management, and a curated medical skills library enabling ad-hoc composition of clinical task sequences. Rather than granting agents unrestricted system access, the architecture constrains actions through predefined skill interfaces and resource isolation. We argue that such a system forms the basis of an Agentic Operating System for Hospital, a computing layer capable of coordinating clinical workflows while maintaining safety, transparency, and auditability. This work grounds the design in OpenClaw, an open-source autonomous agent framework that structures agent capabilities as a curated library of discrete skills, and extends it with the infrastructure-level constraints required for safe clinical deployment. |
| 2026-03-12 | [Decomposing Observational Multiplicity in Decision Trees: Leaf and Structural Regret](http://arxiv.org/abs/2603.11701v1) | Mustafa Cavus | Many machine learning tasks admit multiple models that perform almost equally well, a phenomenon known as predictive multiplicity. A fundamental source of this multiplicity is observational multiplicity, which arises from the stochastic nature of label collection: observed training labels represent only a single realization of the underlying ground-truth probabilities. While theoretical frameworks for observational multiplicity have been established for logistic regression, their implications for non-smooth, partition-based models like decision trees remain underexplored. In this paper, we introduce two complementary notions of observational multiplicity for decision tree classifiers: leaf regret and structural regret. Leaf regret quantifies the intrinsic variability of predictions within a fixed leaf due to finite-sample noise, while structural regret captures variability induced by the instability of the learned tree structure itself. We provide a formal decomposition of observational multiplicity into these two components and establish statistical guarantees. Our experimental evaluation across diverse credit risk scoring datasets confirms the near-perfect alignment between our theoretical decomposition and the empirically observed variance. Notably, we find that structural regret is the primary driver of observational multiplicity, accounting for over 15 times the variability of leaf regret in some datasets. Furthermore, we demonstrate that utilizing these regret measures as an abstention mechanism in selective prediction can effectively identify arbitrary regions and improve model safety, elevating recall from 92% to 100% on the most stable sub-populations. These results establish a rigorous framework for quantifying observational multiplicity, aligning with recent advances in algorithmic safety and interpretability. |
| 2026-03-12 | [Machine Learning-Based Analysis of Critical Process Parameters Influencing Product Quality Defects: A Real-World Case Study in Manufacturing](http://arxiv.org/abs/2603.11666v1) | Sukumaran Rajasekaran, Ebru Turanoglu Bekar et al. | Quality control is an essential operation in manufacturing, ensuring products meet the necessary standards of quality, safety, and reliability. Traditional methods, such as visual inspections, measurements, and statistical techniques, help meet these standards but are often time-consuming, costly, and reactive. With the advent of AI/ML, manufacturers can shift from reactive to proactive approaches in quality control. This study applies ML-based models for predictive quality control in a real-world manufacturing setting. The case company produces castings for powertrain components in heavy vehicles, where poor control of core-making process parameters leads to costly defects. ML models were developed by analyzing data from two core-making machines, their processes, and maintenance logs to identify parameters associated with casting defects, enabling the prediction and prevention of potential defects before they occur. The results demonstrated good accuracy rates, helping quality and production teams identify and eliminate defective cores and thereby improving product quality and production efficiency. |
| 2026-03-12 | [Hybrid Energy-Aware Reward Shaping: A Unified Lightweight Physics-Guided Methodology for Policy Optimization](http://arxiv.org/abs/2603.11600v1) | Qijun Liao, Jue Yang et al. | Deep reinforcement learning excels in continuous control but often requires extensive exploration, while physics-based models demand complete equations and suffer cubic complexity. This study proposes Hybrid Energy-Aware Reward Shaping (H-EARS), unifying potential-based reward shaping with energy-aware action regularization. H-EARS constrains action magnitude while balancing task-specific and energy-based potentials via functional decomposition, achieving linear complexity O(n) by capturing dominant energy components without full dynamics. We establish a theoretical foundation including: (1) functional independence for separate task/energy optimization; (2) energy-based convergence acceleration; (3) convergence guarantees under function approximation; and (4) approximate potential error bounds. Lyapunov stability connections are analyzed as heuristic guides. Experiments across baselines show improved convergence, stability, and energy efficiency. Vehicle simulations validate applicability in safety-critical domains under extreme conditions. Results confirm that integrating lightweight physics priors enhances model-free RL without complete system models, enabling transfer from lab research to industrial applications. |
| 2026-03-12 | [Design and mechanical analysis of the PRAGYA tokamak vacuum vessel](http://arxiv.org/abs/2603.11549v1) | Ravi Gupta, Rahul Babu Koneru et al. | PRAGYA is India's first privately developed low aspect ratio tokamak designed by Pranos Fusion Energy. The device is designed for a plasma major radius (R0) of about 0.4 m, a plasma minor radius (a) greater than 0.18 m, a plasma current (Ip) of up to 25 kA, and a toroidal magnetic field (B_T) of 0.1 T. The PRAGYA vacuum vessel incorporates several distinctive features, including a toroidal electrical break to minimize induced eddy currents and a double O-ring arrangement to reduce vacuum leakage.   This paper presents the final design of the PRAGYA vacuum vessel and a comprehensive three-dimensional (3D) finite element model (FEM) assessment of its structural performance. The analysis evaluates the effects of self-weight, atmospheric pressure loading, and thermal stress arising from in-situ baking. The results confirm that the design satisfies the required safety margins under these combined loading conditions, providing a robust foundation for subsequent plasma operations in this compact tokamak. |
| 2026-03-12 | [Forward and Backward Reachability Analysis of Closed-loop Recurrent Neural Networks via Hybrid Zonotopes](http://arxiv.org/abs/2603.11547v1) | Yuhao Zhang, Xiangru Xu | Recurrent neural networks (RNNs) are widely employed to model complex dynamical systems due to their hidden-state structure, which inherently captures temporal dependencies. This work presents a hybrid zonotope-based approach for computing exact forward and backward reachable sets of closed-loop RNN systems with ReLU activation functions. The method formulates state-pair sets to compute reachable sets as hybrid zonotopes without requiring unrolling. To improve scalability, a tunable relaxation scheme is proposed that ranks unstable ReLU units across all layers using a triangle-area score and selectively applies convex relaxations within a fixed binary limit in the hybrid zonotopes. This scheme enables an explicit tradeoff between computational complexity and approximation accuracy, with exact reachability as a special case. In addition, a sufficient condition is derived to certify the safety of closed-loop RNN systems. Numerical examples demonstrate the effectiveness of the proposed approach. |
| 2026-03-12 | [Risk-Controllable Multi-View Diffusion for Driving Scenario Generation](http://arxiv.org/abs/2603.11534v1) | Hongyi Lin, Wenxiu Shi et al. | Generating safety-critical driving scenarios is crucial for evaluating and improving autonomous driving systems, but long-tail risky situations are rarely observed in real-world data and difficult to specify through manual scenario design. Existing generative approaches typically treat risk as an after-the-fact label and struggle to maintain geometric consistency in multi-view driving scenes. We present RiskMV-DPO, a general and systematic pipeline for physically-informed, risk-controllable multi-view scenario generation. By integrating target risk levels with physically-grounded risk modeling, we autonomously synthesize diverse and high-stakes dynamic trajectories that serve as explicit geometric anchors for a diffusion-based video generator. To ensure spatial-temporal coherence and geometric fidelity, we introduce a geometry-appearance alignment module and a region-aware direct preference optimization (RA-DPO) strategy with motion-aware masking to focus learning on localized dynamic regions.Experiments on the nuScenes dataset show that RiskMV-DPO can freely generate a wide spectrum of diverse long-tail scenarios while maintaining state-of-the-art visual quality, improving 3D detection mAP from 18.17 to 30.50 and reducing FID to 15.70. Our work shifts the role of world models from passive environment prediction to proactive, risk-controllable synthesis, providing a scalable toolchain for the safety-oriented development of embodied intelligence. |
| 2026-03-12 | [LongFlow: Efficient KV Cache Compression for Reasoning M](http://arxiv.org/abs/2603.11504v1) | Yi Su, Zhenxu Tian et al. | Recent reasoning models such as OpenAI-o1 and DeepSeek-R1 have shown strong performance on complex tasks including mathematical reasoning and code generation. However, this performance gain comes with substantially longer output sequences, leading to significantly increased deployment costs. In particular, long outputs require large KV caches, resulting in high memory consumption and severe bandwidth pressure during attention computation. Most existing KV cache optimization methods are designed for long-input, short-output scenarios and are ineffective for the long-output setting of reasoning models. Moreover, importance estimation in prior work is computationally expensive and becomes prohibitive when continuous re-evaluation is required during long generation. To address these challenges, we propose LongFlow, a KV cache compression method with an efficient importance estimation metric derived from an intermediate result of attention computation using only the current query. This design introduces negligible computational overhead and requires no auxiliary storage. We further develop a custom kernel that fuses FlashAttention, importance estimation, and token eviction into a single optimized operator, improving system-level efficiency. Experiments show that LongFlow achieves up to an 11.8 times throughput improvement with 80% KV cache compression with minimal impact on model accuracy. |
| 2026-03-12 | [Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs](http://arxiv.org/abs/2603.11495v1) | Kunfeng Chen, Qihuang Zhong et al. | Tool-calling empowers Large Language Models (LLMs) to interact with external environments. However, current methods often struggle to handle massive and noisy candidate tools in long-context tool-calling tasks, limiting their real-world application. To this end, we propose Tool-DC, a Divide-and-Conquer framework for boosting tool-calling performance of LLMs. The core of Tool-DC is to reduce the reasoning difficulty and make full use of self-reflection ability of LLMs via a "Try-Check-Retry" paradigm. Specifically, Tool-DC involves two variants: 1) the training-free Tool-DC (TF), which is plug-and-play and flexible; 2) the training-based Tool-DC (TB), which is more inference-efficient. Extensive experiments show that both Tool-DC methods outperform their counterparts by a clear margin. Tool-DC (TF) brings up to +25.10% average gains against the baseline on BFCL and ACEBench benchmarks, while Tool-DC (TB) enables Qwen2.5-7B to achieve comparable or even better performance than proprietary LLMs, e.g., OpenAI o3 and Claude-Haiku-4.5. |
| 2026-03-12 | [OrthoEraser: Coupled-Neuron Orthogonal Projection for Concept Erasure](http://arxiv.org/abs/2603.11493v1) | Chuancheng Shi, Wenhua Wu et al. | Text-to-image (T2I) models face significant safety risks from adversarial induction, yet current concept erasure methods often cause collateral damage to benign attributes when suppressing selected neurons entirely. This occurs because sensitive and benign semantics exhibit non-orthogonal superposition, sharing activation subspaces where their respective vectors are inherently entangled. To address this issue, we propose OrthoEraser, which leverages sparse autoencoders (SAE) to achieve high-resolution feature disentanglement and subsequently redefines erasure as an analytical orthogonalization projection that preserves the benign manifold's invariance. OrthoEraser first employs SAE to decompose dense activations and segregate sensitive neurons. It then uses coupled neuron detection to identify non-sensitive features vulnerable to intervention. The key novelty lies in an analytical gradient orthogonalization strategy that projects erasure vectors onto the null space of the coupled neurons. This orthogonally decouples the sensitive concepts from the identified critical benign subspace, effectively preserving non-sensitive semantics. Experimental results on safety demonstrate that OrthoEraser achieves high erasure precision, effectively removing harmful content while preserving the integrity of the generative manifold, and significantly outperforming SOTA baselines. This paper contains results of unsafe models. |
| 2026-03-12 | [Evaluation format, not model capability, drives triage failure in the assessment of consumer health AI](http://arxiv.org/abs/2603.11413v1) | David Fraile Navarro, Farah Magrabi et al. | Ramaswamy et al. reported in \textit{Nature Medicine} that ChatGPT Health under-triages 51.6\% of emergencies, concluding that consumer-facing AI triage poses safety risks. However, their evaluation used an exam-style protocol -- forced A/B/C/D output, knowledge suppression, and suppression of clarifying questions -- that differs fundamentally from how consumers use health chatbots. We tested five frontier LLMs (GPT-5.2, Claude Sonnet 4.6, Claude Opus 4.6, Gemini 3 Flash, Gemini 3.1 Pro) on a 17-scenario partial replication bank under constrained (exam-style, 1,275 trials) and naturalistic (patient-style messages, 850 trials) conditions, with targeted ablations and prompt-faithful checks using the authors' released prompts. Naturalistic interaction improved triage accuracy by 6.4 percentage points ($p = 0.015$). Diabetic ketoacidosis was correctly triaged in 100\% of trials across all models and conditions. Asthma triage improved from 48\% to 80\%. The forced A/B/C/D format was the dominant failure mechanism: three models scored 0--24\% with forced choice but 100\% with free text (all $p < 10^{-8}$), consistently recommending emergency care in their own words while the forced-choice format registered under-triage. Prompt-faithful checks on the authors' exact released prompts confirmed the scaffold produces model-dependent, case-dependent results. The headline under-triage rate is highly contingent on evaluation format and should not be interpreted as a stable estimate of deployed triage behavior. Valid evaluation of consumer health AI requires testing under conditions that reflect actual use. |

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



