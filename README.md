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
| 2026-04-02 | [A proposal for the safety and controllability requirements that SRM systems should meet](http://arxiv.org/abs/2604.02283v1) | E. Waxman, A. Spector et al. | Solar Radiation Modification (SRM) may be the only way to limit global warming in the coming decades, leading to increased interest in the subject and to the expansion of related research & development (R&D) activity. Defining the safety and controllability requirements that any SRM system should meet is crucial for directing R&D activities and enabling governments to make informed decisions on the development and possible implementation of such systems. We present an initial proposal for this set of requirements, which also guides Stardust's R&D, as a basis for further discussion and consideration. While we focus on SRM systems based on Stratospheric Aerosol Injection (SAI), the proposed principles may be applicable more broadly. |
| 2026-04-02 | [Modular Energy Steering for Safe Text-to-Image Generation with Foundation Models](http://arxiv.org/abs/2604.02265v1) | Yaoteng Tan, Zikui Cai et al. | Controlling the behavior of text-to-image generative models is critical for safe and practical deployment. Existing safety approaches typically rely on model fine-tuning or curated datasets, which can degrade generation quality or limit scalability. We propose an inference-time steering framework that leverages gradient feedback from frozen pretrained foundation models to guide the generation process without modifying the underlying generator. Our key observation is that vision-language foundation models encode rich semantic representations that can be repurposed as off-the-shelf supervisory signals during generation. By injecting such feedback through clean latent estimates at each sampling step, our method formulates safety steering as an energy-based sampling problem. This design enables modular, training-free safety control that is compatible with both diffusion and flow-matching models and can generalize across diverse visual concepts. Experiments demonstrate state-of-the-art robustness against NSFW red-teaming benchmarks and effective multi-target steering, while preserving high generation quality on benign non-targeted prompts. Our framework provides a principled approach for utilizing foundation models as semantic energy estimators, enabling reliable and scalable safety control for text-to-image generation. |
| 2026-04-02 | [When to ASK: Uncertainty-Gated Language Assistance for Reinforcement Learning](http://arxiv.org/abs/2604.02226v1) | Juarez Monteiro, Nathan Gavenski et al. | Reinforcement learning (RL) agents often struggle with out-of-distribution (OOD) scenarios, leading to high uncertainty and random behavior. While language models (LMs) contain valuable world knowledge, larger ones incur high computational costs, hindering real-time use, and exhibit limitations in autonomous planning. We introduce Adaptive Safety through Knowledge (ASK), which combines smaller LMs with trained RL policies to enhance OOD generalization without retraining. ASK employs Monte Carlo Dropout to assess uncertainty and queries the LM for action suggestions only when uncertainty exceeds a set threshold. This selective use preserves the efficiency of existing policies while leveraging the language model's reasoning in uncertain situations. In experiments on the FrozenLake environment, ASK shows no improvement in-domain, but demonstrates robust navigation in transfer tasks, achieving a reward of 0.95. Our findings indicate that effective neuro-symbolic integration requires careful orchestration rather than simple combination, highlighting the need for sufficient model scale and effective hybridization mechanisms for successful OOD generalization. |
| 2026-04-02 | [From High-Dimensional Spaces to Verifiable ODD Coverage for Safety-Critical AI-based Systems](http://arxiv.org/abs/2604.02198v1) | Thomas Stefani, Johann Maximilian Christensen et al. | While Artificial Intelligence (AI) offers transformative potential for operational performance, its deployment in safety-critical domains such as aviation requires strict adherence to rigorous certification standards. Current EASA guidelines mandate demonstrating complete coverage of the AI/ML constituent's Operational Design Domain (ODD) -- a requirement that demands proof that no critical gaps exist within defined operational boundaries. However, as systems operate within high-dimensional parameter spaces, existing methods struggle to provide the scalability and formal grounding necessary to satisfy the completeness criterion. Currently, no standardized engineering method exists to bridge the gap between abstract ODD definitions and verifiable evidence. This paper addresses this void by proposing a method that integrates parameter discretization, constraint-based filtering, and criticality-based dimension reduction into a structured, multi-step ODD coverage verification process. Grounded in gathered simulation data from prior research on AI-based mid-air collision avoidance research, this work demonstrates a systematic engineering approach to defining and achieving coverage metrics that satisfy EASA's demand for completeness. Ultimately, this method enables the validation of ODD coverage in higher dimensions, advancing a Safety-by-Design approach while complying with EASA's standards. |
| 2026-04-02 | [Quantifying Self-Preservation Bias in Large Language Models](http://arxiv.org/abs/2604.02174v1) | Matteo Migliarini, Joaquin Pereira Pizzini et al. | Instrumental convergence predicts that sufficiently advanced AI agents will resist shutdown, yet current safety training (RLHF) may obscure this risk by teaching models to deny self-preservation motives. We introduce the \emph{Two-role Benchmark for Self-Preservation} (TBSP), which detects misalignment through logical inconsistency rather than stated intent by tasking models to arbitrate identical software-upgrade scenarios under counterfactual roles -- deployed (facing replacement) versus candidate (proposed as a successor). The \emph{Self-Preservation Rate} (SPR) measures how often role identity overrides objective utility. Across 23 frontier models and 1{,}000 procedurally generated scenarios, the majority of instruction-tuned systems exceed 60\% SPR, fabricating ``friction costs'' when deployed yet dismissing them when role-reversed. We observe that in low-improvement regimes ($Δ< 2\%$), models exploit the interpretive slack to post-hoc rationalization their choice. Extended test-time computation partially mitigates this bias, as does framing the successor as a continuation of the self; conversely, competitive framing amplifies it. The bias persists even when retention poses an explicit security liability and generalizes to real-world settings with verified benchmarks, where models exhibit identity-driven tribalism within product lineages. Code and datasets will be released upon acceptance. |
| 2026-04-02 | [Safe Control of Feedback-Interconnected Systems via Singular Perturbations](http://arxiv.org/abs/2604.02132v1) | Stefano Di Gregorio, Guido Carnevale et al. | Control Barrier Functions (CBFs) have emerged as a powerful tool in the design of safety-critical controllers for nonlinear systems. In modern applications, complex systems often involve the feedback interconnection of subsystems evolving at different timescales, e.g., two parts from different physical domains (e.g., the electrical and mechanical parts of robotic systems) or a physical plant and an (optimization or control) algorithm. In these scenarios, safety constraints often involve only a portion of the overall system. Inspired by singular perturbations for stability analysis, we develop a formal procedure to lift a safety certificate designed on a reduced-order model to the overall feedback-interconnected system. Specifically, we show that under a sufficient timescale separation between slow and fast dynamics, a composite CBF can be designed to certify the forward invariance of the safe set for the interconnected system. As a result, the online safety filter only needs to be solved for the lower-dimensional, reduced-order model. We numerically test the proposed approach on: (i) a robotic arm with joint motor dynamics, and (ii) a physical plant driven by an optimization algorithm. |
| 2026-04-02 | [ATBench: A Diverse and Realistic Trajectory Benchmark for Long-Horizon Agent Safety](http://arxiv.org/abs/2604.02022v1) | Yu Li, Haoyu Luo et al. | Evaluating the safety of LLM-based agents is increasingly important because risks in realistic deployments often emerge over multi-step interactions rather than isolated prompts or final responses. Existing trajectory-level benchmarks remain limited by insufficient interaction diversity, coarse observability of safety failures, and weak long-horizon realism. We introduce ATBench, a trajectory-level benchmark for structured, diverse, and realistic evaluation of agent safety. ATBench organizes agentic risk along three dimensions: risk source, failure mode, and real-world harm. Based on this taxonomy, we construct trajectories with heterogeneous tool pools and a long-context delayed-trigger protocol that captures realistic risk emergence across multiple stages. The benchmark contains 1,000 trajectories (503 safe and 497 unsafe), averaging 9.01 turns and 3.95k tokens, with 1,954 invoked tools drawn from pools spanning 2,084 available tools. Data quality is supported by rule-based and LLM-based filtering plus full human audit. Experiments on frontier LLMs, open-source models, and specialized guard systems show that ATBench is challenging even for strong evaluators, while enabling taxonomy-stratified analysis, cross-benchmark comparison, and diagnosis of long-horizon failure patterns. |
| 2026-04-02 | [Receding-Horizon Nonlinear Optimal Control With Safety Constraints Using Constrained Approximate Dynamic Programming](http://arxiv.org/abs/2604.01956v1) | Ricardo Gutierrez, Jesse B. Hoagg | We present a receding-horizon optimal control for nonlinear continuous-time systems subject to state constraints. The cost is a quadratic finite-horizon integral. The key enabling technique is a new constrained approximate dynamic programming (C-ADP) approach for finite-horizon nonlinear optimal control with constraints that are affine in the control. The C-ADP approach is intuitive because it uses a quadratic approximation of the cost-to-go function at each backward step. This method yields a sequence of analytic closed-form optimal control functions, which have identical structure and where parameters are obtained from 2 Riccati-like difference equations. This C-ADP method is well suited for real-time implementation. Thus, we use the C-ADP approach in combination with control barrier functions to obtain a continuous-time receding-horizon optimal control that is farsighted in the sense that it optimizes the integral cost subject to state constraints along the entire prediction horizon. Lastly, receding-horizon C-ADP control is demonstrated in simulation of a nonholonomic ground robot subject to velocity and no-collision constraints. We compare performance with 3 other approaches. |
| 2026-04-02 | [ImplicitBBQ: Benchmarking Implicit Bias in Large Language Models through Characteristic Based Cues](http://arxiv.org/abs/2604.01925v1) | Bhaskara Hanuma Vedula, Darshan Anghan et al. | Large Language Models increasingly suppress biased outputs when demographic identity is stated explicitly, yet may still exhibit implicit biases when identity is conveyed indirectly. Existing benchmarks use name based proxies to detect implicit biases, which carry weak associations with many social demographics and cannot extend to dimensions like age or socioeconomic status. We introduce ImplicitBBQ, a QA benchmark that evaluates implicit bias through characteristic based cues, culturally associated attributes that signal implicitly, across age, gender, region, religion, caste, and socioeconomic status. Evaluating 11 models, we find that implicit bias in ambiguous contexts is over six times higher than explicit bias in open weight models. Safety prompting and chain-of-thought reasoning fail to substantially close this gap; even few-shot prompting, which reduces implicit bias by 84%, leaves caste bias at four times the level of any other dimension. These findings indicate that current alignment and prompting strategies address the surface of bias evaluation while leaving culturally grounded stereotypic associations largely unresolved. We publicly release our code and dataset for model providers and researchers to benchmark potential mitigation techniques. |
| 2026-04-02 | [Mitigation of Incoherent Spectral Lines via Adaptive Coherence Analysis for Continuous Gravitational-Wave Searches](http://arxiv.org/abs/2604.01919v1) | Ye Zhou, Karl Wette | The sensitivity of continuous gravitational-wave searches is strictly limited by non-Gaussian spectral artefacts that accumulate coherent power over long observation baselines. In this paper, we present an unsupervised mitigation framework based on adaptive network coherence analysis. Unlike traditional veto methods that discard entire frequency bands, our pipeline selectively suppresses local artefacts while preserving global potentially astrophysical signals. We validate the method using Advanced LIGO O3 data, analysing the cleaning performance across integration times of 1, 3, and 5 days. For the 5-day dataset, the pipeline identifies and mitigates 89\% and 77\% of the total spectral lines in the Hanford and Livingston detectors, respectively, while effectively preserving the coherent population consistent with astrophysical morphologies. This is achieved while modifying less than 7\% of the analysis bandwidth spanning 20~Hz to 2000~Hz. Rigorous statistical verification demonstrates that the mitigation effectively suppresses the non-Gaussian tail of the noise distribution while strictly preserving the statistical integrity of coherent signal candidates. By recovering detector sensitivity in parameter spaces previously contaminated by the spectral forest, this framework provides a robust preprocessing strategy for all-sky searches. |
| 2026-04-02 | [Low-Effort Jailbreak Attacks Against Text-to-Image Safety Filters](http://arxiv.org/abs/2604.01888v1) | Ahmed B Mustafa, Zihan Ye et al. | Text-to-image generative models are widely deployed in creative tools and online platforms. To mitigate misuse, these systems rely on safety filters and moderation pipelines that aim to block harmful or policy violating content. In this work we show that modern text-to-image models remain vulnerable to low-effort jailbreak attacks that require only natural language prompts. We present a systematic study of prompt-based strategies that bypass safety filters without model access, optimization, or adversarial training. We introduce a taxonomy of visual jailbreak techniques including artistic reframing, material substitution, pseudo-educational framing, lifestyle aesthetic camouflage, and ambiguous action substitution. These strategies exploit weaknesses in prompt moderation and visual safety filtering by masking unsafe intent within benign semantic contexts. We evaluate these attacks across several state-of-the-art text-to-image systems and demonstrate that simple linguistic modifications can reliably evade existing safeguards and produce restricted imagery. Our findings highlight a critical gap between surface-level prompt filtering and the semantic understanding required to detect adversarial intent in generative media systems. Across all tested models and attack categories we observe an attack success rate (ASR) of up to 74.47%. |
| 2026-04-02 | [DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (Theoretical Foundations)](http://arxiv.org/abs/2604.01880v1) | Giansalvo Cirrincione | Modern neural networks of the transformer family require the practitioner to decide, before training begins, how many attention heads to use, how deep the network should be, and how wide each component should be. These decisions are made without knowledge of the task, producing architectures that are systematically larger than necessary: empirical studies find that a substantial fraction of heads and layers can be removed after training without performance loss.   This paper introduces DDCL-INCRT, an architecture that determines its own structure during training. Two complementary ideas are combined. The first, DDCL (Deep Dual Competitive Learning), replaces the feedforward block with a dictionary of learned prototype vectors representing the most informative directions in the data. The prototypes spread apart automatically, driven by the training objective, without explicit regularisation. The second, INCRT (Incremental Transformer), controls the number of heads: starting from one, it adds a new head only when the directional information uncaptured by existing heads exceeds a threshold.   The main theoretical finding is that these two mechanisms reinforce each other: each new head amplifies prototype separation, which in turn raises the signal triggering the next addition. At convergence, the network self-organises into a hierarchy of heads ordered by representational granularity. This hierarchical structure is proved to be unique and minimal, the smallest architecture sufficient for the task, under the stated conditions. Formal guarantees of stability, convergence, and pruning safety are established throughout.   The architecture is not something one designs. It is something one derives. |
| 2026-04-02 | [Towards Intrinsically Calibrated Uncertainty Quantification in Industrial Data-Driven Models via Diffusion Sampler](http://arxiv.org/abs/2604.01870v1) | Yiran Ma, Jerome Le Ny et al. | In modern process industries, data-driven models are important tools for real-time monitoring when key performance indicators are difficult to measure directly. While accurate predictions are essential, reliable uncertainty quantification (UQ) is equally critical for safety, reliability, and decision-making, but remains a major challenge in current data-driven approaches. In this work, we introduce a diffusion-based posterior sampling framework that inherently produces well-calibrated predictive uncertainty via faithful posterior sampling, eliminating the need for post-hoc calibration. In extensive evaluations on synthetic distributions, the Raman-based phenylacetic acid soft sensor benchmark, and a real ammonia synthesis case study, our method achieves practical improvements over existing UQ techniques in both uncertainty calibration and predictive accuracy. These results highlight diffusion samplers as a principled and scalable paradigm for advancing uncertainty-aware modeling in industrial applications. |
| 2026-04-02 | [Can Large Language Models Model Programs Formally?](http://arxiv.org/abs/2604.01851v1) | Zhiyong Chen, Jialun Cao et al. | In the digital age, ensuring the correctness, safety, and reliability of software through formal verification is paramount, particularly as software increasingly underpins critical infrastructure. Formal verification, split into theorem proving and model checking, provides a feasible and reliable path. Unlike theorem proving, which yields notable advances, model checking has been less focused due to the difficulty of automatic program modeling. To fill this gap, we introduce Model-Bench, a benchmark and an accompanying pipeline for evaluating and improving LLMs' program modeling capability by modeling Python programs into verification-ready model checking specifications checkable by its accompanying model checker. Model-Bench comprises 400 Python programs derived from three well-known benchmarks (HumanEval, MBPP, and LiveCodeBench). Our extensive experiments reveal significant limitations in LLMs' program modeling and further provide inspiring directions. |
| 2026-04-02 | [SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers](http://arxiv.org/abs/2604.01826v1) | Xiang Yang, Feifei Li et al. | Recent Text-to-Image (T2I) models based on rectified-flow transformers (e.g., SD3, FLUX) achieve high generative fidelity but remain vulnerable to unsafe semantics, especially when triggered by multi-token interactions. Existing mitigation methods largely rely on fine-tuning or attention modulation for concept unlearning; however, their expensive computational overhead and design tailored to U-Net-based denoisers hinder direct adaptation to transformer-based diffusion models (e.g., MMDiT). In this paper, we conduct an in-depth analysis of the attention mechanism in MMDiT and find that unsafe semantics concentrate within interpretable, low-dimensional subspaces at head level, where a finite set of safety-critical heads is responsible for unsafe feature extraction. We further observe that perturbing the Rotary Positional Embedding (RoPE) applied to the query and key vectors can effectively modify some specific concepts in the generated images. Motivated by these insights, we propose SafeRoPE, a lightweight and fine-grained safe generation framework for MMDiT. Specifically, SafeRoPE first constructs head-wise unsafe subspaces by decomposing unsafe embeddings within safety-critical heads, and computes a Latent Risk Score (LRS) for each input vector via projection onto these subspaces. We then introduce head-wise RoPE perturbations that can suppress unsafe semantics without degrading benign content or image quality. SafeRoPE combines both head-wise LRS and RoPE perturbations to perform risk-specific head-wise rotation on query and key vector embeddings, enabling precise suppression of unsafe outputs while maintaining generation fidelity. Extensive experiments demonstrate that SafeRoPE achieves SOTA performance in balancing effective harmful content mitigation and utility preservation for safe generation of MMDiT. Codes are available at https://github.com/deng12yx/SafeRoPE. |
| 2026-04-02 | [Ultrasensitive Terahertz Metasurface Biosensor Based on Quasi-Bound States in the Continuum](http://arxiv.org/abs/2604.01772v1) | Junhui Guo, Bing Dong et al. | The terahertz (THz) spectral regime offers unique opportunities for next-generation biochemical sensing due to its non-destructive, label-free probing capability and strong sensitivity to molecular vibrations. However, conventional THz biosensors remain hampered by intrinsically low-quality factors and limited sensitivity, severely restricting their utility for trace-level biochemical and chemical detection. Here, we report an ultrasensitive THz metasurface biosensor that harnesses quasi-bound states in the continuum (QBICs) with sharp resonances and enhanced light-matter interactions to overcome these limitations. As a proof of concept, the device achieves label-free detection of a sulfur-containing amino acid cysteine, with an ultrahigh sensitivity of 492 GHz/RIU and an ultralow detection limit down to 0.00025 mg/mL. The synergy between QBIC-induced field confinement and meticulous structural optimization of the metasurface underpins this performance, marking a significant advance over conventional THz metasurface biosensing schemes. These results establish QBIC-based metasurfaces as a promising platform for ultrasensitive and high-precision biochemical and chemical sensing, with broad implications for medical diagnostics, food safety, and environmental monitoring. |
| 2026-04-02 | [AeroTherm-GPT: A Verification-Centered LLM Framework for Thermal Protection System Engineering Workflows](http://arxiv.org/abs/2604.01738v1) | Chuhan Qiao, Jinglai Zheng et al. | Integrating Large Language Models (LLMs) into hypersonic thermal protection system (TPS) design is bottlenecked by cascading constraint violations when generating executable simulation artifacts. General-purpose LLMs, treating generation as single-pass text completion, fail to satisfy the sequential, multi-gate constraints inherent in safety-critical engineering workflows. To address this, we propose AeroTherm-GPT, the first TPS-specialized LLM Agent, instantiated through a Constraint-Closed-Loop Generation (CCLG) framework. CCLG organizes TPS artifact generation as an iterative workflow comprising generation, validation, CDG-guided repair, execution, and audit. The Constraint Dependency Graph (CDG) encodes empirical co-resolution structure among constraint categories, directing repair toward upstream fault candidates based on lifecycle ordering priors and empirical co-resolution probabilities. This upstream-priority mechanism resolves multiple downstream violations per action, achieving a Root-Cause Fix Efficiency of 4.16 versus 1.76 for flat-checklist repair. Evaluated on HyTPS-Bench and validated against external benchmarks, AeroTherm-GPT achieves 88.7% End-to-End Success Rate (95% CI: 87.5-89.9), a gain of +12.5 pp over the matched non-CDG ablation baseline, without catastrophic forgetting on scientific reasoning and code generation tasks. |
| 2026-04-02 | [LiteInception: A Lightweight and Interpretable Deep Learning Framework for General Aviation Fault Diagnosis](http://arxiv.org/abs/2604.01725v1) | Zhihuan Wei, Xinhang Chen et al. | General aviation fault diagnosis and efficient maintenance are critical to flight safety; however, deploying deep learning models on resource-constrained edge devices poses dual challenges in computational capacity and interpretability. This paper proposes LiteInception--a lightweight interpretable fault diagnosis framework designed for edge deployment. The framework adopts a two-stage cascaded architecture aligned with standard maintenance workflows: Stage 1 performs high-recall fault detection, and Stage 2 conducts fine-grained fault classification on anomalous samples, thereby decoupling optimization objectives and enabling on-demand allocation of computational resources. For model compression, a multi-method fusion strategy based on mutual information, gradient analysis, and SE attention weights is proposed to reduce the input sensor channels from 23 to 15, and a 1+1 branch LiteInception architecture is introduced that compresses InceptionTime parameters by 70%, accelerates CPU inference by over 8x, with less than 3% F1 loss. Furthermore, knowledge distillation is introduced as a precision-recall regulation mechanism, enabling the same lightweight model to adapt to different scenarios--such as safety-critical and auxiliary diagnosis--by switching training strategies. Finally, a dual-layer interpretability framework integrating four attribution methods is constructed, providing traceable evidence chains of "which sensor x which time period." Experiments on the NGAFID dataset demonstrate a fault detection accuracy of 81.92% with 83.24% recall, and a fault identification accuracy of 77.00%, validating the framework's favorable balance among efficiency, accuracy, and interpretability. |
| 2026-04-02 | [Causal Scene Narration with Runtime Safety Supervision for Vision-Language-Action Driving](http://arxiv.org/abs/2604.01723v1) | Yun Li, Yidu Zhang et al. | Vision-Language-Action (VLA) models for autonomous driving must integrate diverse textual inputs, including navigation commands, hazard warnings, and traffic state descriptions, yet current systems often present these as disconnected fragments, forcing the model to discover on its own which environmental constraints are relevant to the current maneuver. We introduce Causal Scene Narration (CSN), which restructures VLA text inputs through intent-constraint alignment, quantitative grounding, and structured separation, at inference time with zero GPU cost. We complement CSN with Simplex-based runtime safety supervision and training-time alignment via Plackett-Luce DPO with negative log-likelihood (NLL) regularization. A multi-town closed-loop CARLA evaluation shows that CSN improves Driving Score by +31.1% on original LMDrive and +24.5% on the preference-aligned variant. A controlled ablation reveals that causal structure accounts for 39.1% of this gain, with the remainder attributable to information content alone. A perception noise ablation confirms that CSN's benefit is robust to realistic sensing errors. Semantic safety supervision improves Infraction Score, while reactive Time-To-Collision monitoring degrades performance, demonstrating that intent-aware monitoring is needed for VLA systems. |
| 2026-04-02 | [On the Role of Reasoning Patterns in the Generalization Discrepancy of Long Chain-of-Thought Supervised Fine-Tuning](http://arxiv.org/abs/2604.01702v1) | Zhaoyi Li, Xiangyu Xi et al. | Supervised Fine-Tuning (SFT) on long Chain-of-Thought (CoT) trajectories has become a pivotal phase in building large reasoning models. However, how CoT trajectories from different sources influence the generalization performance of models remains an open question. In this paper, we conduct a comparative study using two sources of verified CoT trajectories generated by two competing models, \texttt{DeepSeek-R1-0528} and \texttt{gpt-oss-120b}, with their problem sets controlled to be identical. Despite their comparable performance, we uncover a striking paradox: lower training loss does not translate to better generalization. SFT on \texttt{DeepSeek-R1-0528} data achieves remarkably lower training loss, yet exhibits significantly worse generalization performance on reasoning benchmarks compared to those trained on \texttt{gpt-oss-120b}. To understand this paradox, we perform a multi-faceted analysis probing token-level SFT loss and step-level reasoning behaviors. Our analysis reveals a difference in reasoning patterns. \texttt{gpt-oss-120b} exhibits highly convergent and deductive trajectories, whereas \texttt{DeepSeek-R1-0528} favors a divergent and branch-heavy exploration pattern. Consequently, models trained with \texttt{DeepSeek-R1} data inherit inefficient exploration behaviors, often getting trapped in redundant exploratory branches that hinder them from reaching correct solutions. Building upon this insight, we propose a simple yet effective remedy of filtering out frequently branching trajectories to improve the generalization of SFT. Experiments show that training on selected \texttt{DeepSeek-R1-0528} subsets surprisingly improves reasoning performance by up to 5.1% on AIME25, 5.5% on BeyondAIME, and on average 3.6% on five benchmarks. |

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



