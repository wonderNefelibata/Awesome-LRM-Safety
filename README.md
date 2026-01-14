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
| 2026-01-13 | [MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm](http://arxiv.org/abs/2601.08800v1) | Bowen Zhou, Jinrui Jia et al. | The Mixture of Experts (MoE) models are emerging as the latest paradigm for Large Language Models (LLMs). However, due to memory constraints, MoE models with billions or even trillions of parameters can only be deployed in multi-GPU or even multi-node & multi-GPU based serving systems. Thus, communication has became a major bottleneck in distributed serving systems, especially inter-node communication. Contemporary distributed MoE models are primarily implemented using all-reduce (AR) based tensor parallelism (TP) and all-to-all (A2A) based expert parallelism (EP). However, TP generally exhibits low inter-node efficiency and is thus confined to high-speed intra-node bandwidth. In contrast, EP tends to suffer from load imbalance, especially when the parallel degree is high.   In this work, we introduce MixServe, a novel automatic distributed serving system for efficient deployment of MoE models by a novel TP-EP hybrid parallelism based on fused AR-A2A communication algorithm. MixServe begins by evaluating the communication overhead associated with various parallel strategies, taking into account the model hyperparameters and the configurations of network and hardware resources, and then automatically selects the most efficient parallel strategy. Then, we propose the TP-EP hybrid parallelism based on fused AR-A2A communication algorithm that overlaps intra-node AR communication and inter-node A2A communication. Extensive experiments on DeepSeek-R1 and Qwen3 models demonstrate that MixServe achieves superior inference performance, with 1.08~3.80x acceleration in time to first token (TTFT), 1.03~1.66x acceleration in inter-token latency (ITL), and 5.2%~50.3% throughput improvement compared to existing approaches. |
| 2026-01-13 | [TerraFormer: Automated Infrastructure-as-Code with LLMs Fine-Tuned via Policy-Guided Verifier Feedback](http://arxiv.org/abs/2601.08734v1) | Prithwish Jana, Sam Davidson et al. | Automating Infrastructure-as-Code (IaC) is challenging, and large language models (LLMs) often produce incorrect configurations from natural language (NL). We present TerraFormer, a neuro-symbolic framework for IaC generation and mutation that combines supervised fine-tuning with verifier-guided reinforcement learning, using formal verification tools to provide feedback on syntax, deployability, and policy compliance. We curate two large, high-quality NL-to-IaC datasets, TF-Gen (152k instances) and TF-Mutn (52k instances), via multi-stage verification and iterative LLM self-correction. Evaluations against 17 state-of-the-art LLMs, including ~50x larger models like Sonnet 3.7, DeepSeek-R1, and GPT-4.1, show that TerraFormer improves correctness over its base LLM by 15.94% on IaC-Eval, 11.65% on TF-Gen (Test), and 19.60% on TF-Mutn (Test). It outperforms larger models on both TF-Gen (Test) and TF-Mutn (Test), ranks third on IaC-Eval, and achieves top best-practices and security compliance. |
| 2026-01-13 | [Analyzing Bias in False Refusal Behavior of Large Language Models for Hate Speech Detoxification](http://arxiv.org/abs/2601.08668v1) | Kyuri Im, Shuzhou Yuan et al. | While large language models (LLMs) have increasingly been applied to hate speech detoxification, the prompts often trigger safety alerts, causing LLMs to refuse the task. In this study, we systematically investigate false refusal behavior in hate speech detoxification and analyze the contextual and linguistic biases that trigger such refusals. We evaluate nine LLMs on both English and multilingual datasets, our results show that LLMs disproportionately refuse inputs with higher semantic toxicity and those targeting specific groups, particularly nationality, religion, and political ideology. Although multilingual datasets exhibit lower overall false refusal rates than English datasets, models still display systematic, language-dependent biases toward certain targets. Based on these findings, we propose a simple cross-translation strategy, translating English hate speech into Chinese for detoxification and back, which substantially reduces false refusals while preserving the original content, providing an effective and lightweight mitigation approach. |
| 2026-01-13 | [Provably Safe Reinforcement Learning using Entropy Regularizer](http://arxiv.org/abs/2601.08646v1) | Abhijit Mazumdar, Rafal Wisniewski et al. | We consider the problem of learning the optimal policy for Markov decision processes with safety constraints. We formulate the problem in a reach-avoid setup. Our goal is to design online reinforcement learning algorithms that ensure safety constraints with arbitrarily high probability during the learning phase. To this end, we first propose an algorithm based on the optimism in the face of uncertainty (OFU) principle. Based on the first algorithm, we propose our main algorithm, which utilizes entropy regularization. We investigate the finite-sample analysis of both algorithms and derive their regret bounds. We demonstrate that the inclusion of entropy regularization improves the regret and drastically controls the episode-to-episode variability that is inherent in OFU-based safe RL algorithms. |
| 2026-01-13 | [SafeRedir: Prompt Embedding Redirection for Robust Unlearning in Image Generation Models](http://arxiv.org/abs/2601.08623v1) | Renyang Liu, Kangjie Chen et al. | Image generation models (IGMs), while capable of producing impressive and creative content, often memorize a wide range of undesirable concepts from their training data, leading to the reproduction of unsafe content such as NSFW imagery and copyrighted artistic styles. Such behaviors pose persistent safety and compliance risks in real-world deployments and cannot be reliably mitigated by post-hoc filtering, owing to the limited robustness of such mechanisms and a lack of fine-grained semantic control. Recent unlearning methods seek to erase harmful concepts at the model level, which exhibit the limitations of requiring costly retraining, degrading the quality of benign generations, or failing to withstand prompt paraphrasing and adversarial attacks. To address these challenges, we introduce SafeRedir, a lightweight inference-time framework for robust unlearning via prompt embedding redirection. Without modifying the underlying IGMs, SafeRedir adaptively routes unsafe prompts toward safe semantic regions through token-level interventions in the embedding space. The framework comprises two core components: a latent-aware multi-modal safety classifier for identifying unsafe generation trajectories, and a token-level delta generator for precise semantic redirection, equipped with auxiliary predictors for token masking and adaptive scaling to localize and regulate the intervention. Empirical results across multiple representative unlearning tasks demonstrate that SafeRedir achieves effective unlearning capability, high semantic and perceptual preservation, robust image quality, and enhanced resistance to adversarial attacks. Furthermore, SafeRedir generalizes effectively across a variety of diffusion backbones and existing unlearned models, validating its plug-and-play compatibility and broad applicability. Code and data are available at https://github.com/ryliu68/SafeRedir. |
| 2026-01-13 | [Coverage-Guided Road Selection and Prioritization for Efficient Testing in Autonomous Driving Systems](http://arxiv.org/abs/2601.08609v1) | Qurban Ali, Andrea Stocco et al. | Autonomous Driving Assistance Systems (ADAS) rely on extensive testing to ensure safety and reliability, yet road scenario datasets often contain redundant cases that slow down the testing process without improving fault detection. To address this issue, we present a novel test prioritization framework that reduces redundancy while preserving geometric and behavioral diversity. Road scenarios are clustered based on geometric and dynamic features of the ADAS driving behavior, from which representative cases are selected to guarantee coverage. Roads are finally prioritized based on geometric complexity, driving difficulty, and historical failures, ensuring that the most critical and challenging tests are executed first. We evaluate our framework on the OPENCAT dataset and the Udacity self-driving car simulator using two ADAS models. On average, our approach achieves an 89% reduction in test suite size while retaining an average of 79% of failed road scenarios. The prioritization strategy improves early failure detection by up to 95x compared to random baselines. |
| 2026-01-13 | [Formalization and Implementation of Safe Destination Passing in Pure Functional Programming Settings](http://arxiv.org/abs/2601.08529v1) | Thomas Bagrel | Destination-passing style programming introduces destinations, which represent the address of a write-once memory cell. These destinations can be passed as function parameters, allowing the caller to control memory management: the callee simply fills the cell instead of allocating space for a return value. While typically used in systems programming, destination passing also has applications in pure functional programming, where it enables programs that were previously unexpressible using usual immutable data structures.   In this thesis, we develop a core λ-calculus with destinations, {λ_d}. Our new calculus is more expressive than similar existing systems, with destination passing designed to be as flexible as possible. This is achieved through a modal type system combining linear types with a system of ages to manage scopes, in order to make destination-passing safe. Type safety of our core calculus was proved formally with the Coq proof assistant.   Then, we see how this core calculus can be adapted into an existing pure functional language, Haskell, whose type system is less powerful than our custom theoretical one. Retaining safety comes at the cost of removing some flexibility in the handling of destinations. We later refine the implementation to recover much of this flexibility, at the cost of increased user complexity.   The prototype implementation in Haskell shows encouraging results for adopting destination-passing style programming when traversing or mapping over large data structures such as lists or data trees. |
| 2026-01-13 | [QP-Based Control of an Underactuated Aerial Manipulator under Constraints](http://arxiv.org/abs/2601.08523v1) | Nesserine Laribi, Mohammed Rida Mokhtari et al. | This paper presents a constraint-aware control framework for underactuated aerial manipulators, enabling accurate end-effector trajectory tracking while explicitly accounting for safety and feasibility constraints. The control problem is formulated as a quadratic program that computes dynamically consistent generalized accelerations subject to underactuation, actuator bounds, and system constraints. To enhance robustness against disturbances, modeling uncertainties, and steady-state errors, a passivity-based integral action is incorporated at the torque level without compromising feasibility. The effectiveness of the proposed approach is demonstrated through high-fidelity physics-based simulations, which include parameter perturbations, viscous joint friction, and realistic sensing and state-estimation effects. This demonstrates accurate tracking, smooth control inputs, and reliable constraint satisfaction under realistic operating conditions. |
| 2026-01-13 | [Design and construction of a cryogenic subcooler-box for supplying single phase supercritical helium to dark matter and gravitational wave experiments](http://arxiv.org/abs/2601.08496v1) | Udai Raj Singh, Rajinikumar Ramalingam et al. | We report on the design, development, and installation of the ALPS Cryo-Platform Subcooler Box (ACPS), which is part of the cryogenic platform being established in the HERA North Hall at DESY to supply helium for cooling large-scale dark-matter and gravitational-wave experiments with very high heat loads. The ACPS is capable of subcooling supercritical helium supplied via the 1.6-km-long HERA transfer line by means of a pipe heat exchanger immersed in a subcooler bath filled with liquid helium produced through Joule-Thomson valves. It is also equipped with numerous cryogenic components, including control valves, flow meters, and safety valves, enabling experimental operation to be carried out directly by the ACPS itself and thereby reducing the cryogenic requirements imposed on the experiments. To support a wide range of experiments, the ACPS provides three transfer lines that deliver different levels of cooling power. |
| 2026-01-13 | [On Deciding Constant Runtime of Linear Loops](http://arxiv.org/abs/2601.08492v1) | Florian Frohn, Jürgen Giesl et al. | We consider linear single-path loops of the form \[   \textbf{while} \quad \varphi \quad \textbf{do} \quad \vec{x} \gets A \vec{x} + \vec{b} \quad \textbf{end} \] where $\vec{x}$ is a vector of variables, the loop guard $\varphi$ is a conjunction of linear inequations over the variables $\vec{x}$, and the update of the loop is represented by the matrix $A$ and the vector $\vec{b}$. It is already known that termination of such loops is decidable. In this work, we consider loops where $A$ has real eigenvalues, and prove that it is decidable whether the loop's runtime (for all inputs) is bounded by a constant if the variables range over $\mathbb R$ or $\mathbb Q$. This is an important problem in automatic program verification, since safety of linear while-programs is decidable if all loops have constant runtime, and it is closely connected to the existence of multiphase-linear ranking functions, which are often used for termination and complexity analysis. To evaluate its practical applicability, we also present an implementation of our decision procedure. |
| 2026-01-13 | [BenchOverflow: Measuring Overflow in Large Language Models via Plain-Text Prompts](http://arxiv.org/abs/2601.08490v1) | Erin Feiglin, Nir Hutnik et al. | We investigate a failure mode of large language models (LLMs) in which plain-text prompts elicit excessive outputs, a phenomenon we term Overflow. Unlike jailbreaks or prompt injection, Overflow arises under ordinary interaction settings and can lead to elevated serving cost, latency, and cross-user performance degradation, particularly when scaled across many requests. Beyond usability, the stakes are economic and environmental: unnecessary tokens increase per-request cost and energy consumption, compounding into substantial operational spend and carbon footprint at scale. Moreover, Overflow represents a practical vector for compute amplification and service degradation in shared environments. We introduce BenchOverflow, a model-agnostic benchmark of nine plain-text prompting strategies that amplify output volume without adversarial suffixes or policy circumvention. Using a standardized protocol with a fixed budget of 5000 new tokens, we evaluate nine open- and closed-source models and observe pronounced rightward shifts and heavy tails in length distributions. Cap-saturation rates (CSR@1k/3k/5k) and empirical cumulative distribution functions (ECDFs) quantify tail risk; within-prompt variance and cross-model correlations show that Overflow is broadly reproducible yet heterogeneous across families and attack vectors. A lightweight mitigation-a fixed conciseness reminder-attenuates right tails and lowers CSR for all strategies across the majority of models. Our findings position length control as a measurable reliability, cost, and sustainability concern rather than a stylistic quirk. By enabling standardized comparison of length-control robustness across models, BenchOverflow provides a practical basis for selecting deployments that minimize resource waste and operating expense, and for evaluating defenses that curb compute amplification without eroding task performance. |
| 2026-01-13 | [Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided Spectral Cleaning](http://arxiv.org/abs/2601.08489v1) | Tony Cristofano | Safety-aligned language models systematically refuse harmful requests. While activation steering can modulate refusal, ablating the raw "refusal vector" calculated from contrastive harmful and harmless prompts often causes collateral damage and distribution drift. We argue this degradation occurs because the raw vector is polysemantic, entangling the refusal signal with core capability circuits and linguistic style.   We introduce Surgical Refusal Ablation (SRA) to distill these steering directions. SRA constructs a registry of independent Concept Atoms representing protected capabilities and stylistic confounds, then uses ridge-regularized spectral residualization to orthogonalize the refusal vector against these directions. This yields a clean refusal direction that targets refusal-relevant structure while minimizing disruption to the model's semantic geometry.   Across five models (Qwen3-VL and Ministral series), SRA achieves deep refusal reduction (0-2%) with negligible perplexity impact on Wikitext-2 (mean delta PPL approx. 0.02) and minimal distribution drift. Notably, standard ablation on Qwen3-VL-4B induces severe drift (first-token KL = 2.088), whereas SRA maintains the original distribution (KL = 0.044) while achieving the same 0% refusal rate. Using teacher-forced perplexity on GSM8K and MBPP as a high-resolution capability proxy, we show SRA preserves math and code distributions. These results suggest that common "model damage" is often "Ghost Noise," defined as the spectral bleeding of the dirty refusal direction into capability subspaces. |
| 2026-01-13 | [Zero-Shot Distracted Driver Detection via Vision Language Models with Double Decoupling](http://arxiv.org/abs/2601.08467v1) | Takamichi Miyata, Sumiko Miyata et al. | Distracted driving is a major cause of traffic collisions, calling for robust and scalable detection methods. Vision-language models (VLMs) enable strong zero-shot image classification, but existing VLM-based distracted driver detectors often underperform in real-world conditions. We identify subject-specific appearance variations (e.g., clothing, age, and gender) as a key bottleneck: VLMs entangle these factors with behavior cues, leading to decisions driven by who the driver is rather than what the driver is doing. To address this, we propose a subject decoupling framework that extracts a driver appearance embedding and removes its influence from the image embedding prior to zero-shot classification, thereby emphasizing distraction-relevant evidence. We further orthogonalize text embeddings via metric projection onto Stiefel manifold to improve separability while staying close to the original semantics. Experiments demonstrate consistent gains over prior baselines, indicating the promise of our approach for practical road-safety applications. |
| 2026-01-13 | [Current and temperature imbalances in parallel-connected grid storage battery modules](http://arxiv.org/abs/2601.08459v1) | Joseph Ross, Damien Frost et al. | A key challenge with large battery systems is heterogeneous currents and temperatures in modules with parallel-connected cells. Although extreme currents and temperatures are detrimental to the performance and lifetime of battery cells, there is not a consensus on the scale of typical imbalances within grid storage modules. Here, we quantify these imbalances through simulations and experiments on an industrially representative grid storage battery module consisting of prismatic lithium iron phosphate cells, elucidating the evolution of current and temperature imbalances and their dependence on individual cell and module parameter variations. Using a sensitivity analysis, we find that varying contact resistances and cell resistances contribute strongly to temperature differences between cells, from which we define safety thresholds on cell-to-cell variability. Finally, we investigate how these thresholds change for different applications, to outline a set of robustness metrics that show how cycling at lower C-rates and narrower SOC ranges can mitigate failures. |
| 2026-01-13 | [YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation](http://arxiv.org/abs/2601.08441v1) | Abdelaziz Bounhar, Rania Hossam Elmohamady Elbadry et al. | Steering Large Language Models (LLMs) through activation interventions has emerged as a lightweight alternative to fine-tuning for alignment and personalization. Recent work on Bi-directional Preference Optimization (BiPO) shows that dense steering vectors can be learned directly from preference data in a Direct Preference Optimization (DPO) fashion, enabling control over truthfulness, hallucinations, and safety behaviors. However, dense steering vectors often entangle multiple latent factors due to neuron multi-semanticity, limiting their effectiveness and stability in fine-grained settings such as cultural alignment, where closely related values and behaviors (e.g., among Middle Eastern cultures) must be distinguished. In this paper, we propose Yet another Policy Optimization (YaPO), a \textit{reference-free} method that learns \textit{sparse steering vectors} in the latent space of a Sparse Autoencoder (SAE). By optimizing sparse codes, YaPO produces disentangled, interpretable, and efficient steering directions. Empirically, we show that YaPO converges faster, achieves stronger performance, and exhibits improved training stability compared to dense steering baselines. Beyond cultural alignment, YaPO generalizes to a range of alignment-related behaviors, including hallucination, wealth-seeking, jailbreak, and power-seeking. Importantly, YaPO preserves general knowledge, with no measurable degradation on MMLU. Overall, our results show that YaPO provides a general recipe for efficient, stable, and fine-grained alignment of LLMs, with broad applications to controllability and domain adaptation. The associated code and data are publicly available\footnote{https://github.com/MBZUAI-Paris/YaPO}. |
| 2026-01-13 | [Bio-RV: Low-Power Resource-Efficient RISC-V Processor for Biomedical Applications](http://arxiv.org/abs/2601.08428v1) | Vijay Pratap Sharma, Annu Kumar et al. | This work presents Bio-RV, a compact and resource-efficient RISC-V processor intended for biomedical control applications, such as accelerator-based biomedical SoCs and implantable pacemaker systems. The proposed Bio-RV is a multi-cycle RV32I core that provides explicit execution control and external instruction loading with capabilities that enable controlled firmware deployment, ASIC bring-up, and post-silicon testing. In addition to coordinating accelerator configuration and data transmission in heterogeneous systems, Bio-RV is designed to function as a lightweight host controller, handling interfaces with pacing, sensing, electrogram (EGM), telemetry, and battery management modules. With 708 LUTs and 235 flip-flops on FPGA prototypes, Bio-RV, implemented in a 180 nm CMOS technology, operate at 50 MHz and feature a compact hardware footprint. According to post-layout results, the proposed architectural decisions align with minimal energy use. Ultimately, Bio-RV prioritises deterministic execution, minimal hardware complexity, and integration flexibility over peak computing speed to meet the demands of ultra-low-power, safety-critical biomedical systems. |
| 2026-01-13 | [Semantic Misalignment in Vision-Language Models under Perceptual Degradation](http://arxiv.org/abs/2601.08355v1) | Guo Cheng | Vision-Language Models (VLMs) are increasingly deployed in autonomous driving and embodied AI systems, where reliable perception is critical for safe semantic reasoning and decision-making. While recent VLMs demonstrate strong performance on multimodal benchmarks, their robustness to realistic perception degradation remains poorly understood. In this work, we systematically study semantic misalignment in VLMs under controlled degradation of upstream visual perception, using semantic segmentation on the Cityscapes dataset as a representative perception module. We introduce perception-realistic corruptions that induce only moderate drops in conventional segmentation metrics, yet observe severe failures in downstream VLM behavior, including hallucinated object mentions, omission of safety-critical entities, and inconsistent safety judgments. To quantify these effects, we propose a set of language-level misalignment metrics that capture hallucination, critical omission, and safety misinterpretation, and analyze their relationship with segmentation quality across multiple contrastive and generative VLMs. Our results reveal a clear disconnect between pixel-level robustness and multimodal semantic reliability, highlighting a critical limitation of current VLM-based systems and motivating the need for evaluation frameworks that explicitly account for perception uncertainty in safety-critical applications. |
| 2026-01-13 | [Detecting Mental Manipulation in Speech via Synthetic Multi-Speaker Dialogue](http://arxiv.org/abs/2601.08342v1) | Run Chen, Wen Liang et al. | Mental manipulation, the strategic use of language to covertly influence or exploit others, is a newly emerging task in computational social reasoning. Prior work has focused exclusively on textual conversations, overlooking how manipulative tactics manifest in speech. We present the first study of mental manipulation detection in spoken dialogues, introducing a synthetic multi-speaker benchmark SPEECHMENTALMANIP that augments a text-based dataset with high-quality, voice-consistent Text-to-Speech rendered audio. Using few-shot large audio-language models and human annotation, we evaluate how modality affects detection accuracy and perception. Our results reveal that models exhibit high specificity but markedly lower recall on speech compared to text, suggesting sensitivity to missing acoustic or prosodic cues in training. Human raters show similar uncertainty in the audio setting, underscoring the inherent ambiguity of manipulative speech. Together, these findings highlight the need for modality-aware evaluation and safety alignment in multimodal dialogue systems. |
| 2026-01-13 | [Safe Heterogeneous Multi-Agent RL with Communication Regularization for Coordinated Target Acquisition](http://arxiv.org/abs/2601.08327v1) | Gabriele Calzolari, Vidya Sumathy et al. | This paper introduces a decentralized multi-agent reinforcement learning framework enabling structurally heterogeneous teams of agents to jointly discover and acquire randomly located targets in environments characterized by partial observability, communication constraints, and dynamic interactions. Each agent's policy is trained with the Multi-Agent Proximal Policy Optimization algorithm and employs a Graph Attention Network encoder that integrates simulated range-sensing data with communication embeddings exchanged among neighboring agents, enabling context-aware decision-making from both local sensing and relational information. In particular, this work introduces a unified framework that integrates graph-based communication and trajectory-aware safety through safety filters. The architecture is supported by a structured reward formulation designed to encourage effective target discovery and acquisition, collision avoidance, and de-correlation between the agents' communication vectors by promoting informational orthogonality. The effectiveness of the proposed reward function is demonstrated through a comprehensive ablation study. Moreover, simulation results demonstrate safe and stable task execution, confirming the framework's effectiveness. |
| 2026-01-13 | [YOLOBirDrone: Dataset for Bird vs Drone Detection and Classification and a YOLO based enhanced learning architecture](http://arxiv.org/abs/2601.08319v1) | Dapinder Kaur, Neeraj Battish et al. | The use of aerial drones for commercial and defense applications has benefited in many ways and is therefore utilized in several different application domains. However, they are also increasingly used for targeted attacks, posing a significant safety challenge and necessitating the development of drone detection systems. Vision-based drone detection systems currently have an accuracy limitation and struggle to distinguish between drones and birds, particularly when the birds are small in size. This research work proposes a novel YOLOBirDrone architecture that improves the detection and classification accuracy of birds and drones. YOLOBirDrone has different components, including an adaptive and extended layer aggregation (AELAN), a multi-scale progressive dual attention module (MPDA), and a reverse MPDA (RMPDA) to preserve shape information and enrich features with local and global spatial and channel information. A large-scale dataset, BirDrone, is also introduced in this article, which includes small and challenging objects for robust aerial object identification. Experimental results demonstrate an improvement in performance metrics through the proposed YOLOBirDrone architecture compared to other state-of-the-art algorithms, with detection accuracy reaching approximately 85% across various scenarios. |

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



