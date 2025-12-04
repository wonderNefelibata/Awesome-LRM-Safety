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
| 2025-12-03 | [Noise-induced stop-and-go traffic dynamics: Modelling and control](http://arxiv.org/abs/2512.04073v1) | Raphael Korbmacher, Parthib Khound et al. | This contribution investigates an original stochastic approach for the emergence of stop-and-go waves in traffic flow, a collective phenomenon with significant safety and environmental implications. Using a stable nonlinear car-following model, the study shows that minimal white Gaussian noise can destabilise the flow, leading to a phase transition from laminar to periodic dynamics through a nonlinear instability phenomenon, analogous to Kapitza's pendulum. Furthermore, a simple linear transformation of the model, which amplifies the response and introduces a positive acceleration bias, counteracts noise-induced effects and recovers the stability of uniform solutions. The findings are supported by simulations, offering new insights into the modelling and mitigation of oscillatory traffic dynamics. |
| 2025-12-03 | [SkillFactory: Self-Distillation For Learning Cognitive Behaviors](http://arxiv.org/abs/2512.04072v1) | Zayne Sprague, Jack Lu et al. | Reasoning models leveraging long chains of thought employ various cognitive skills, such as verification of their answers, backtracking, retrying by an alternate method, and more. Previous work has shown that when a base language model exhibits these skills, training that model further with reinforcement learning (RL) can learn to leverage them. How can we get models to leverage skills that aren't exhibited by base models? Our work, SkillFactory, is a method for fine-tuning models to roughly learn these skills during a supervised fine-tuning (SFT) stage prior to RL. Our approach does not rely on distillation from a stronger model, but instead uses samples from the model itself, rearranged to provide training data in the format of those skills. These "silver" SFT traces may be imperfect, but are nevertheless effective for priming a model to acquire skills during RL. Our evaluation shows that (1) starting from SkillFactory SFT initialization helps a model to generalize to harder variants of a task post-RL, despite lower performance pre-RL; (2) cognitive skills are indeed used by the model; (3) RLed SkillFactory models are more robust to regression on out-of-domain tasks than RLed base models. Our work suggests that inductive biases learned prior to RL help models learn robust cognitive skill use. |
| 2025-12-03 | [Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](http://arxiv.org/abs/2512.03994v1) | Oren Rachmil, Roy Betser et al. | Aligning proprietary large language models (LLMs) with internal organizational policies has become an urgent priority as organizations increasingly deploy LLMs in sensitive domains such as legal support, finance, and medical services. Beyond generic safety filters, enterprises require reliable mechanisms to detect policy violations within their regulatory and operational frameworks, where breaches can trigger legal and reputational risks. Existing content moderation frameworks, such as guardrails, remain largely confined to the safety domain and lack the robustness to capture nuanced organizational policies. LLM-as-a-judge and fine-tuning approaches, though flexible, introduce significant latency and lack interpretability. To address these limitations, we propose a training-free and efficient method that treats policy violation detection as an out-of-distribution (OOD) detection problem. Inspired by whitening techniques, we apply a linear transformation to decorrelate the model's hidden activations and standardize them to zero mean and unit variance, yielding near-identity covariance matrix. In this transformed space, we use the Euclidean norm as a compliance score to detect policy violations. The method requires only the policy text and a small number of illustrative samples, which makes it light-weight and easily deployable. On a challenging policy benchmark, our approach achieves state-of-the-art results, surpassing both existing guardrails and fine-tuned reasoning models. This work provides organizations with a practical and statistically grounded framework for policy-aware oversight of LLMs, advancing the broader goal of deployable AI governance. Code is available at: https://tinyurl.com/policy-violation-detection |
| 2025-12-03 | [DIQ-H: Evaluating Hallucination Persistence in VLMs Under Temporal Visual Degradation](http://arxiv.org/abs/2512.03992v1) | Zexin Lin, Hawen Wan et al. | Vision-Language Models (VLMs) deployed in safety-critical applications such as autonomous driving must handle continuous visual streams under imperfect conditions. However, existing benchmarks focus on static, high-quality images and ignore temporal degradation and error propagation, which are critical failure modes where transient visual corruption induces hallucinations that persist across subsequent frames. We introduce DIQ-H, the first benchmark for evaluating VLM robustness under dynamic visual degradation in temporal sequences. DIQ-H applies physics-based corruptions including motion blur, sensor noise, and compression artifacts, and measures hallucination persistence, error recovery, and temporal consistency through multi-turn question-answering tasks. To enable scalable annotation, we propose Uncertainty-Guided Iterative Refinement (UIR), which generates reliable pseudo-ground-truth using lightweight VLMs with uncertainty filtering, achieving a 15.3 percent accuracy improvement. Experiments on 16 state-of-the-art VLMs reveal substantial robustness gaps: even advanced models such as GPT-4o achieve only a 78.5 percent recovery rate, while open-source models struggle with temporal consistency at less than 60 percent. DIQ-H provides a comprehensive platform for evaluating VLM reliability in real-world deployments. |
| 2025-12-03 | [Digital Twin-based Control Co-Design of Full Vehicle Active Suspensions via Deep Reinforcement Learning](http://arxiv.org/abs/2512.03891v1) | Ying-Kuan Tsai, Yi-Ping Chen et al. | Active suspension systems are critical for enhancing vehicle comfort, safety, and stability, yet their performance is often limited by fixed hardware designs and control strategies that cannot adapt to uncertain and dynamic operating conditions. Recent advances in digital twins (DTs) and deep reinforcement learning (DRL) offer new opportunities for real-time, data-driven optimization across a vehicle's lifecycle. However, integrating these technologies into a unified framework remains an open challenge. This work presents a DT-based control co-design (CCD) framework for full-vehicle active suspensions using multi-generation design concepts. By integrating automatic differentiation into DRL, we jointly optimize physical suspension components and control policies under varying driver behaviors and environmental uncertainties. DRL also addresses the challenge of partial observability, where only limited states can be sensed and fed back to the controller, by learning optimal control actions directly from available sensor information. The framework incorporates model updating with quantile learning to capture data uncertainty, enabling real-time decision-making and adaptive learning from digital-physical interactions. The approach demonstrates personalized optimization of suspension systems under two distinct driving settings (mild and aggressive). Results show that the optimized systems achieve smoother trajectories and reduce control efforts by approximately 43% and 52% for mild and aggressive, respectively, while maintaining ride comfort and stability. Contributions include: developing a DT-enabled CCD framework integrating DRL and uncertainty-aware model updating for full-vehicle active suspensions, introducing a multi-generation design strategy for self-improving systems, and demonstrating personalized optimization of active suspension systems for distinct driver types. |
| 2025-12-03 | [MPCFormer: A physics-informed data-driven approach for explainable socially-aware autonomous driving](http://arxiv.org/abs/2512.03795v1) | Jia Hu, Zhexi Lian et al. | Autonomous Driving (AD) vehicles still struggle to exhibit human-like behavior in highly dynamic and interactive traffic scenarios. The key challenge lies in AD's limited ability to interact with surrounding vehicles, largely due to a lack of understanding the underlying mechanisms of social interaction. To address this issue, we introduce MPCFormer, an explainable socially-aware autonomous driving approach with physics-informed and data-driven coupled social interaction dynamics. In this model, the dynamics are formulated into a discrete space-state representation, which embeds physics priors to enhance modeling explainability. The dynamics coefficients are learned from naturalistic driving data via a Transformer-based encoder-decoder architecture. To the best of our knowledge, MPCFormer is the first approach to explicitly model the dynamics of multi-vehicle social interactions. The learned social interaction dynamics enable the planner to generate manifold, human-like behaviors when interacting with surrounding traffic. By leveraging the MPC framework, the approach mitigates the potential safety risks typically associated with purely learning-based methods. Open-looped evaluation on NGSIM dataset demonstrates that MPCFormer achieves superior social interaction awareness, yielding the lowest trajectory prediction errors compared with other state-of-the-art approach. The prediction achieves an ADE as low as 0.86 m over a long prediction horizon of 5 seconds. Close-looped experiments in highly intense interaction scenarios, where consecutive lane changes are required to exit an off-ramp, further validate the effectiveness of MPCFormer. Results show that MPCFormer achieves the highest planning success rate of 94.67%, improves driving efficiency by 15.75%, and reduces the collision rate from 21.25% to 0.5%, outperforming a frontier Reinforcement Learning (RL) based planner. |
| 2025-12-03 | [Poly- and single-crystalline diamond nitrogen-induced TLS losses estimation with superconducting lumped elements micro-resonators](http://arxiv.org/abs/2512.03780v1) | Francesco Mazzocchi, Martin Neidig et al. | Research on diamond has intensified due to its exceptional thermal, optical, and mechanical properties, making it a key material in quantum technologies and high-power applications. Diamonds with engineered nitrogen-vacancy (NV) centers represent a very sensitive platform for quantum sensing, while high-optical quality diamond windows represent a fundamental safety component inside Electron Cyclotron Resonance Heating (ECRH) systems in nuclear fusion reactors. A major challenge is the development of ultra-low-loss, high-optical-quality single-crystal diamond substrates to meet growing demands for quantum coherence and power handling. Traditionally, dielectric losses ($\tan δ$) in diamonds are evaluated using Fabry-Perot microwave resonators, in which the resonance quality factors Q of the cavity with and without the sample are compared. These devices are limited to resolutions around 10$^{-5}$ by the need to keep the resonator dimensions within a reasonable range. In contrast, superconducting thin-film micro-strip resonators, with Q factors exceeding 10$^6$, are stated to provide higher sensitivity for assessing ultra-low-loss materials. This study examines four diamond samples grown through different processes, analyzing their dielectric losses at extreme low temperatures (sub-Kelvin) within the Two-Level System (TLS) framework. Complementary Raman spectroscopy measurements allowed us not only to associate higher nitrogen content with increased losses, but also to investigate how the different growth process influence the way these defects are incorporated in the crystal lattice. |
| 2025-12-03 | [Safety Reinforced Model Predictive Control (SRMPC): Improving MPC with Reinforcement Learning for Motion Planning in Autonomous Driving](http://arxiv.org/abs/2512.03774v1) | Johannes Fischer, Marlon Steiner et al. | Model predictive control (MPC) is widely used for motion planning, particularly in autonomous driving. Real-time capability of the planner requires utilizing convex approximation of optimal control problems (OCPs) for the planner. However, such approximations confine the solution to a subspace, which might not contain the global optimum. To address this, we propose using safe reinforcement learning (SRL) to obtain a new and safe reference trajectory within MPC. By employing a learning-based approach, the MPC can explore solutions beyond the close neighborhood of the previous one, potentially finding global optima. We incorporate constrained reinforcement learning (CRL) to ensure safety in automated driving, using a handcrafted energy function-based safety index as the constraint objective to model safe and unsafe regions. Our approach utilizes a state-dependent Lagrangian multiplier, learned concurrently with the safe policy, to solve the CRL problem. Through experimentation in a highway scenario, we demonstrate the superiority of our approach over both MPC and SRL in terms of safety and performance measures. |
| 2025-12-03 | [In-Context Representation Hijacking](http://arxiv.org/abs/2512.03771v1) | Itay Yona, Amir Sarid et al. | We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level. |
| 2025-12-03 | [Sample-Efficient Counterfactual Tuning for Compressor Pressure Control](http://arxiv.org/abs/2512.03747v1) | Margarita A. Guerrero, Rodrigo A. González et al. | In controlled industrial environments, ensuring safety and performance during controller tuning is a challenging and critical task. In particular, control loops in compressor-plenum-throttle systems cannot tolerate costly interruptions, and aggressive excitation may lead to unsafe operating regimes. Given the wide availability of historical data, this paper introduces a counterfactual explainability approach for sample-efficient retuning of compressor control loops. The proposed data-driven algorithm determines, without an explicit plant model or previous control law, the smallest controller adjustment required to achieve predefined performance specifications while guaranteeing stability. The effectiveness of the method is demonstrated through an extensive Monte Carlo simulation study. |
| 2025-12-03 | [AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation](http://arxiv.org/abs/2512.03737v1) | Chuyue Wang, Jie Feng et al. | Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering powerful semantic understanding to bridge this gap. Despite their potential, deploying LLMs in this high-stakes domain is fraught with challenges, including factual hallucinations, specialized knowledge gaps, and high operational costs. To overcome these barriers, we introduce \textbf{AR-Med}, a novel framework for \textbf{A}utomated \textbf{R}elevance assessment for \textbf{Med}ical search that has been successfully deployed at scale on the Online Medical Delivery Platforms. AR-Med grounds LLM reasoning in verified medical knowledge through a retrieval-augmented approach, ensuring high accuracy and reliability. To enable efficient online service, we design a practical knowledge distillation scheme that compresses large teacher models into compact yet powerful student models. We also introduce LocalQSMed, a multi-expert annotated benchmark developed to guide model iteration and ensure strong alignment between offline and online performance. Extensive experiments show AR-Med achieves an offline accuracy of over 93\%, a 24\% absolute improvement over the original online system, and delivers significant gains in online relevance and user satisfaction. Our work presents a practical and scalable blueprint for developing trustworthy, LLM-powered systems in real-world healthcare applications. |
| 2025-12-03 | [ContactRL: Safe Reinforcement Learning based Motion Planning for Contact based Human Robot Collaboration](http://arxiv.org/abs/2512.03707v1) | Sundas Rafat Mulkana, Ronyu Yu et al. | In collaborative human-robot tasks, safety requires not only avoiding collisions but also ensuring safe, intentional physical contact. We present ContactRL, a reinforcement learning (RL) based framework that directly incorporates contact safety into the reward function through force feedback. This enables a robot to learn adaptive motion profiles that minimize human-robot contact forces while maintaining task efficiency. In simulation, ContactRL achieves a low safety violation rate of 0.2\% with a high task success rate of 87.7\%, outperforming state-of-the-art constrained RL baselines. In order to guarantee deployment safety, we augment the learned policy with a kinetic energy based Control Barrier Function (eCBF) shield. Real-world experiments on an UR3e robotic platform performing small object handovers from a human hand across 360 trials confirm safe contact, with measured normal forces consistently below 10N. These results demonstrate that ContactRL enables safe and efficient physical collaboration, thereby advancing the deployment of collaborative robots in contact-rich tasks. |
| 2025-12-03 | [Context-Triggered Contingency Games for Strategic Multi-Agent Interaction](http://arxiv.org/abs/2512.03639v1) | Kilian Schweppe, Anne-Kathrin Schmuck | We address the challenge of reliable and efficient interaction in autonomous multi-agent systems, where agents must balance long-term strategic objectives with short-term dynamic adaptation. We propose context-triggered contingency games, a novel integration of strategic games derived from temporal logic specifications with dynamic contingency games solved in real time. Our two-layered architecture leverages strategy templates to guarantee satisfaction of high-level objectives, while a new factor-graph-based solver enables scalable, real-time model predictive control of dynamic interactions. The resulting framework ensures both safety and progress in uncertain, interactive environments. We validate our approach through simulations and hardware experiments in autonomous driving and robotic navigation, demonstrating efficient, reliable, and adaptive multi-agent interaction. |
| 2025-12-03 | [Physics-Based Communication Compression via Lyapunov-Weighted Event-Triggered Control](http://arxiv.org/abs/2512.03604v1) | Abbas Tariverdi | Event-Triggered Control (ETC) reduces communication overhead in networked systems by transmitting only when stability requires it. Conventional mechanisms use isotropic error thresholds ($\|e\| \le σ\|x\|$), treating all directions equally. This ignores stability geometry and triggers conservatively. We propose a static directional triggering mechanism that exploits this asymmetry. By weighting errors via the Lyapunov matrix $P$, we define an anisotropic half-space scaling with instantaneous energy margins: larger deviations tolerated along stable modes, strict bounds where instability threatens. We prove global asymptotic stability and exclusion of Zeno behavior. Monte Carlo simulations ($N=100$) show 43.6\% fewer events than optimally tuned isotropic methods while achieving $2.1\times$ better control performance than time-varying alternatives. The mechanism functions as a runtime safety gate for learning-based controllers operating under communication constraints. |
| 2025-12-03 | [Towards Irreversible Machine Unlearning for Diffusion Models](http://arxiv.org/abs/2512.03564v1) | Xun Yuan, Zilong Zhao et al. | Diffusion models are renowned for their state-of-the-art performance in generating synthetic images. However, concerns related to safety, privacy, and copyright highlight the need for machine unlearning, which can make diffusion models forget specific training data and prevent the generation of sensitive or unwanted content. Current machine unlearning methods for diffusion models are primarily designed for conditional diffusion models and focus on unlearning specific data classes or features. Among these methods, finetuning-based machine unlearning methods are recognized for their efficiency and effectiveness, which update the parameters of pre-trained diffusion models by minimizing carefully designed loss functions. However, in this paper, we propose a novel attack named Diffusion Model Relearning Attack (DiMRA), which can reverse the finetuning-based machine unlearning methods, posing a significant vulnerability of this kind of technique. Without prior knowledge of the unlearning elements, DiMRA optimizes the unlearned diffusion model on an auxiliary dataset to reverse the unlearning, enabling the model to regenerate previously unlearned elements. To mitigate this vulnerability, we propose a novel machine unlearning method for diffusion models, termed as Diffusion Model Unlearning by Memorization (DiMUM). Unlike traditional methods that focus on forgetting, DiMUM memorizes alternative data or features to replace targeted unlearning data or features in order to prevent generating such elements. In our experiments, we demonstrate the effectiveness of DiMRA in reversing state-of-the-art finetuning-based machine unlearning methods for diffusion models, highlighting the need for more robust solutions. We extensively evaluate DiMUM, demonstrating its superior ability to preserve the generative performance of diffusion models while enhancing robustness against DiMRA. |
| 2025-12-03 | [Resilient AFE Drive Control using Neural Networks with Tracking Guarantees](http://arxiv.org/abs/2512.03545v1) | Nicolas Kirsch, Catalin Arghir et al. | Industrial installations across several sectors have seen a dramatic increase in productivity, accuracy and efficiency over the last decade due to expanded utilization of medium voltage, variable speed power electronic converters to drive their processes. Specifically, active front-end (AFE) drives have become popular due to their ability to deliver power while maintaining safe electrical setpoints. However, under abnormal grid conditions such as phase loss, conventional AFE control may fail to enforce safety constraints, potentially leading to drive shutdown and significant financial losses. In this work, we propose using reference-tracking Performance Boosting (rPB) to improve the resilience of standard AFE control to faults. This neural-network control framework provides a principled way to optimize transient performance while preserving the steady-state tracking properties of AFE-based drives. By carefully shaping the input signals to the rPB controller, we ensure that it activates only during grid faults, leaving nominal operation unaffected. Simulation results show that the proposed approach successfully maintains the DC bus voltage and the grid current within safe limits during single-phase loss events. |
| 2025-12-03 | [Think Before You Drive: World Model-Inspired Multimodal Grounding for Autonomous Vehicles](http://arxiv.org/abs/2512.03454v1) | Haicheng Liao, Huanming Shen et al. | Interpreting natural-language commands to localize target objects is critical for autonomous driving (AD). Existing visual grounding (VG) methods for autonomous vehicles (AVs) typically struggle with ambiguous, context-dependent instructions, as they lack reasoning over 3D spatial relations and anticipated scene evolution. Grounded in the principles of world models, we propose ThinkDeeper, a framework that reasons about future spatial states before making grounding decisions. At its core is a Spatial-Aware World Model (SA-WM) that learns to reason ahead by distilling the current scene into a command-aware latent state and rolling out a sequence of future latent states, providing forward-looking cues for disambiguation. Complementing this, a hypergraph-guided decoder then hierarchically fuses these states with the multimodal input, capturing higher-order spatial dependencies for robust localization. In addition, we present DrivePilot, a multi-source VG dataset in AD, featuring semantic annotations generated by a Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT)-prompted LLM pipeline. Extensive evaluations on six benchmarks, ThinkDeeper ranks #1 on the Talk2Car leaderboard and surpasses state-of-the-art baselines on DrivePilot, MoCAD, and RefCOCO/+/g benchmarks. Notably, it shows strong robustness and efficiency in challenging scenes (long-text, multi-agent, ambiguity) and retains superior performance even when trained on 50% of the data. |
| 2025-12-03 | [Exploring the Potential and Limitations of Large Language Models for Novice Program Fault Localization](http://arxiv.org/abs/2512.03421v1) | Hexiang Xu, Hengyuan Liu et al. | Novice programmers often face challenges in fault localization due to their limited experience and understanding of programming syntax and logic. Traditional methods like Spectrum-Based Fault Localization (SBFL) and Mutation-Based Fault Localization (MBFL) help identify faults but often lack the ability to understand code context, making them less effective for beginners. In recent years, Large Language Models (LLMs) have shown promise in overcoming these limitations by utilizing their ability to understand program syntax and semantics. LLM-based fault localization provides more accurate and context-aware results than traditional techniques. This study evaluates six closed-source and seven open-source LLMs using the Codeflaws, Condefects, and BugT datasets, with BugT being a newly constructed dataset specifically designed to mitigate data leakage concerns. Advanced models with reasoning capabilities, such as OpenAI o3 and DeepSeekR1, achieve superior accuracy with minimal reliance on prompt engineering. In contrast, models without reasoning capabilities, like GPT-4, require carefully designed prompts to maintain performance. While LLMs perform well in simple fault localization, their accuracy decreases as problem difficulty increases, though top models maintain robust performance in the BugT dataset. Over-reasoning is another challenge, where some models generate excessive explanations that hinder fault localization clarity. Additionally, the computational cost of deploying LLMs remains a significant barrier for real-time debugging. LLM's explanations demonstrate significant value for novice programmer assistance, with one-year experience participants consistently rating them highly. Our findings demonstrate the potential of LLMs to improve debugging efficiency while stressing the need for further refinement in their reasoning and computational efficiency for practical adoption. |
| 2025-12-03 | [Immunity memory-based jailbreak detection: multi-agent adaptive guard for large language models](http://arxiv.org/abs/2512.03356v1) | Jun Leng, Litian Zhang et al. | Large language models (LLMs) have become foundational in AI systems, yet they remain vulnerable to adversarial jailbreak attacks. These attacks involve carefully crafted prompts that bypass safety guardrails and induce models to produce harmful content. Detecting such malicious input queries is therefore critical for maintaining LLM safety. Existing methods for jailbreak detection typically involve fine-tuning LLMs as static safety LLMs using fixed training datasets. However, these methods incur substantial computational costs when updating model parameters to improve robustness, especially in the face of novel jailbreak attacks. Inspired by immunological memory mechanisms, we propose the Multi-Agent Adaptive Guard (MAAG) framework for jailbreak detection. The core idea is to equip guard with memory capabilities: upon encountering novel jailbreak attacks, the system memorizes attack patterns, enabling it to rapidly and accurately identify similar threats in future encounters. Specifically, MAAG first extracts activation values from input prompts and compares them to historical activations stored in a memory bank for quick preliminary detection. A defense agent then simulates responses based on these detection results, and an auxiliary agent supervises the simulation process to provide secondary filtering of the detection outcomes. Extensive experiments across five open-source models demonstrate that MAAG significantly outperforms state-of-the-art (SOTA) methods, achieving 98% detection accuracy and a 96% F1-score across a diverse range of attack scenarios. |
| 2025-12-03 | [HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration](http://arxiv.org/abs/2512.03345v1) | Seunghoi Kim, Henry F. J. Tregidgo et al. | Generative models are prone to hallucinations: plausible but incorrect structures absent in the ground truth. This issue is problematic in image restoration for safety-critical domains such as medical imaging, industrial inspection, and remote sensing, where such errors undermine reliability and trust. For example, in low-field MRI, widely used in resource-limited settings, restoration models are essential for enhancing low-quality scans, yet hallucinations can lead to serious diagnostic errors. Progress has been hindered by a circular dependency: evaluating hallucinations requires labeled data, yet such labels are costly and subjective. We introduce HalluGen, a diffusion-based framework that synthesizes realistic hallucinations with controllable type, location, and severity, producing perceptually realistic but semantically incorrect outputs (segmentation IoU drops from 0.86 to 0.36). Using HalluGen, we construct the first large-scale hallucination dataset comprising 4,350 annotated images derived from 1,450 brain MR images for low-field enhancement, enabling systematic evaluation of hallucination detection and mitigation. We demonstrate its utility in two applications: (1) benchmarking image quality metrics and developing Semantic Hallucination Assessment via Feature Evaluation (SHAFE), a feature-based metric with soft-attention pooling that improves hallucination sensitivity over traditional metrics; and (2) training reference-free hallucination detectors that generalize to real restoration failures. Together, HalluGen and its open dataset establish the first scalable foundation for evaluating hallucinations in safety-critical image restoration. |

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



