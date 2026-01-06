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
| 2026-01-05 | [Multi-Fidelity Predictive Model for Shock Response of Energetic Materials Using Conditional U-Net](http://arxiv.org/abs/2601.02327v1) | Brian H. Lee, Chunyu Li et al. | Mapping microstructure to properties is central to materials science. Perhaps most famously, the Hall-Petch relationship relates average grain size to strength. More challenging has been deriving relationships for properties that depend on subtle microstructural features and not average properties. One such example is the initiation of energetic materials under dynamical loading, dominated by energy localization on microstructural features such as pores, cracks, and interfaces. We propose a conditional convolutional neural network to predict the shock-induced temperature field as a function of shock strength, for a wide range of microstructures, and obtained via two different simulation methods. The proposed model, denoted MISTnet2, significantly extends prior work that was limited to a single shock strength, model, and type of microstructure. MISTnet2 can contribute to bridging atomistics with coarse-grain simulations and enable first principles predictions of detonation initiation and safety of this class of materials. |
| 2026-01-05 | [Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents](http://arxiv.org/abs/2601.02314v1) | Sourena Khanzadeh | As Large Language Model (LLM) agents are increasingly tasked with high-stakes autonomous decision-making, the transparency of their reasoning processes has become a critical safety concern. While \textit{Chain-of-Thought} (CoT) prompting allows agents to generate human-readable reasoning traces, it remains unclear whether these traces are \textbf{faithful} generative drivers of the model's output or merely \textbf{post-hoc rationalizations}. We introduce \textbf{Project Ariadne}, a novel XAI framework that utilizes Structural Causal Models (SCMs) and counterfactual logic to audit the causal integrity of agentic reasoning. Unlike existing interpretability methods that rely on surface-level textual similarity, Project Ariadne performs \textbf{hard interventions} ($do$-calculus) on intermediate reasoning nodes -- systematically inverting logic, negating premises, and reversing factual claims -- to measure the \textbf{Causal Sensitivity} ($φ$) of the terminal answer. Our empirical evaluation of state-of-the-art models reveals a persistent \textit{Faithfulness Gap}. We define and detect a widespread failure mode termed \textbf{Causal Decoupling}, where agents exhibit a violation density ($ρ$) of up to $0.77$ in factual and scientific domains. In these instances, agents arrive at identical conclusions despite contradictory internal logic, proving that their reasoning traces function as "Reasoning Theater" while decision-making is governed by latent parametric priors. Our findings suggest that current agentic architectures are inherently prone to unfaithful explanation, and we propose the Ariadne Score as a new benchmark for aligning stated logic with model action. |
| 2026-01-05 | [Variability of MHD Instabilities in Benign Termination of High-Current Runaway Electron Beams in the JET and DIII-D Tokamaks](http://arxiv.org/abs/2601.02262v1) | C. F. B. Zimmermann, C. Paz-Soldan et al. | Benign termination, in which magnetohydrodynamic (MHD) instabilities deconfine runaway electrons (REs) following hydrogenic injections, is a promising strategy for mitigating dangerous RE loads after disruptions. Recent experiments on the Joint European Torus (JET) have explored this scenario at higher pre-disruptive plasma currents than are achievable on other devices, revealing challenges in obtaining benign terminations at $I_p \geq 2.5$ MA. This work analyzes the evolution of these high-current RE beams and their terminating MHD events using fast magnetic sensor measurements and EFIT equilibrium reconstructions for approximately $40$ JET and $20$ DIII-D tokamak discharges. On JET, unsuccessful non-benign terminations occur at low edge safety factor ($q_{\text{edge}} \approx 2$), and are preceded by intermittent, non-terminating MHD events at higher rational $q_{\text{edge}}$. Trends in the internal inductance $l_i$ indicate more peaked RE current profiles in the high-$I_p$ non-benign population, which may hinder successful recombination through re-ionization. In contrast, benign terminations on JET typically occur at higher $q_{\text{edge}} \geq 3$ and exhibit less peaked RE current profiles. DIII-D displays a range of terminating edge safety factors, correlated with the measured $l_i$ values. Across both tokamaks, the RE current peaking is therefore found to determine which MHD instability boundary is encountered, confirmed by linear resistive MHD modeling with the CASTOR3D code. Measured growth rates are similar for benign and non-benign cases, indicating that ideal MHD timescales at low density after hydrogenic injection do not alone explain efficient RE deconfinement. Instead, non-benign cases are characterized by their lower MHD perturbation amplitudes $δB$. These observations suggest that the interplay between ideal and resistive dynamics governs the termination process. |
| 2026-01-05 | [From XAI to Stories: A Factorial Study of LLM-Generated Explanation Quality](http://arxiv.org/abs/2601.02224v1) | Fabian Lukassen, Jan Herrmann et al. | Explainable AI (XAI) methods like SHAP and LIME produce numerical feature attributions that remain inaccessible to non expert users. Prior work has shown that Large Language Models (LLMs) can transform these outputs into natural language explanations (NLEs), but it remains unclear which factors contribute to high-quality explanations. We present a systematic factorial study investigating how Forecasting model choice, XAI method, LLM selection, and prompting strategy affect NLE quality. Our design spans four models (XGBoost (XGB), Random Forest (RF), Multilayer Perceptron (MLP), and SARIMAX - comparing black-box Machine-Learning (ML) against classical time-series approaches), three XAI conditions (SHAP, LIME, and a no-XAI baseline), three LLMs (GPT-4o, Llama-3-8B, DeepSeek-R1), and eight prompting strategies. Using G-Eval, an LLM-as-a-judge evaluation method, with dual LLM judges and four evaluation criteria, we evaluate 660 explanations for time-series forecasting. Our results suggest that: (1) XAI provides only small improvements over no-XAI baselines, and only for expert audiences; (2) LLM choice dominates all other factors, with DeepSeek-R1 outperforming GPT-4o and Llama-3; (3) we observe an interpretability paradox: in our setting, SARIMAX yielded lower NLE quality than ML models despite higher prediction accuracy; (4) zero-shot prompting is competitive with self-consistency at 7-times lower cost; and (5) chain-of-thought hurts rather than helps. |
| 2026-01-05 | [LLM-Empowered Functional Safety and Security by Design in Automotive Systems](http://arxiv.org/abs/2601.02215v1) | Nenad Petrovic, Vahid Zolfaghari et al. | This paper presents LLM-empowered workflow to support Software Defined Vehicle (SDV) software development, covering the aspects of security-aware system topology design, as well as event-driven decision-making code analysis. For code analysis we adopt event chains model which provides formal foundations to systematic validation of functional safety, taking into account the semantic validity of messages exchanged between key components, including both CAN and Vehicle Signal Specification (VSS). Analysis of security aspects for topology relies on synergy with Model-Driven Engineering (MDE) approach and Object Constraint Language (OCL) rules. Both locally deployable and proprietary solution are taken into account for evaluation within Advanced Driver-Assistance Systems (ADAS)-related scenarios. |
| 2026-01-05 | [From Chat Control to Robot Control: The Backdoors Left Open for the Sake of Safety](http://arxiv.org/abs/2601.02205v1) | Neziha Akalin, Alberto Giaretta | This paper explores how a recent European Union proposal, the so-called Chat Control law, which creates regulatory incentives for providers to implement content detection and communication scanning, could transform the foundations of human-robot interaction (HRI). As robots increasingly act as interpersonal communication channels in care, education, and telepresence, they convey not only speech but also gesture, emotion, and contextual cues. We argue that extending digital surveillance laws to such embodied systems would entail continuous monitoring, embedding observation into the very design of everyday robots. This regulation blurs the line between protection and control, turning companions into potential informants. At the same time, monitoring mechanisms that undermine end-to-end encryption function as de facto backdoors, expanding the attack surface and allowing adversaries to exploit legally induced monitoring infrastructures. This creates a paradox of safety through insecurity: systems introduced to protect users may instead compromise their privacy, autonomy, and trust. This work does not aim to predict the future, but to raise awareness and help prevent certain futures from materialising. |
| 2026-01-05 | [Streaming Hallucination Detection in Long Chain-of-Thought Reasoning](http://arxiv.org/abs/2601.02170v1) | Haolang Lu, Minghui Pan et al. | Long chain-of-thought (CoT) reasoning improves the performance of large language models, yet hallucinations in such settings often emerge subtly and propagate across reasoning steps. We suggest that hallucination in long CoT reasoning is better understood as an evolving latent state rather than a one-off erroneous event. Accordingly, we treat step-level hallucination judgments as local observations and introduce a cumulative prefix-level hallucination signal that tracks the global evolution of the reasoning state over the entire trajectory. Overall, our approach enables streaming hallucination detection in long CoT reasoning, providing real-time, interpretable evidence. |
| 2026-01-05 | [Simulation of Radiation Chemistry by a One-Shot Hybrid Continuum / Monte Carlo Method](http://arxiv.org/abs/2601.02132v1) | Charlie Fynn Perkins, Marcus Webb et al. | Understanding the spatio-temporal evolution of radiolytic species created by high-energy electrons in water underpins key applications from radiotherapy and nuclear safety to environmental processing and electron microscopy. Here, using the Manchester Inhomogeneous Radiation Chemistry by Linear Expansions (MIRaCLE) toolkit, we introduce and benchmark a novel approach to simulating these processes. Although the initial conditions are determined stochastically, the subsequent time evolution is calculated deterministically using a continuum representation, derived from those initial conditions. This hybrid approach essentially averages over many chemistry ``trajectories'' simultaneously, often converging to the 1% level in one shot, not requiring multiple runs. We demonstrate this new approach through the calculation of time-dependent G-values for e_{aq}^-$, \dot{\mathrm{OH}} and other radiolytic products, including at unprecedented dose rates where calculations which would take years with a conventional Monte Carlo approach can be performed in mere hours on a commercial laptop. We demonstrate that the main artifact of continuum modelling can be mitigated by a correction term. These results establish MIRaCLE as a flexible and efficient platform for modelling long-timescale radiolysis, providing a bridge between Monte Carlo approaches and macroscopic reaction--diffusion schemes, with broad implications for radiation chemistry in medicine, energy, and materials science. |
| 2026-01-05 | [Realistic adversarial scenario generation via human-like pedestrian model for autonomous vehicle control parameter optimisation](http://arxiv.org/abs/2601.02082v1) | Yueyang Wang, Mehmet Dogar et al. | Autonomous vehicles (AVs) are rapidly advancing and are expected to play a central role in future mobility. Ensuring their safe deployment requires reliable interaction with other road users, not least pedestrians. Direct testing on public roads is costly and unsafe for rare but critical interactions, making simulation a practical alternative. Within simulation-based testing, adversarial scenarios are widely used to probe safety limits, but many prioritise difficulty over realism, producing exaggerated behaviours which may result in AV controllers that are overly conservative. We propose an alternative method, instead using a cognitively inspired pedestrian model featuring both inter-individual and intra-individual variability to generate behaviourally plausible adversarial scenarios. We provide a proof of concept demonstration of this method's potential for AV control optimisation, in closed-loop testing and tuning of an AV controller. Our results show that replacing the rule-based CARLA pedestrian with the human-like model yields more realistic gap acceptance patterns and smoother vehicle decelerations. Unsafe interactions occur only for certain pedestrian individuals and conditions, underscoring the importance of human variability in AV testing. Adversarial scenarios generated by this model can be used to optimise AV control towards safer and more efficient behaviour. Overall, this work illustrates how incorporating human-like road user models into simulation-based adversarial testing can enhance the credibility of AV evaluation and provide a practical basis to behaviourally informed controller optimisation. |
| 2026-01-05 | [Simulated Reasoning is Reasoning](http://arxiv.org/abs/2601.02043v1) | Hendrik Kempt, Alon Lavie | Reasoning has long been understood as a pathway between stages of understanding. Proper reasoning leads to understanding of a given subject. This reasoning was conceptualized as a process of understanding in a particular way, i.e., "symbolic reasoning". Foundational Models (FM) demonstrate that this is not a necessary condition for many reasoning tasks: they can "reason" by way of imitating the process of "thinking out loud", testing the produced pathways, and iterating on these pathways on their own. This leads to some form of reasoning that can solve problems on its own or with few-shot learning, but appears fundamentally different from human reasoning due to its lack of grounding and common sense, leading to brittleness of the reasoning process. These insights promise to substantially alter our assessment of reasoning and its necessary conditions, but also inform the approaches to safety and robust defences against this brittleness of FMs. This paper offers and discusses several philosophical interpretations of this phenomenon, argues that the previously apt metaphor of the "stochastic parrot" has lost its relevance and thus should be abandoned, and reflects on different normative elements in the safety- and appropriateness-considerations emerging from these reasoning models and their growing capacity. |
| 2026-01-05 | [Exploring Approaches for Detecting Memorization of Recommender System Data in Large Language Models](http://arxiv.org/abs/2601.02002v1) | Antonio Colacicco, Vito Guida et al. | Large Language Models (LLMs) are increasingly applied in recommendation scenarios due to their strong natural language understanding and generation capabilities. However, they are trained on vast corpora whose contents are not publicly disclosed, raising concerns about data leakage. Recent work has shown that the MovieLens-1M dataset is memorized by both the LLaMA and OpenAI model families, but the extraction of such memorized data has so far relied exclusively on manual prompt engineering. In this paper, we pose three main questions: Is it possible to enhance manual prompting? Can LLM memorization be detected through methods beyond manual prompting? And can the detection of data leakage be automated? To address these questions, we evaluate three approaches: (i) jailbreak prompt engineering; (ii) unsupervised latent knowledge discovery, probing internal activations via Contrast-Consistent Search (CCS) and Cluster-Norm; and (iii) Automatic Prompt Engineering (APE), which frames prompt discovery as a meta-learning process that iteratively refines candidate instructions. Experiments on MovieLens-1M using LLaMA models show that jailbreak prompting does not improve the retrieval of memorized items and remains inconsistent; CCS reliably distinguishes genuine from fabricated movie titles but fails on numerical user and rating data; and APE retrieves item-level information with moderate success yet struggles to recover numerical interactions. These findings suggest that automatically optimizing prompts is the most promising strategy for extracting memorized samples. |
| 2026-01-05 | [Agentic AI in Remote Sensing: Foundations, Taxonomy, and Emerging Systems](http://arxiv.org/abs/2601.01891v1) | Niloufar Alipour Talemi, Julia Boone et al. | The paradigm of Earth Observation analysis is shifting from static deep learning models to autonomous agentic AI. Although recent vision foundation models and multimodal large language models advance representation learning, they often lack the sequential planning and active tool orchestration required for complex geospatial workflows. This survey presents the first comprehensive review of agentic AI in remote sensing. We introduce a unified taxonomy distinguishing between single-agent copilots and multi-agent systems while analyzing architectural foundations such as planning mechanisms, retrieval-augmented generation, and memory structures. Furthermore, we review emerging benchmarks that move the evaluation from pixel-level accuracy to trajectory-aware reasoning correctness. By critically examining limitations in grounding, safety, and orchestration, this work outlines a strategic roadmap for the development of robust, autonomous geospatial intelligence. |
| 2026-01-05 | [Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance](http://arxiv.org/abs/2601.01887v1) | Jiawen Zhang, Lipeng He et al. | Fine-tuning safety-aligned large language models (LLMs) can substantially compromise their safety. Previous approaches require many safety samples or calibration sets, which not only incur significant computational overhead during realignment but also lead to noticeable degradation in model utility. Contrary to this belief, we show that safety alignment can be fully recovered with only a single safety example, without sacrificing utility and at minimal cost. Remarkably, this recovery is effective regardless of the number of harmful examples used in fine-tuning or the size of the underlying model, and convergence is achieved within just a few epochs. Furthermore, we uncover the low-rank structure of the safety gradient, which explains why such efficient correction is possible. We validate our findings across five safety-aligned LLMs and multiple datasets, demonstrating the generality of our approach. |
| 2026-01-05 | [COMPASS: A Framework for Evaluating Organization-Specific Policy Alignment in LLMs](http://arxiv.org/abs/2601.01836v1) | Dasol Choi, DongGeon Lee et al. | As large language models are deployed in high-stakes enterprise applications, from healthcare to finance, ensuring adherence to organization-specific policies has become essential. Yet existing safety evaluations focus exclusively on universal harms. We present COMPASS (Company/Organization Policy Alignment Assessment), the first systematic framework for evaluating whether LLMs comply with organizational allowlist and denylist policies. We apply COMPASS to eight diverse industry scenarios, generating and validating 5,920 queries that test both routine compliance and adversarial robustness through strategically designed edge cases. Evaluating seven state-of-the-art models, we uncover a fundamental asymmetry: models reliably handle legitimate requests (>95% accuracy) but catastrophically fail at enforcing prohibitions, refusing only 13-40% of adversarial denylist violations. These results demonstrate that current LLMs lack the robustness required for policy-critical deployments, establishing COMPASS as an essential evaluation framework for organizational AI safety. |
| 2026-01-05 | [Sparse Threats, Focused Defense: Criticality-Aware Robust Reinforcement Learning for Safe Autonomous Driving](http://arxiv.org/abs/2601.01800v1) | Qi Wei, Junchao Fan et al. | Reinforcement learning (RL) has shown considerable potential in autonomous driving (AD), yet its vulnerability to perturbations remains a critical barrier to real-world deployment. As a primary countermeasure, adversarial training improves policy robustness by training the AD agent in the presence of an adversary that deliberately introduces perturbations. Existing approaches typically model the interaction as a zero-sum game with continuous attacks. However, such designs overlook the inherent asymmetry between the agent and the adversary and then fail to reflect the sparsity of safety-critical risks, rendering the achieved robustness inadequate for practical AD scenarios. To address these limitations, we introduce criticality-aware robust RL (CARRL), a novel adversarial training approach for handling sparse, safety-critical risks in autonomous driving. CARRL consists of two interacting components: a risk exposure adversary (REA) and a risk-targeted robust agent (RTRA). We model the interaction between the REA and RTRA as a general-sum game, allowing the REA to focus on exposing safety-critical failures (e.g., collisions) while the RTRA learns to balance safety with driving efficiency. The REA employs a decoupled optimization mechanism to better identify and exploit sparse safety-critical moments under a constrained budget. However, such focused attacks inevitably result in a scarcity of adversarial data. The RTRA copes with this scarcity by jointly leveraging benign and adversarial experiences via a dual replay buffer and enforces policy consistency under perturbations to stabilize behavior. Experimental results demonstrate that our approach reduces the collision rate by at least 22.66\% across all cases compared to state-of-the-art baseline methods. |
| 2026-01-05 | [LIA: Supervised Fine-Tuning of Large Language Models for Automatic Issue Assignment](http://arxiv.org/abs/2601.01780v1) | Arsham Khosravani, Alireza Hosseinpour et al. | Issue assignment is a critical process in software maintenance, where new issue reports are validated and assigned to suitable developers. However, manual issue assignment is often inconsistent and error-prone, especially in large open-source projects where thousands of new issues are reported monthly. Existing automated approaches have shown promise, but many rely heavily on large volumes of project-specific training data or relational information that is often sparse and noisy, which limits their effectiveness. To address these challenges, we propose LIA (LLM-based Issue Assignment), which employs supervised fine-tuning to adapt an LLM, DeepSeek-R1-Distill-Llama-8B in this work, for automatic issue assignment. By leveraging the LLM's pretrained semantic understanding of natural language and software-related text, LIA learns to generate ranked developer recommendations directly from issue titles and descriptions. The ranking is based on the model's learned understanding of historical issue-to-developer assignments, using patterns from past tasks to infer which developers are most likely to handle new issues. Through comprehensive evaluation, we show that LIA delivers substantial improvements over both its base pretrained model and state-of-the-art baselines. It achieves up to +187.8% higher Hit@1 compared to the DeepSeek-R1-Distill-Llama-8B pretrained base model, and outperforms four leading issue assignment methods by as much as +211.2% in Hit@1 score. These results highlight the effectiveness of domain-adapted LLMs for software maintenance tasks and establish LIA as a practical, high-performing solution for issue assignment. |
| 2026-01-05 | [AlignDrive: Aligned Lateral-Longitudinal Planning for End-to-End Autonomous Driving](http://arxiv.org/abs/2601.01762v1) | Yanhao Wu, Haoyang Zhang et al. | End-to-end autonomous driving has rapidly progressed, enabling joint perception and planning in complex environments. In the planning stage, state-of-the-art (SOTA) end-to-end autonomous driving models decouple planning into parallel lateral and longitudinal predictions. While effective, this parallel design can lead to i) coordination failures between the planned path and speed, and ii) underutilization of the drive path as a prior for longitudinal planning, thus redundantly encoding static information. To address this, we propose a novel cascaded framework that explicitly conditions longitudinal planning on the drive path, enabling coordinated and collision-aware lateral and longitudinal planning. Specifically, we introduce a path-conditioned formulation that explicitly incorporates the drive path into longitudinal planning. Building on this, the model predicts longitudinal displacements along the drive path rather than full 2D trajectory waypoints. This design simplifies longitudinal reasoning and more tightly couples it with lateral planning. Additionally, we introduce a planning-oriented data augmentation strategy that simulates rare safety-critical events, such as vehicle cut-ins, by adding agents and relabeling longitudinal targets to avoid collision. Evaluated on the challenging Bench2Drive benchmark, our method sets a new SOTA, achieving a driving score of 89.07 and a success rate of 73.18%, demonstrating significantly improved coordination and safety |
| 2026-01-05 | [Crafting Adversarial Inputs for Large Vision-Language Models Using Black-Box Optimization](http://arxiv.org/abs/2601.01747v1) | Jiwei Guan, Haibo Jin et al. | Recent advancements in Large Vision-Language Models (LVLMs) have shown groundbreaking capabilities across diverse multimodal tasks. However, these models remain vulnerable to adversarial jailbreak attacks, where adversaries craft subtle perturbations to bypass safety mechanisms and trigger harmful outputs. Existing white-box attacks methods require full model accessibility, suffer from computing costs and exhibit insufficient adversarial transferability, making them impractical for real-world, black-box settings. To address these limitations, we propose a black-box jailbreak attack on LVLMs via Zeroth-Order optimization using Simultaneous Perturbation Stochastic Approximation (ZO-SPSA). ZO-SPSA provides three key advantages: (i) gradient-free approximation by input-output interactions without requiring model knowledge, (ii) model-agnostic optimization without the surrogate model and (iii) lower resource requirements with reduced GPU memory consumption. We evaluate ZO-SPSA on three LVLMs, including InstructBLIP, LLaVA and MiniGPT-4, achieving the highest jailbreak success rate of 83.0% on InstructBLIP, while maintaining imperceptible perturbations comparable to white-box methods. Moreover, adversarial examples generated from MiniGPT-4 exhibit strong transferability to other LVLMs, with ASR reaching 64.18%. These findings underscore the real-world feasibility of black-box jailbreaks and expose critical weaknesses in the safety mechanisms of current LVLMs |
| 2026-01-05 | [AI Agent Systems: Architectures, Applications, and Evaluation](http://arxiv.org/abs/2601.01743v1) | Bin Xu | AI agents -- systems that combine foundation models with reasoning, planning, memory, and tool use -- are rapidly becoming a practical interface between natural-language intent and real-world computation. This survey synthesizes the emerging landscape of AI agent architectures across: (i) deliberation and reasoning (e.g., chain-of-thought-style decomposition, self-reflection and verification, and constraint-aware decision making), (ii) planning and control (from reactive policies to hierarchical and multi-step planners), and (iii) tool calling and environment interaction (retrieval, code execution, APIs, and multimodal perception). We organize prior work into a unified taxonomy spanning agent components (policy/LLM core, memory, world models, planners, tool routers, and critics), orchestration patterns (single-agent vs.\ multi-agent; centralized vs.\ decentralized coordination), and deployment settings (offline analysis vs.\ online interactive assistance; safety-critical vs.\ open-ended tasks). We discuss key design trade-offs -- latency vs.\ accuracy, autonomy vs.\ controllability, and capability vs.\ reliability -- and highlight how evaluation is complicated by non-determinism, long-horizon credit assignment, tool and environment variability, and hidden costs such as retries and context growth. Finally, we summarize measurement and benchmarking practices (task suites, human preference and utility metrics, success under constraints, robustness and security) and identify open challenges including verification and guardrails for tool actions, scalable memory and context management, interpretability of agent decisions, and reproducible evaluation under realistic workloads. |
| 2026-01-05 | [Simulations and Advancements in MRI-Guided Power-Driven Ferric Tools for Wireless Therapeutic Interventions](http://arxiv.org/abs/2601.01726v1) | Wenhui Chu, Aobo Jin et al. | Designing a robotic system that functions effectively within the specific environment of a Magnetic Resonance Imaging (MRI) scanner requires solving numerous technical issues, such as maintaining the robot's precision and stability under strong magnetic fields. This research focuses on enhancing MRI's role in medical imaging, especially in its application to guide intravascular interventions using robot-assisted devices. A newly developed computational system is introduced, designed for seamless integration with the MRI scanner, including a computational unit and user interface. This system processes MR images to delineate the vascular network, establishing virtual paths and boundaries within vessels to prevent procedural damage. Key findings reveal the system's capability to create tailored magnetic field gradient patterns for device control, considering the vessel's geometry and safety norms, and adapting to different blood flow characteristics for finer navigation. Additionally, the system's modeling aspect assesses the safety and feasibility of navigating pre-set vascular paths. Conclusively, this system, based on the Qt framework and C/C++, with specialized software modules, represents a major step forward in merging imaging technology with robotic aid, significantly enhancing precision and safety in intravascular procedures. |

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



