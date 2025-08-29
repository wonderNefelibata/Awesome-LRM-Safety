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
| 2025-08-28 | [Lithiation Analysis of Metal Components for Li-Ion Battery using Ion Beams](http://arxiv.org/abs/2508.21017v1) | Arturo Galindo, Neubi Xavier et al. | Metal components are extensively used as current collectors, anodes, and interlayers in lithium-ion batteries. Integrating these functions into one component enhances the cell energy density and simplifies its design. However, this multifunctional component must meet stringent requirements, including high and reversible Li storage capacity, rapid lithiation/delithiation kinetics, mechanical stability, and safety. Six single-atom metals (Mg, Zn, Al, Ag, Sn and Cu) are screened for lithiation behavior through their interaction with ion beams in electrochemically tested samples subjected to both weak and strong lithiation regimes. These different lithiation regimes allowed us to differentiate between the thermodynamics and kinetic aspects of the lithiation process. Three types of ions are used to determine Li depth profile: $H^+$ for nuclear reaction analysis (NRA), $He^+$ for Rutherford backscattering (RBS), and $Ga^+$ for focused ion beam (FIB) milling. The study reveals three lithiation behaviors: (i) Zn, Al, Sn form pure alloys with Li; (ii) Mg, Ag create intercalation solid solutions; (iii) Cu acts as a lithiation barrier. NRA and RBS offer direct and quantitative data, providing a more comprehensive understanding of the lithiation process in LIB components. These findings fit well with our ab-initio simulation results, establishing a direct correlation between electrochemical features and fundamental thermodynamic parameters. |
| 2025-08-28 | [Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees](http://arxiv.org/abs/2508.21001v1) | Yaniv Hassidof, Tom Jurgenson et al. | Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: https://sites.google.com/view/ditree. |
| 2025-08-28 | [ProactiveEval: A Unified Evaluation Framework for Proactive Dialogue Agents](http://arxiv.org/abs/2508.20973v1) | Tianjian Liu, Fanqi Wan et al. | Proactive dialogue has emerged as a critical and challenging research problem in advancing large language models (LLMs). Existing works predominantly focus on domain-specific or task-oriented scenarios, which leads to fragmented evaluations and limits the comprehensive exploration of models' proactive conversation abilities. In this work, we propose ProactiveEval, a unified framework designed for evaluating proactive dialogue capabilities of LLMs. This framework decomposes proactive dialogue into target planning and dialogue guidance, establishing evaluation metrics across various domains. Moreover, it also enables the automatic generation of diverse and challenging evaluation data. Based on the proposed framework, we develop 328 evaluation environments spanning 6 distinct domains. Through experiments with 22 different types of LLMs, we show that DeepSeek-R1 and Claude-3.7-Sonnet exhibit exceptional performance on target planning and dialogue guidance tasks, respectively. Finally, we investigate how reasoning capabilities influence proactive behaviors and discuss their implications for future model development. |
| 2025-08-28 | [AI Reasoning Models for Problem Solving in Physics](http://arxiv.org/abs/2508.20941v1) | Amir Bralin, N. Sanjay Rebello | Reasoning models are the new generation of Large Language Models (LLMs) capable of complex problem solving. Their reliability in solving introductory physics problems was tested by evaluating a sample of n = 5 solutions generated by one such model -- OpenAI's o3-mini -- per each problem from 20 chapters of a standard undergraduate textbook. In total, N = 408 problems were given to the model and N x n = 2,040 generated solutions examined. The model successfully solved 94% of the problems posed, excelling at the beginning topics in mechanics but struggling with the later ones such as waves and thermodynamics. |
| 2025-08-28 | [COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans](http://arxiv.org/abs/2508.20920v1) | Enrico Martini, Ho Jin Choi et al. | In the era of Industry 5.0, monitoring human activity is essential for ensuring both ergonomic safety and overall well-being. While multi-camera centralized setups improve pose estimation accuracy, they often suffer from high computational costs and bandwidth requirements, limiting scalability and real-time applicability. Distributing processing across edge devices can reduce network bandwidth and computational load. On the other hand, the constrained resources of edge devices lead to accuracy degradation, and the distribution of computation leads to temporal and spatial inconsistencies. We address this challenge by proposing COMETH (Convex Optimization for Multiview Estimation and Tracking of Humans), a lightweight algorithm for real-time multi-view human pose fusion that relies on three concepts: it integrates kinematic and biomechanical constraints to increase the joint positioning accuracy; it employs convex optimization-based inverse kinematics for spatial fusion; and it implements a state observer to improve temporal consistency. We evaluate COMETH on both public and industrial datasets, where it outperforms state-of-the-art methods in localization, detection, and tracking accuracy. The proposed fusion pipeline enables accurate and scalable human motion tracking, making it well-suited for industrial and safety-critical applications. The code is publicly available at https://github.com/PARCO-LAB/COMETH. |
| 2025-08-28 | [JADES: A Universal Framework for Jailbreak Assessment via Decompositional Scoring](http://arxiv.org/abs/2508.20848v1) | Junjie Chu, Mingjie Li et al. | Accurately determining whether a jailbreak attempt has succeeded is a fundamental yet unresolved challenge. Existing evaluation methods rely on misaligned proxy indicators or naive holistic judgments. They frequently misinterpret model responses, leading to inconsistent and subjective assessments that misalign with human perception. To address this gap, we introduce JADES (Jailbreak Assessment via Decompositional Scoring), a universal jailbreak evaluation framework. Its key mechanism is to automatically decompose an input harmful question into a set of weighted sub-questions, score each sub-answer, and weight-aggregate the sub-scores into a final decision. JADES also incorporates an optional fact-checking module to strengthen the detection of hallucinations in jailbreak responses. We validate JADES on JailbreakQR, a newly introduced benchmark proposed in this work, consisting of 400 pairs of jailbreak prompts and responses, each meticulously annotated by humans. In a binary setting (success/failure), JADES achieves 98.5% agreement with human evaluators, outperforming strong baselines by over 9%. Re-evaluating five popular attacks on four LLMs reveals substantial overestimation (e.g., LAA's attack success rate on GPT-3.5-Turbo drops from 93% to 69%). Our results show that JADES could deliver accurate, consistent, and interpretable evaluations, providing a reliable basis for measuring future jailbreak attacks. |
| 2025-08-28 | [Activated Backstepping with Control Barrier Functions for the Safe Navigation of Automated Vehicles](http://arxiv.org/abs/2508.20822v1) | Laszlo Gacsi, Max H. Cohen et al. | This paper introduces a novel safety-critical control method through the synthesis of control barrier functions (CBFs) for systems with high-relative-degree safety constraints. By extending the procedure of CBF backstepping, we propose activated backstepping - a constructive method to synthesize valid CBFs. The novelty of our method is the incorporation of an activation function into the CBF, which offers less conservative safe sets in the state space than standard CBF backstepping. We demonstrate the proposed method on an inverted pendulum example, where we explain the underlying geometric meaning in the state space and provide comparisons with existing CBF synthesis techniques. Finally, we implement our method to achieve collision-free navigation for automated vehicles using a kinematic bicycle model in simulation. |
| 2025-08-28 | [Uncertainty Aware-Predictive Control Barrier Functions: Safer Human Robot Interaction through Probabilistic Motion Forecasting](http://arxiv.org/abs/2508.20812v1) | Lorenzo Busellato, Federico Cunico et al. | To enable flexible, high-throughput automation in settings where people and robots share workspaces, collaborative robotic cells must reconcile stringent safety guarantees with the need for responsive and effective behavior. A dynamic obstacle is the stochastic, task-dependent variability of human motion: when robots fall back on purely reactive or worst-case envelopes, they brake unnecessarily, stall task progress, and tamper with the fluidity that true Human-Robot Interaction demands. In recent years, learning-based human-motion prediction has rapidly advanced, although most approaches produce worst-case scenario forecasts that often do not treat prediction uncertainty in a well-structured way, resulting in over-conservative planning algorithms, limiting their flexibility. We introduce Uncertainty-Aware Predictive Control Barrier Functions (UA-PCBFs), a unified framework that fuses probabilistic human hand motion forecasting with the formal safety guarantees of Control Barrier Functions. In contrast to other variants, our framework allows for dynamic adjustment of the safety margin thanks to the human motion uncertainty estimation provided by a forecasting module. Thanks to uncertainty estimation, UA-PCBFs empower collaborative robots with a deeper understanding of future human states, facilitating more fluid and intelligent interactions through informed motion planning. We validate UA-PCBFs through comprehensive real-world experiments with an increasing level of realism, including automated setups (to perform exactly repeatable motions) with a robotic hand and direct human-robot interactions (to validate promptness, usability, and human confidence). Relative to state-of-the-art HRI architectures, UA-PCBFs show better performance in task-critical metrics, significantly reducing the number of violations of the robot's safe space during interaction with respect to the state-of-the-art. |
| 2025-08-28 | [When technology is not enough: Insights from a pilot cybersecurity culture assessment in a safety-critical industrial organisation](http://arxiv.org/abs/2508.20811v1) | Tita Alissa Bach, Linn Pedersen et al. | As cyber threats increasingly exploit human behaviour, technical controls alone cannot ensure organisational cybersecurity (CS). Strengthening cybersecurity culture (CSC) is vital in safety-critical industries, yet empirical research in real-world industrial setttings is scarce. This paper addresses this gap through a pilot mixed-methods CSC assessment in a global safety-critical organisation. We examined employees' CS knowledge, attitudes, behaviours, and organisational factors shaping them. A survey and semi-structured interviews were conducted at a global organisation in safety-critical industries, across two countries chosen for contrasting phishing simulation performance: Country 1 stronger, Country 2 weaker. In Country 1, 258 employees were invited (67%), in Country 2, 113 were invited (30%). Interviews included 20 and 10 participants respectively. Overall CSC profiles were similar but revealed distinct challenges. Both showed strong phishing awareness and prioritised CS, yet most viewed phishing as the main risk and lacked clarity on handling other incidents. Line managers were default contacts, but follow-up on reported concerns was unclear. Participants emphasized aligning CS expectations with job relevance and workflows. Key contributors to differences emerged: Country 1 had external employees with limited access to CS training and policies, highlighting monitoring gaps. In Country 2, low survey response stemmed from a "no-link in email" policy. While this policy may have boosted phishing performance, it also underscored inconsistencies in CS practices. Findings show that resilient CSC requires leadership involvement, targeted communication, tailored measures, policy-practice alignment, and regular assessments. Embedding these into strategy complements technical defences and strengthens sustainable CS in safety-critical settings. |
| 2025-08-28 | [Safer Skin Lesion Classification with Global Class Activation Probability Map Evaluation and SafeML](http://arxiv.org/abs/2508.20776v1) | Kuniko Paxton, Koorosh Aslansefat et al. | Recent advancements in skin lesion classification models have significantly improved accuracy, with some models even surpassing dermatologists' diagnostic performance. However, in medical practice, distrust in AI models remains a challenge. Beyond high accuracy, trustworthy, explainable diagnoses are essential. Existing explainability methods have reliability issues, with LIME-based methods suffering from inconsistency, while CAM-based methods failing to consider all classes. To address these limitations, we propose Global Class Activation Probabilistic Map Evaluation, a method that analyses all classes' activation probability maps probabilistically and at a pixel level. By visualizing the diagnostic process in a unified manner, it helps reduce the risk of misdiagnosis. Furthermore, the application of SafeML enhances the detection of false diagnoses and issues warnings to doctors and patients as needed, improving diagnostic reliability and ultimately patient safety. We evaluated our method using the ISIC datasets with MobileNetV2 and Vision Transformers. |
| 2025-08-28 | [Turning the Spell Around: Lightweight Alignment Amplification via Rank-One Safety Injection](http://arxiv.org/abs/2508.20766v1) | Harethah Abu Shairah, Hasan Abed Al Kader Hammoud et al. | Safety alignment in Large Language Models (LLMs) often involves mediating internal representations to refuse harmful requests. Recent research has demonstrated that these safety mechanisms can be bypassed by ablating or removing specific representational directions within the model. In this paper, we propose the opposite approach: Rank-One Safety Injection (ROSI), a white-box method that amplifies a model's safety alignment by permanently steering its activations toward the refusal-mediating subspace. ROSI operates as a simple, fine-tuning-free rank-one weight modification applied to all residual stream write matrices. The required safety direction can be computed from a small set of harmful and harmless instruction pairs. We show that ROSI consistently increases safety refusal rates - as evaluated by Llama Guard 3 - while preserving the utility of the model on standard benchmarks such as MMLU, HellaSwag, and Arc. Furthermore, we show that ROSI can also re-align 'uncensored' models by amplifying their own latent safety directions, demonstrating its utility as an effective last-mile safety procedure. Our results suggest that targeted, interpretable weight steering is a cheap and potent mechanism to improve LLM safety, complementing more resource-intensive fine-tuning paradigms. |
| 2025-08-28 | [From Law to Gherkin: A Human-Centred Quasi-Experiment on the Quality of LLM-Generated Behavioural Specifications from Food-Safety Regulations](http://arxiv.org/abs/2508.20744v1) | Shabnam Hassani, Mehrdad Sabetzadeh et al. | Context: Laws and regulations increasingly affect software design and quality assurance, but legal texts are written in technology-neutral language. This creates challenges for engineers who must develop compliance artifacts such as requirements and acceptance criteria. Manual creation is labor-intensive, error-prone, and requires domain expertise. Advances in Generative AI (GenAI), especially Large Language Models (LLMs), offer a way to automate deriving such artifacts.   Objective: We present the first systematic human-subject study of LLMs' ability to derive behavioral specifications from legal texts using a quasi-experimental design. These specifications translate legal requirements into a developer-friendly form.   Methods: Ten participants evaluated specifications generated from food-safety regulations by Claude and Llama. Using Gherkin, a structured BDD language, 60 specifications were produced. Each participant assessed 12 across five criteria: Relevance, Clarity, Completeness, Singularity, and Time Savings. Each specification was reviewed by two participants, yielding 120 assessments.   Results: For Relevance, 75% of ratings were highest and 20% second-highest. Clarity reached 90% highest. Completeness: 75% highest, 19% second. Singularity: 82% highest, 12% second. Time Savings: 68% highest, 24% second. No lowest ratings occurred. Mann-Whitney U tests showed no significant differences across participants or models. Llama slightly outperformed Claude in Clarity, Completeness, and Time Savings, while Claude was stronger in Singularity. Feedback noted hallucinations and omissions but confirmed the utility of the specifications.   Conclusion: LLMs can generate high-quality Gherkin specifications from legal texts, reducing manual effort and providing structured artifacts useful for implementation, assurance, and test generation. |
| 2025-08-28 | [Rethinking Testing for LLM Applications: Characteristics, Challenges, and a Lightweight Interaction Protocol](http://arxiv.org/abs/2508.20737v1) | Wei Ma, Yixiao Yang et al. | Applications of Large Language Models~(LLMs) have evolved from simple text generators into complex software systems that integrate retrieval augmentation, tool invocation, and multi-turn interactions. Their inherent non-determinism, dynamism, and context dependence pose fundamental challenges for quality assurance. This paper decomposes LLM applications into a three-layer architecture: \textbf{\textit{System Shell Layer}}, \textbf{\textit{Prompt Orchestration Layer}}, and \textbf{\textit{LLM Inference Core}}. We then assess the applicability of traditional software testing methods in each layer: directly applicable at the shell layer, requiring semantic reinterpretation at the orchestration layer, and necessitating paradigm shifts at the inference core. A comparative analysis of Testing AI methods from the software engineering community and safety analysis techniques from the AI community reveals structural disconnects in testing unit abstraction, evaluation metrics, and lifecycle management. We identify four fundamental differences that underlie 6 core challenges. To address these, we propose four types of collaborative strategies (\emph{Retain}, \emph{Translate}, \emph{Integrate}, and \emph{Runtime}) and explore a closed-loop, trustworthy quality assurance framework that combines pre-deployment validation with runtime monitoring. Based on these strategies, we offer practical guidance and a protocol proposal to support the standardization and tooling of LLM application testing. We propose a protocol \textbf{\textit{Agent Interaction Communication Language}} (AICL) that is used to communicate between AI agents. AICL has the test-oriented features and is easily integrated in the current agent framework. |
| 2025-08-28 | [rStar2-Agent: Agentic Reasoning Technical Report](http://arxiv.org/abs/2508.20722v1) | Ning Shang, Yifei Liu et al. | We introduce rStar2-Agent, a 14B math reasoning model trained with agentic reinforcement learning to achieve frontier-level performance. Beyond current long CoT, the model demonstrates advanced cognitive behaviors, such as thinking carefully before using Python coding tools and reflecting on code execution feedback to autonomously explore, verify, and refine intermediate steps in complex problem-solving. This capability is enabled through three key innovations that makes agentic RL effective at scale: (i) an efficient RL infrastructure with a reliable Python code environment that supports high-throughput execution and mitigates the high rollout costs, enabling training on limited GPU resources (64 MI300X GPUs); (ii) GRPO-RoC, an agentic RL algorithm with a Resample-on-Correct rollout strategy that addresses the inherent environment noises from coding tools, allowing the model to reason more effectively in a code environment; (iii) An efficient agent training recipe that starts with non-reasoning SFT and progresses through multi-RL stages, yielding advanced cognitive abilities with minimal compute cost. To this end, rStar2-Agent boosts a pre-trained 14B model to state of the art in only 510 RL steps within one week, achieving average pass@1 scores of 80.6% on AIME24 and 69.8% on AIME25, surpassing DeepSeek-R1 (671B) with significantly shorter responses. Beyond mathematics, rStar2-Agent-14B also demonstrates strong generalization to alignment, scientific reasoning, and agentic tool-use tasks. Code and training recipes are available at https://github.com/microsoft/rStar. |
| 2025-08-28 | [Upper Limits on the Isotropic Gravitational-Wave Background from the first part of LIGO, Virgo, and KAGRA's fourth Observing Run](http://arxiv.org/abs/2508.20721v1) | The LIGO Scientific Collaboration, the Virgo Collaboration et al. | We present results from the search for an isotropic gravitational-wave background using Advanced LIGO and Advanced Virgo data from O1 through O4a, the first part of the fourth observing run. This background is the accumulated signal from unresolved sources throughout cosmic history and encodes information about the merger history of compact binaries throughout the Universe, as well as exotic physics and potentially primordial processes from the early cosmos. Our cross-correlation analysis reveals no statistically significant background signal, enabling us to constrain several theoretical scenarios. For compact binary coalescences which approximately follow a 2/3 power-law spectrum, we constrain the fractional energy density to $\Omega_{\rm GW}(25{\rm Hz})\leq 2.0\times 10^{-9}$ (95% cred.), a factor of 1.7 improvement over previous results. Scale-invariant backgrounds are constrained to $\Omega_{\rm GW}(25{\rm Hz})\leq 2.8\times 10^{-9}$, representing a 2.1x sensitivity gain. We also place new limits on gravity theories predicting non-standard polarization modes and confirm that terrestrial magnetic noise sources remain below detection threshold. Combining these spectral limits with population models for GWTC-4, the latest gravitational-wave event catalog, we find our constraints remain above predicted merger backgrounds but are approaching detectability. The joint analysis combining the background limits shown here with the GWTC-4 catalog enables improved inference of the binary black hole merger rate evolution across cosmic time. Employing GWTC-4 inference results and standard modeling choices, we estimate that the total background arising from compact binary coalescences is $\Omega_{\rm CBC}(25{\rm Hz})={0.9^{+1.1}_{-0.5}\times 10^{-9}}$ at 90% confidence, where the largest contribution is due to binary black holes only, $\Omega_{\rm BBH}(25{\rm Hz})=0.8^{+1.1}_{-0.5}\times 10^{-9}$. |
| 2025-08-28 | [Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning](http://arxiv.org/abs/2508.20697v1) | Weitao Feng, Lixu Wang et al. | As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense. |
| 2025-08-28 | [Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking](http://arxiv.org/abs/2508.20661v1) | TianChen Huang, Wei Gao et al. | Traversing narrow beams is challenging for humanoids due to sparse, safety-critical contacts and the fragility of purely learned policies. We propose a physically grounded, two-stage framework that couples an XCoM/LIPM footstep template with a lightweight residual planner and a simple low-level tracker. Stage-1 is trained on flat ground: the tracker learns to robustly follow footstep targets by adding small random perturbations to heuristic footsteps, without any hand-crafted centerline locking, so it acquires stable contact scheduling and strong target-tracking robustness. Stage-2 is trained in simulation on a beam: a high-level planner predicts a body-frame residual (Delta x, Delta y, Delta psi) for the swing foot only, refining the template step to prioritize safe, precise placement under narrow support while preserving interpretability. To ease deployment, sensing is kept minimal and consistent between simulation and hardware: the planner consumes compact, forward-facing elevation cues together with onboard IMU and joint signals. On a Unitree G1, our system reliably traverses a 0.2 m-wide, 3 m-long beam. Across simulation and real-world studies, residual refinement consistently outperforms template-only and monolithic baselines in success rate, centerline adherence, and safety margins, while the structured footstep interface enables transparent analysis and low-friction sim-to-real transfer. |
| 2025-08-28 | [Improving Alignment in LVLMs with Debiased Self-Judgment](http://arxiv.org/abs/2508.20655v1) | Sihan Yang, Chenhang Cui et al. | The rapid advancements in Large Language Models (LLMs) and Large Visual-Language Models (LVLMs) have opened up new opportunities for integrating visual and linguistic modalities. However, effectively aligning these modalities remains challenging, often leading to hallucinations--where generated outputs are not grounded in the visual input--and raising safety concerns across various domains. Existing alignment methods, such as instruction tuning and preference tuning, often rely on external datasets, human annotations, or complex post-processing, which limit scalability and increase costs. To address these challenges, we propose a novel approach that generates the debiased self-judgment score, a self-evaluation metric created internally by the model without relying on external resources. This enables the model to autonomously improve alignment. Our method enhances both decoding strategies and preference tuning processes, resulting in reduced hallucinations, enhanced safety, and improved overall capability. Empirical results show that our approach significantly outperforms traditional methods, offering a more effective solution for aligning LVLMs. |
| 2025-08-28 | [CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics](http://arxiv.org/abs/2508.20643v1) | Stefano Fumero, Kai Huang et al. | Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents. |
| 2025-08-28 | [UTA-Sign: Unsupervised Thermal Video Augmentation via Event-Assisted Traffic Signage Sketching](http://arxiv.org/abs/2508.20594v1) | Yuqi Han, Songqian Zhang et al. | The thermal camera excels at perceiving outdoor environments under low-light conditions, making it ideal for applications such as nighttime autonomous driving and unmanned navigation. However, thermal cameras encounter challenges when capturing signage from objects made of similar materials, which can pose safety risks for accurately understanding semantics in autonomous driving systems. In contrast, the neuromorphic vision camera, also known as an event camera, detects changes in light intensity asynchronously and has proven effective in high-speed, low-light traffic environments. Recognizing the complementary characteristics of these two modalities, this paper proposes UTA-Sign, an unsupervised thermal-event video augmentation for traffic signage in low-illumination environments, targeting elements such as license plates and roadblock indicators. To address the signage blind spots of thermal imaging and the non-uniform sampling of event cameras, we developed a dual-boosting mechanism that fuses thermal frames and event signals for consistent signage representation over time. The proposed method utilizes thermal frames to provide accurate motion cues as temporal references for aligning the uneven event signals. At the same time, event signals contribute subtle signage content to the raw thermal frames, enhancing the overall understanding of the environment. The proposed method is validated on datasets collected from real-world scenarios, demonstrating superior quality in traffic signage sketching and improved detection accuracy at the perceptual level. |

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



