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
| 2025-10-06 | [Automaton Constrained Q-Learning](http://arxiv.org/abs/2510.05061v1) | Anastasios Manganaris, Vittorio Giammarino et al. | Real-world robotic tasks often require agents to achieve sequences of goals while respecting time-varying safety constraints. However, standard Reinforcement Learning (RL) paradigms are fundamentally limited in these settings. A natural approach to these problems is to combine RL with Linear-time Temporal Logic (LTL), a formal language for specifying complex, temporally extended tasks and safety constraints. Yet, existing RL methods for LTL objectives exhibit poor empirical performance in complex and continuous environments. As a result, no scalable methods support both temporally ordered goals and safety simultaneously, making them ill-suited for realistic robotics scenarios. We propose Automaton Constrained Q-Learning (ACQL), an algorithm that addresses this gap by combining goal-conditioned value learning with automaton-guided reinforcement. ACQL supports most LTL task specifications and leverages their automaton representation to explicitly encode stage-wise goal progression and both stationary and non-stationary safety constraints. We show that ACQL outperforms existing methods across a range of continuous control tasks, including cases where prior methods fail to satisfy either goal-reaching or safety constraints. We further validate its real-world applicability by deploying ACQL on a 6-DOF robotic arm performing a goal-reaching task in a cluttered, cabinet-like space with safety constraints. Our results demonstrate that ACQL is a robust and scalable solution for learning robotic behaviors according to rich temporal specifications. |
| 2025-10-06 | [Proactive defense against LLM Jailbreak](http://arxiv.org/abs/2510.05052v1) | Weiliang Zhao, Jinjun Peng et al. | The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks. |
| 2025-10-06 | [Imperceptible Jailbreaking against Large Language Models](http://arxiv.org/abs/2510.05025v1) | Kuofeng Gao, Yiming Li et al. | Jailbreaking attacks on the vision modality typically rely on imperceptible adversarial perturbations, whereas attacks on the textual modality are generally assumed to require visible modifications (e.g., non-semantic suffixes). In this paper, we introduce imperceptible jailbreaks that exploit a class of Unicode characters called variation selectors. By appending invisible variation selectors to malicious questions, the jailbreak prompts appear visually identical to original malicious questions on screen, while their tokenization is "secretly" altered. We propose a chain-of-search pipeline to generate such adversarial suffixes to induce harmful responses. Our experiments show that our imperceptible jailbreaks achieve high attack success rates against four aligned LLMs and generalize to prompt injection attacks, all without producing any visible modifications in the written prompt. Our code is available at https://github.com/sail-sg/imperceptible-jailbreaks. |
| 2025-10-06 | [Risk-Adjusted Policy Learning and the Social Cost of Uncertainty: Theory and Evidence from CAP evaluation](http://arxiv.org/abs/2510.05007v1) | Giovanni Cerulli, Francesco Caracciolo | This paper develops a risk-adjusted alternative to standard optimal policy learning (OPL) for observational data by importing Roy's (1952) safety-first principle into the treatment assignment problem. We formalize a welfare functional that maximizes the probability that outcomes exceed a socially required threshold and show that the associated pointwise optimal rule ranks treatments by the ratio of conditional means to conditional standard deviations. We implement the framework using microdata from the Italian Farm Accountancy Data Network to evaluate the allocation of subsidies under the EU Common Agricultural Policy. Empirically, risk-adjusted optimal policies systematically dominate the realized allocation across specifications, while risk aversion lowers overall welfare relative to the risk-neutral benchmark, making transparent the social cost of insurance against uncertainty. The results illustrate how safety-first OPL provides an implementable, interpretable tool for risk-sensitive policy design, quantifying the efficiency-insurance trade-off that policymakers face when outcomes are volatile. |
| 2025-10-06 | [Latent Uncertainty Representations for Video-based Driver Action and Intention Recognition](http://arxiv.org/abs/2510.05006v1) | Koen Vellenga, H. Joe Steinhauer et al. | Deep neural networks (DNNs) are increasingly applied to safety-critical tasks in resource-constrained environments, such as video-based driver action and intention recognition. While last layer probabilistic deep learning (LL-PDL) methods can detect out-of-distribution (OOD) instances, their performance varies. As an alternative to last layer approaches, we propose extending pre-trained DNNs with transformation layers to produce multiple latent representations to estimate the uncertainty. We evaluate our latent uncertainty representation (LUR) and repulsively trained LUR (RLUR) approaches against eight PDL methods across four video-based driver action and intention recognition datasets, comparing classification performance, calibration, and uncertainty-based OOD detection. We also contribute 28,000 frame-level action labels and 1,194 video-level intention labels for the NuScenes dataset. Our results show that LUR and RLUR achieve comparable in-distribution classification performance to other LL-PDL approaches. For uncertainty-based OOD detection, LUR matches top-performing PDL methods while being more efficient to train and easier to tune than approaches that require Markov-Chain Monte Carlo sampling or repulsive training procedures. |
| 2025-10-06 | [SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](http://arxiv.org/abs/2510.04891v1) | Punya Syon Pandey, Hai Son Le et al. | Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench. |
| 2025-10-06 | [RL Is a Hammer and LLMs Are Nails: A Simple Reinforcement Learning Recipe for Strong Prompt Injection](http://arxiv.org/abs/2510.04885v1) | Yuxin Wen, Arman Zharmagambetov et al. | Prompt injection poses a serious threat to the reliability and safety of LLM agents. Recent defenses against prompt injection, such as Instruction Hierarchy and SecAlign, have shown notable robustness against static attacks. However, to more thoroughly evaluate the robustness of these defenses, it is arguably necessary to employ strong attacks such as automated red-teaming. To this end, we introduce RL-Hammer, a simple recipe for training attacker models that automatically learn to perform strong prompt injections and jailbreaks via reinforcement learning. RL-Hammer requires no warm-up data and can be trained entirely from scratch. To achieve high ASRs against industrial-level models with defenses, we propose a set of practical techniques that enable highly effective, universal attacks. Using this pipeline, RL-Hammer reaches a 98% ASR against GPT-4o and a $72\%$ ASR against GPT-5 with the Instruction Hierarchy defense. We further discuss the challenge of achieving high diversity in attacks, highlighting how attacker models tend to reward-hack diversity objectives. Finally, we show that RL-Hammer can evade multiple prompt injection detectors. We hope our work advances automatic red-teaming and motivates the development of stronger, more principled defenses. Code is available at https://github.com/facebookresearch/rl-injector. |
| 2025-10-06 | [Less is More: Recursive Reasoning with Tiny Networks](http://arxiv.org/abs/2510.04871v1) | Alexia Jolicoeur-Martineau | Hierarchical Reasoning Model (HRM) is a novel approach using two small neural networks recursing at different frequencies. This biologically inspired method beats Large Language models (LLMs) on hard puzzle tasks such as Sudoku, Maze, and ARC-AGI while trained with small models (27M parameters) on small data (around 1000 examples). HRM holds great promise for solving hard problems with small networks, but it is not yet well understood and may be suboptimal. We propose Tiny Recursive Model (TRM), a much simpler recursive reasoning approach that achieves significantly higher generalization than HRM, while using a single tiny network with only 2 layers. With only 7M parameters, TRM obtains 45% test-accuracy on ARC-AGI-1 and 8% on ARC-AGI-2, higher than most LLMs (e.g., Deepseek R1, o3-mini, Gemini 2.5 Pro) with less than 0.01% of the parameters. |
| 2025-10-06 | [Read the Room: Inferring Social Context Through Dyadic Interaction Recognition in Cyber-physical-social Infrastructure Systems](http://arxiv.org/abs/2510.04854v1) | Cheyu Lin, John Martins et al. | Cyber-physical systems (CPS) integrate sensing, computing, and control to improve infrastructure performance, focusing on economic goals like performance and safety. However, they often neglect potential human-centered (or ''social'') benefits. Cyber-physical-social infrastructure systems (CPSIS) aim to address this by aligning CPS with social objectives. This involves defining social benefits, understanding human interactions with each other and infrastructure, developing privacy-preserving measurement methods, modeling these interactions for prediction, linking them to social benefits, and actuating the physical environment to foster positive social outcomes. This paper delves into recognizing dyadic human interactions using real-world data, which is the backbone to measuring social behavior. This lays a foundation to address the need to enhance understanding of the deeper meanings and mutual responses inherent in human interactions. While RGB cameras are informative for interaction recognition, privacy concerns arise. Depth sensors offer a privacy-conscious alternative by analyzing skeletal movements. This study compares five skeleton-based interaction recognition algorithms on a dataset of 12 dyadic interactions. Unlike single-person datasets, these interactions, categorized into communication types like emblems and affect displays, offer insights into the cultural and emotional aspects of human interactions. |
| 2025-10-06 | [An Active Fault-Tolerant Online Control Allocation Scheme for a Dual-System UAV in Transition Flight](http://arxiv.org/abs/2510.04853v1) | Junfeng Cai, Marco Lovera | A novel active fault-tolerant control (AFTC) scheme for a dual-system vertical takeoff and landing (VTOL) unmanned aerial vehicle (UAV) during transition flight is proposed in this paper. The AFTC scheme is composed of a baseline control law and an online control reallocation module. First, the structured $H_{\infty}$ baseline control law is able to guarantee the stability of closed-loop systems without being reconfigured under simultaneous actuator fault conditions. Second, compared to the existing mainstream method of sliding mode control that is a discontinuous control strategy, the AFTC scheme can effectively avoid control chattering problem by adopting the structured $H_{\infty}$ baseline control law. Third, an online control allocation (CA) module is implemented to carry out a unified CA for all the available actuators. When actuator faults/failures occur, the CA matrix is updated according to fault information and real-time airspeed, which is able to redistribute the virtual control signals to the remaining healthy actuators, avoiding significant performance degradation. Based on the developed AFTC scheme, symmetric and non-symmetric actuator fault scenarios are simulated on a nonlinear six-degree-of-freedom simulator, where the cases of merely structured $H_{\infty}$ control and structured $H_{\infty}$ based AFTC are compared and analyzed. The results show that the proposed structured $H_{\infty}$ based AFTC system is capable of handling more complicated fault scenarios and model uncertainties with no need to reconfigure the baseline control law. The proposed AFTC scheme significantly improves the safety and reliability of the transition flight of dual-system VTOL UAVs. |
| 2025-10-06 | [Did you just see that? Arbitrary view synthesis for egocentric replay of operating room workflows from ambient sensors](http://arxiv.org/abs/2510.04802v1) | Han Zhang, Lalithkumar Seenivasan et al. | Observing surgical practice has historically relied on fixed vantage points or recollections, leaving the egocentric visual perspectives that guide clinical decisions undocumented. Fixed-camera video can capture surgical workflows at the room-scale, but cannot reconstruct what each team member actually saw. Thus, these videos only provide limited insights into how decisions that affect surgical safety, training, and workflow optimization are made. Here we introduce EgoSurg, the first framework to reconstruct the dynamic, egocentric replays for any operating room (OR) staff directly from wall-mounted fixed-camera video, and thus, without intervention to clinical workflow. EgoSurg couples geometry-driven neural rendering with diffusion-based view enhancement, enabling high-visual fidelity synthesis of arbitrary and egocentric viewpoints at any moment. In evaluation across multi-site surgical cases and controlled studies, EgoSurg reconstructs person-specific visual fields and arbitrary viewpoints with high visual quality and fidelity. By transforming existing OR camera infrastructure into a navigable dynamic 3D record, EgoSurg establishes a new foundation for immersive surgical data science, enabling surgical practice to be visualized, experienced, and analyzed from every angle. |
| 2025-10-06 | [MPC strategies for density profile control with pellet fueling in nuclear fusion tokamaks under uncertainty](http://arxiv.org/abs/2510.04784v1) | Christopher A. Orrico, Hari Prasad Varadarajan et al. | Control of the density profile based on pellet fueling for the ITER nuclear fusion tokamak involves a multi-rate nonlinear system with safety-critical constraints, input delays, and discrete actuators with parametric uncertainty. To address this challenging problem, we propose a multi-stage MPC (msMPC) approach to handle uncertainty in the presence of mixed-integer inputs. While the scenario tree of msMPC accounts for uncertainty, it also adds complexity to an already computationally intensive mixed-integer MPC (MI-MPC) problem. To achieve real-time density profile controller with discrete pellets and uncertainty handling, we systematically reduce the problem complexity by (1) reducing the identified prediction model size through dynamic mode decomposition with control, (2) applying principal component analysis to reduce the number of scenarios needed to capture the parametric uncertainty in msMPC, and (3) utilizing the penalty term homotopy for MPC (PTH-MPC) algorithm to reduce the computational burden caused by the presence of mixed-integer inputs. We compare the performance and safety of the msMPC strategy against a nominal MI-MPC in plant simulations, demonstrating the first predictive density control strategy with uncertainty handling, viable for real-time pellet fueling in ITER. |
| 2025-10-06 | [Distribution Preference Optimization: A Fine-grained Perspective for LLM Unlearning](http://arxiv.org/abs/2510.04773v1) | Kai Qin, Jiaqi Wu et al. | As Large Language Models (LLMs) demonstrate remarkable capabilities learned from vast corpora, concerns regarding data privacy and safety are receiving increasing attention. LLM unlearning, which aims to remove the influence of specific data while preserving overall model utility, is becoming an important research area. One of the mainstream unlearning classes is optimization-based methods, which achieve forgetting directly through fine-tuning, exemplified by Negative Preference Optimization (NPO). However, NPO's effectiveness is limited by its inherent lack of explicit positive preference signals. Attempts to introduce such signals by constructing preferred responses often necessitate domain-specific knowledge or well-designed prompts, fundamentally restricting their generalizability. In this paper, we shift the focus to the distribution-level, directly targeting the next-token probability distribution instead of entire responses, and derive a novel unlearning algorithm termed \textbf{Di}stribution \textbf{P}reference \textbf{O}ptimization (DiPO). We show that the requisite preference distribution pairs for DiPO, which are distributions over the model's output tokens, can be constructed by selectively amplifying or suppressing the model's high-confidence output logits, thereby effectively overcoming NPO's limitations. We theoretically prove the consistency of DiPO's loss function with the desired unlearning direction. Extensive experiments demonstrate that DiPO achieves a strong trade-off between model utility and forget quality. Notably, DiPO attains the highest forget quality on the TOFU benchmark, and maintains leading scalability and sustainability in utility preservation on the MUSE benchmark. |
| 2025-10-06 | [On Prediction-Based Properties of Discrete-Event Systems: Notions, Applications and Supervisor Synthesis](http://arxiv.org/abs/2510.04616v1) | Bohan Cui, Yu Chen et al. | In this work, we investigate the problem of synthesizing property-enforcing supervisors for partially-observed discrete-event systems (DES). Unlike most existing approaches, where the enforced property depends solely on the executed behavior of the system, here we consider a more challenging scenario in which the property relies on predicted future behaviors that have not yet occurred. This problem arises naturally in applications involving future information, such as active prediction or intention protection. To formalize the problem, we introduce the notion of prediction-based properties, a new class of observational properties tied to the system's future information. We demonstrate that this notion is very generic and can model various practical properties, including predictability in fault prognosis and pre-opacity in intention security. We then present an effective approach for synthesizing supervisors that enforce prediction-based properties. Our method relies on a novel information structure that addresses the fundamental challenge arising from the dependency between current predictions and the control policy. The key idea is to first borrow information from future instants and then ensure information consistency. This reduces the supervisor synthesis problem to a safety game in the information space. We prove that the proposed algorithm is both sound and complete, and the resulting supervisor is maximally permissive. |
| 2025-10-06 | [Pathology-CoT: Learning Visual Chain-of-Thought Agent from Expert Whole Slide Image Diagnosis Behavior](http://arxiv.org/abs/2510.04587v1) | Sheng Wang, Ruiming Wu et al. | Diagnosing a whole-slide image is an interactive, multi-stage process involving changes in magnification and movement between fields. Although recent pathology foundation models are strong, practical agentic systems that decide what field to examine next, adjust magnification, and deliver explainable diagnoses are still lacking. The blocker is data: scalable, clinically aligned supervision of expert viewing behavior that is tacit and experience-based, not written in textbooks or online, and therefore absent from large language model training. We introduce the AI Session Recorder, which works with standard WSI viewers to unobtrusively record routine navigation and convert the viewer logs into standardized behavioral commands (inspect or peek at discrete magnifications) and bounding boxes. A lightweight human-in-the-loop review turns AI-drafted rationales into the Pathology-CoT dataset, a form of paired "where to look" and "why it matters" supervision produced at roughly six times lower labeling time. Using this behavioral data, we build Pathologist-o3, a two-stage agent that first proposes regions of interest and then performs behavior-guided reasoning. On gastrointestinal lymph-node metastasis detection, it achieved 84.5% precision, 100.0% recall, and 75.4% accuracy, exceeding the state-of-the-art OpenAI o3 model and generalizing across backbones. To our knowledge, this constitutes one of the first behavior-grounded agentic systems in pathology. Turning everyday viewer logs into scalable, expert-validated supervision, our framework makes agentic pathology practical and establishes a path to human-aligned, upgradeable clinical AI. |
| 2025-10-06 | [Tail-Safe Hedging: Explainable Risk-Sensitive Reinforcement Learning with a White-Box CBF--QP Safety Layer in Arbitrage-Free Markets](http://arxiv.org/abs/2510.04555v1) | Jian'an Zhang | We introduce Tail-Safe, a deployability-oriented framework for derivatives hedging that unifies distributional, risk-sensitive reinforcement learning with a white-box control-barrier-function (CBF) quadratic-program (QP) safety layer tailored to financial constraints. The learning component combines an IQN-based distributional critic with a CVaR objective (IQN--CVaR--PPO) and a Tail-Coverage Controller that regulates quantile sampling through temperature tilting and tail boosting to stabilize small-$\alpha$ estimation. The safety component enforces discrete-time CBF inequalities together with domain-specific constraints -- ellipsoidal no-trade bands, box and rate limits, and a sign-consistency gate -- solved as a convex QP whose telemetry (active sets, tightness, rate utilization, gate scores, slack, and solver status) forms an auditable trail for governance. We provide guarantees of robust forward invariance of the safe set under bounded model mismatch, a minimal-deviation projection interpretation of the QP, a KL-to-DRO upper bound linking per-state KL regularization to worst-case CVaR, concentration and sample-complexity results for the temperature-tilted CVaR estimator, and a CVaR trust-region improvement inequality under KL limits, together with feasibility persistence under expiry-aware tightening. Empirically, in arbitrage-free, microstructure-aware synthetic markets (SSVI $\to$ Dupire $\to$ VIX with ABIDES/MockLOB execution), Tail-Safe improves left-tail risk without degrading central performance and yields zero hard-constraint violations whenever the QP is feasible with zero slack. Telemetry is mapped to governance dashboards and incident workflows to support explainability and auditability. Limitations include reliance on synthetic data and simplified execution to isolate methodological contributions. |
| 2025-10-06 | [Unified Threat Detection and Mitigation Framework (UTDMF): Combating Prompt Injection, Deception, and Bias in Enterprise-Scale Transformers](http://arxiv.org/abs/2510.04528v1) | Santhosh KumarRavindran | The rapid adoption of large language models (LLMs) in enterprise systems exposes vulnerabilities to prompt injection attacks, strategic deception, and biased outputs, threatening security, trust, and fairness. Extending our adversarial activation patching framework (arXiv:2507.09406), which induced deception in toy networks at a 23.9% rate, we introduce the Unified Threat Detection and Mitigation Framework (UTDMF), a scalable, real-time pipeline for enterprise-grade models like Llama-3.1 (405B), GPT-4o, and Claude-3.5. Through 700+ experiments per model, UTDMF achieves: (1) 92% detection accuracy for prompt injection (e.g., jailbreaking); (2) 65% reduction in deceptive outputs via enhanced patching; and (3) 78% improvement in fairness metrics (e.g., demographic bias). Novel contributions include a generalized patching algorithm for multi-threat detection, three groundbreaking hypotheses on threat interactions (e.g., threat chaining in enterprise workflows), and a deployment-ready toolkit with APIs for enterprise integration. |
| 2025-10-06 | [Psychological Steering in LLMs: An Evaluation of Effectiveness and Trustworthiness](http://arxiv.org/abs/2510.04484v1) | Amin Banayeeanzade, Ala N. Tak et al. | The ability to control LLMs' emulated emotional states and personality traits is essential for enabling rich, human-centered interactions in socially interactive settings. We introduce PsySET, a Psychologically-informed benchmark to evaluate LLM Steering Effectiveness and Trustworthiness across the emotion and personality domains. Our study spans four models from different LLM families paired with various steering strategies, including prompting, fine-tuning, and representation engineering. Our results indicate that prompting is consistently effective but limited in intensity control, whereas vector injections achieve finer controllability while slightly reducing output quality. Moreover, we explore the trustworthiness of steered LLMs by assessing safety, truthfulness, fairness, and ethics, highlighting potential side effects and behavioral shifts. Notably, we observe idiosyncratic effects; for instance, even a positive emotion like joy can degrade robustness to adversarial factuality, lower privacy awareness, and increase preferential bias. Meanwhile, anger predictably elevates toxicity yet strengthens leakage resistance. Our framework establishes the first holistic evaluation of emotion and personality steering, offering insights into its interpretability and reliability for socially interactive applications. |
| 2025-10-06 | [On the Role of Unobserved Sequences on Sample-based Uncertainty Quantification for LLMs](http://arxiv.org/abs/2510.04439v1) | Lucie Kunitomo-Jacquin, Edison Marrese-Taylor et al. | Quantifying uncertainty in large language models (LLMs) is important for safety-critical applications because it helps spot incorrect answers, known as hallucinations. One major trend of uncertainty quantification methods is based on estimating the entropy of the distribution of the LLM's potential output sequences. This estimation is based on a set of output sequences and associated probabilities obtained by querying the LLM several times. In this paper, we advocate and experimentally show that the probability of unobserved sequences plays a crucial role, and we recommend future research to integrate it to enhance such LLM uncertainty quantification methods. |
| 2025-10-06 | [Investigating mixed traffic dynamics of pedestrians and non-motorized vehicles at urban intersections: Observation experiments and modelling](http://arxiv.org/abs/2510.04423v1) | Chaojia Yu, Kaixin Wang et al. | Urban intersections with mixed pedestrian and non-motorized vehicle traffic present complex safety challenges, yet traditional models fail to account for dynamic interactions arising from speed heterogeneity and collision anticipation. This study introduces the Time and Angle Based Social Force Model (TASFM), an enhanced framework extending the classical Social Force Model by integrating Time-to-Collision (TTC) metrics and velocity-angle-dependent tangential forces to simulate collision avoidance behaviors more realistically. Using aerial trajectory data from a high-density intersection in Shenzhen, China, we validated TASFM against real-world scenarios, achieving a Mean Trajectory Error (MTE) of 0.154 m (0.77% of the experimental area width). Key findings reveal distinct behavioral patterns: pedestrians self-organize into lanes along designated routes (e.g., zebra crossings), while non-motorized vehicles exhibit flexible path deviations that heighten collision risks. Simulations of three conflict types (overtaking, frontal/lateral crossing) demonstrate TASFM's capacity to replicate adaptive strategies like bidirectional path adjustments and speed modulation. The model provides actionable insights for urban planners, including conflict hotspot prediction and infrastructure redesign (e.g., segregated lanes), while offering a scalable framework for future research integrating motorized traffic and environmental variables. This work advances the understanding of mixed traffic dynamics and bridges the gap between theoretical modeling and data-driven urban safety solutions. |

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



