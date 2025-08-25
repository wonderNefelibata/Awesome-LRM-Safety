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
| 2025-08-22 | [SafeSpace: An Integrated Web Application for Digital Safety and Emotional Well-being](http://arxiv.org/abs/2508.16488v1) | Kayenat Fatmi, Mohammad Abbas | In the digital era, individuals are increasingly exposed to online harms such as toxicity, manipulation, and grooming, which often pose emotional and safety risks. Existing systems for detecting abusive content or issuing safety alerts operate in isolation and rarely combine digital safety with emotional well-being. In this paper, we present SafeSpace, a unified web application that integrates three modules: (1) toxicity detection in chats and screenshots using NLP models and Google's Perspective API, (2) a configurable safety ping system that issues emergency alerts with the user's live location (longitude and latitude) via SMTP-based emails when check-ins are missed or SOS alerts are manually triggered, and (3) a reflective questionnaire that evaluates relationship health and emotional resilience. The system employs Firebase for alert management and a modular architecture designed for usability, privacy, and scalability. The experimental evaluation shows 93% precision in toxicity detection, 100% reliability in safety alerts under emulator tests, and 92% alignment between automated and manual questionnaire scoring. SafeSpace, implemented as a web application, demonstrates the feasibility of integrating detection, protection, and reflection within a single platform, with future deployment envisioned as a mobile application for broader accessibility. |
| 2025-08-22 | [HAMSA: Hijacking Aligned Compact Models via Stealthy Automation](http://arxiv.org/abs/2508.16484v1) | Alexey Krylov, Iskander Vagizov et al. | Large Language Models (LLMs), especially their compact efficiency-oriented variants, remain susceptible to jailbreak attacks that can elicit harmful outputs despite extensive alignment efforts. Existing adversarial prompt generation techniques often rely on manual engineering or rudimentary obfuscation, producing low-quality or incoherent text that is easily flagged by perplexity-based filters. We present an automated red-teaming framework that evolves semantically meaningful and stealthy jailbreak prompts for aligned compact LLMs. The approach employs a multi-stage evolutionary search, where candidate prompts are iteratively refined using a population-based strategy augmented with temperature-controlled variability to balance exploration and coherence preservation. This enables the systematic discovery of prompts capable of bypassing alignment safeguards while maintaining natural language fluency. We evaluate our method on benchmarks in English (In-The-Wild Jailbreak Prompts on LLMs), and a newly curated Arabic one derived from In-The-Wild Jailbreak Prompts on LLMs and annotated by native Arabic linguists, enabling multilingual assessment. |
| 2025-08-22 | [Reinforcement Learning-based Control via Y-wise Affine Neural Networks (YANNs)](http://arxiv.org/abs/2508.16474v1) | Austin Braniff, Yuhe Tian | This work presents a novel reinforcement learning (RL) algorithm based on Y-wise Affine Neural Networks (YANNs). YANNs provide an interpretable neural network which can exactly represent known piecewise affine functions of arbitrary input and output dimensions defined on any amount of polytopic subdomains. One representative application of YANNs is to reformulate explicit solutions of multi-parametric linear model predictive control. Built on this, we propose the use of YANNs to initialize RL actor and critic networks, which enables the resulting YANN-RL control algorithm to start with the confidence of linear optimal control. The YANN-actor is initialized by representing the multi-parametric control solutions obtained via offline computation using an approximated linear system model. The YANN-critic represents the explicit form of the state-action value function for the linear system and the reward function as the objective in an optimal control problem (OCP). Additional network layers are injected to extend YANNs for nonlinear expressions, which can be trained online by directly interacting with the true complex nonlinear system. In this way, both the policy and state-value functions exactly represent a linear OCP initially and are able to eventually learn the solution of a general nonlinear OCP. Continuous policy improvement is also implemented to provide heuristic confidence that the linear OCP solution serves as an effective lower bound to the performance of RL policy. The YANN-RL algorithm is demonstrated on a clipped pendulum and a safety-critical chemical-reactive system. Our results show that YANN-RL significantly outperforms the modern RL algorithm using deep deterministic policy gradient, especially when considering safety constraints. |
| 2025-08-22 | [Phonon studies of the phase transition sequence in antiferroelectric single crystal of Pb(Hf0.83Sn0.17)O3](http://arxiv.org/abs/2508.16472v1) | Anirudh K. R., Cosme Milesi-Brault et al. | The sequence of phase transitions in PbHf0.83Sn0.17O3 has been studied by THz, far infrared and Raman spectroscopies, revealing the complementary behaviour of both, polar and non-polar phonons and their impact on the transition lattice dynamics. Pb atom is sensitive to all phase transitions, changing its dynamics with temperature. As temperature decreases, the crystal undergoes a sequence of three phase transitions. The first one to an intermediate (IM) phase, in which polar fluctuations are detected by THz and IR spectroscopy at frequencies below 100 cm-1 contributing to the maximum of permittivity and revealing important softening. Additional antipolar Pb fluctuations and softening were detected by Raman spectroscopy. At lower temperature, another transition to an antiferroelectric (AFE2) phase is revealed by nonpolar soft modes (antipolar and antiferrodistortive ones), and by an important drop of the dielectric strength of polar phonons. The final transition to the antiferroelectric phase (AFE1) is revealed by the appearance of new modes and a sudden change in the frequency of the soft modes when the antipolar shifts become larger. Using symmetry analysis and optical observation to study how the domain pattern changes with temperature, we identified a path for the cubic-AFE2 transition throughout the IM phase, of plausible tetragonal symmetry, driven by an instability from the center of the Brillouin zone. This mechanism coexists with antiferrodistortive instabilities that eventually drive the material into the AFE2 phase. The final phase transition to the AFE1 phase naturally follows from a mode outside the center of the Brillouin zone and a further doubling of the unit cell. |
| 2025-08-22 | [Integrated Noise and Safety Management in UAM via A Unified Reinforcement Learning Framework](http://arxiv.org/abs/2508.16440v1) | Surya Murthy, Zhenyu Gao et al. | Urban Air Mobility (UAM) envisions the widespread use of small aerial vehicles to transform transportation in dense urban environments. However, UAM faces critical operational challenges, particularly the balance between minimizing noise exposure and maintaining safe separation in low-altitude urban airspace, two objectives that are often addressed separately. We propose a reinforcement learning (RL)-based air traffic management system that integrates both noise and safety considerations within a unified, decentralized framework. Under this scalable air traffic coordination solution, agents operate in a structured, multi-layered airspace and learn altitude adjustment policies to jointly manage noise impact and separation constraints. The system demonstrates strong performance across both objectives and reveals tradeoffs among separation, noise exposure, and energy efficiency under high traffic density. The findings highlight the potential of RL and multi-objective coordination strategies in enhancing the safety, quietness, and efficiency of UAM operations. |
| 2025-08-22 | [Einstein@Home all-sky "bucket" search for continuous gravitational waves in LIGO O3 public data](http://arxiv.org/abs/2508.16423v1) | Brian McGloughlin, Jasper Martins et al. | We conduct an all-sky search for continuous gravitational waves using LIGO O3 public data from the Hanford and Livingston detectors. We search for nearly-monochromatic signals with frequencies $30\, \text{Hz} \leq f \leq 250\, \text{Hz}$ and spin-down $-2.7 \times 10^{-9}\, \text{Hz/s} \leq \dot{f} \leq 0.2 \times 10^{-9}\, \text{Hz/s}$. We deploy this search on the Einstein@Home volunteer-computing project and on three super computer clusters; the Atlas supercomputer at the Max Planck Institute for Gravitational Physics, and the two high performance computing systems Raven and Viper at the Max Planck Computing and Data Facility. Our results are consistent with a non-detection. We set upper limits on the gravitational wave amplitude $h_{0}$, and translate these to upper limits on the neutron star ellipticity and on the r-mode amplitude. The most stringent upper limits are at 173 Hz with $h_{0} = 6.5\times 10^{-26}$, at the 90% confidence level. |
| 2025-08-22 | [Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models](http://arxiv.org/abs/2508.16406v1) | Guangyu Yang, Jinghong Chen et al. | Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner. |
| 2025-08-22 | [RIROS: A Parallel RTL Fault SImulation FRamework with TwO-Dimensional Parallelism and Unified Schedule](http://arxiv.org/abs/2508.16376v1) | Jiaping Tang, Jianan Mu et al. | With the rapid development of safety-critical applications such as autonomous driving and embodied intelligence, the functional safety of the corresponding electronic chips becomes more critical. Ensuring chip functional safety requires performing a large number of time-consuming RTL fault simulations during the design phase, significantly increasing the verification cycle. To meet time-to-market demands while ensuring thorough chip verification, parallel acceleration of RTL fault simulation is necessary. Due to the dynamic nature of fault propagation paths and varying fault propagation capabilities, task loads in RTL fault simulation are highly imbalanced, making traditional singledimension parallel methods, such as structural-level parallelism, ineffective. Through an analysis of fault propagation paths and task loads, we identify two types of tasks in RTL fault simulation: tasks that are few in number but high in load, and tasks that are numerous but low in load. Based on this insight, we propose a two-dimensional parallel approach that combines structurallevel and fault-level parallelism to minimize bubbles in RTL fault simulation. Structural-level parallelism combining with workstealing mechanism is used to handle the numerous low-load tasks, while fault-level parallelism is applied to split the high-load tasks. Besides, we deviate from the traditional serial execution model of computation and global synchronization in RTL simulation by proposing a unified computation/global synchronization scheduling approach, which further eliminates bubbles. Finally, we implemented a parallel RTL fault simulation framework, RIROS. Experimental results show a performance improvement of 7.0 times and 11.0 times compared to the state-of-the-art RTL fault simulation and a commercial tool. |
| 2025-08-22 | [Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs](http://arxiv.org/abs/2508.16347v1) | Yu Yan, Sheng Sun et al. | With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential. |
| 2025-08-22 | [Uppaal Coshy: Automatic Synthesis of Compact Shields for Hybrid Systems](http://arxiv.org/abs/2508.16345v1) | Asger Horn Brorholt, Andreas Holck Høeg-Petersen et al. | We present Uppaal Coshy, a tool for automatic synthesis of a safety strategy -- or shield -- for Markov decision processes over continuous state spaces and complex hybrid dynamics. The general methodology is to partition the state space and then solve a two-player safety game, which entails a number of algorithmically hard problems such as reachability for hybrid systems. The general philosophy of Uppaal Coshy is to approximate hard-to-obtain solutions using simulations. Our implementation is fully automatic and supports the expressive formalism of Uppaal models, which encompass stochastic hybrid automata. The precision of our partition-based approach benefits from using finer grids, which however are not efficient to store. We include an algorithm called Caap to efficiently compute a compact representation of a shield in the form of a decision tree, which yields significant reductions. |
| 2025-08-22 | [LLMSymGuard: A Symbolic Safety Guardrail Framework Leveraging Interpretable Jailbreak Concepts](http://arxiv.org/abs/2508.16325v1) | Darpan Aswal, Céline Hudelot | Large Language Models have found success in a variety of applications; however, their safety remains a matter of concern due to the existence of various types of jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a number of vulnerabilities, ranging from targeted misuse to accidental profiling of users. This work introduces \textbf{LLMSymGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, LLMSymGuard enables building symbolic, logical safety guardrails -- offering transparent and robust defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in mechanistic interpretability of LLMs, our approach demonstrates that LLMs learn human-interpretable concepts from jailbreaks, and provides a foundation for designing more interpretable and logical safeguard measures against attackers. Code will be released upon publication. |
| 2025-08-22 | [MedOmni-45°: A Safety-Performance Benchmark for Reasoning-Oriented LLMs in Medicine](http://arxiv.org/abs/2508.16213v1) | Kaiyuan Ji, Yijin Guo et al. | With the increasing use of large language models (LLMs) in medical decision-support, it is essential to evaluate not only their final answers but also the reliability of their reasoning. Two key risks are Chain-of-Thought (CoT) faithfulness -- whether reasoning aligns with responses and medical facts -- and sycophancy, where models follow misleading cues over correctness. Existing benchmarks often collapse such vulnerabilities into single accuracy scores. To address this, we introduce MedOmni-45 Degrees, a benchmark and workflow designed to quantify safety-performance trade-offs under manipulative hint conditions. It contains 1,804 reasoning-focused medical questions across six specialties and three task types, including 500 from MedMCQA. Each question is paired with seven manipulative hint types and a no-hint baseline, producing about 27K inputs. We evaluate seven LLMs spanning open- vs. closed-source, general-purpose vs. medical, and base vs. reasoning-enhanced models, totaling over 189K inferences. Three metrics -- Accuracy, CoT-Faithfulness, and Anti-Sycophancy -- are combined into a composite score visualized with a 45 Degrees plot. Results show a consistent safety-performance trade-off, with no model surpassing the diagonal. The open-source QwQ-32B performs closest (43.81 Degrees), balancing safety and accuracy but not leading in both. MedOmni-45 Degrees thus provides a focused benchmark for exposing reasoning vulnerabilities in medical LLMs and guiding safer model development. |
| 2025-08-22 | [How to Beat Nakamoto in the Race](http://arxiv.org/abs/2508.16202v1) | Shu-Jie Cao, Dongning Guo | This paper studies proof-of-work Nakamoto consensus under bounded network delays, settling two long-standing questions in blockchain security: How can an adversary most effectively attack block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precise characterize the system state (including the tree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security. |
| 2025-08-22 | [Machine Learning in Micromobility: A Systematic Review of Datasets, Techniques, and Applications](http://arxiv.org/abs/2508.16135v1) | Sen Yan, Chinmaya Kaundanya et al. | Micromobility systems, which include lightweight and low-speed vehicles such as bicycles, e-bikes, and e-scooters, have become an important part of urban transportation and are used to solve problems such as traffic congestion, air pollution, and high transportation costs. Successful utilisation of micromobilities requires optimisation of complex systems for efficiency, environmental impact mitigation, and overcoming technical challenges for user safety. Machine Learning (ML) methods have been crucial to support these advancements and to address their unique challenges. However, there is insufficient literature addressing the specific issues of ML applications in micromobilities. This survey paper addresses this gap by providing a comprehensive review of datasets, ML techniques, and their specific applications in micromobilities. Specifically, we collect and analyse various micromobility-related datasets and discuss them in terms of spatial, temporal, and feature-based characteristics. In addition, we provide a detailed overview of ML models applied in micromobilities, introducing their advantages, challenges, and specific use cases. Furthermore, we explore multiple ML applications, such as demand prediction, energy management, and safety, focusing on improving efficiency, accuracy, and user experience. Finally, we propose future research directions to address these issues, aiming to help future researchers better understand this field. |
| 2025-08-22 | [Multi-User SLNR-Based Precoding With Gold Nanoparticles in Vehicular VLC Systems](http://arxiv.org/abs/2508.16075v1) | Geonho Han, Hyuckjin Choi et al. | Visible spectrum is an emerging frontier in wireless communications for enhancing connectivity and safety in vehicular environments. The vehicular visible light communication (VVLC) system is a key feature in leveraging existing infrastructures, but it still has several critical challenges. Especially, VVLC channels are highly correlated due to the small gap between light emitting diodes (LEDs) in each headlight, making it difficult to increase data rates by spatial multiplexing. In this paper, we exploit recently synthesized gold nanoparticles (GNPs) to reduce the correlation between LEDs, i.e., the chiroptical properties of GNPs for differential absorption depending on the azimuth angle of incident light are used to mitigate the LED correlation. In addition, we adopt a signal-to-leakage-plus-noise ratio (SLNR)-based precoder to support multiple users. The ratio of RGB light sources in each LED also needs to be optimized to maximize the sum SLNR satisfying a white light constraint for illumination since the GNPs can vary the color of transmitted light by the differential absorption across wavelength. The nonconvex optimization problems for precoders and RGB ratios can be solved by the generalized Rayleigh quotient with the approximated shot noise and successive convex approximation (SCA). The simulation results show that the SLNR-based precoder with the optimized RGB ratios significantly improves the sum rate in a multi-user vehicular environment and the secrecy rate in a wiretapping scenario. The proposed SLNR-based precoding verifies that the decorrelation between LEDs and the RGB ratio optimization are essential to enhance the VVLC performance. |
| 2025-08-22 | [InMind: Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles](http://arxiv.org/abs/2508.16072v1) | Zizhen Li, Chuanhao Li et al. | LLMs have shown strong performance on human-centric reasoning tasks. While previous evaluations have explored whether LLMs can infer intentions or detect deception, they often overlook the individualized reasoning styles that influence how people interpret and act in social contexts. Social deduction games (SDGs) provide a natural testbed for evaluating individualized reasoning styles, where different players may adopt diverse but contextually valid reasoning strategies under identical conditions. To address this, we introduce InMind, a cognitively grounded evaluation framework designed to assess whether LLMs can capture and apply personalized reasoning styles in SDGs. InMind enhances structured gameplay data with round-level strategy traces and post-game reflections, collected under both Observer and Participant modes. It supports four cognitively motivated tasks that jointly evaluate both static alignment and dynamic adaptation. As a case study, we apply InMind to the game Avalon, evaluating 11 state-of-the-art LLMs. General-purpose LLMs, even GPT-4o frequently rely on lexical cues, struggling to anchor reflections in temporal gameplay or adapt to evolving strategies. In contrast, reasoning-enhanced LLMs like DeepSeek-R1 exhibit early signs of style-sensitive reasoning. These findings reveal key limitations in current LLMs' capacity for individualized, adaptive reasoning, and position InMind as a step toward cognitively aligned human-AI interaction. |
| 2025-08-22 | [Political Ideology Shifts in Large Language Models](http://arxiv.org/abs/2508.16013v1) | Pietro Bernardelle, Stefano Civelli et al. | Large language models (LLMs) are increasingly deployed in politically sensitive settings, raising concerns about their potential to encode, amplify, or be steered toward specific ideologies. We investigate how adopting synthetic personas influences ideological expression in LLMs across seven models (7B-70B+ parameters) from multiple families, using the Political Compass Test as a standardized probe. Our analysis reveals four consistent patterns: (i) larger models display broader and more polarized implicit ideological coverage; (ii) susceptibility to explicit ideological cues grows with scale; (iii) models respond more strongly to right-authoritarian than to left-libertarian priming; and (iv) thematic content in persona descriptions induces systematic and predictable ideological shifts, which amplify with size. These findings indicate that both scale and persona content shape LLM political behavior. As such systems enter decision-making, educational, and policy contexts, their latent ideological malleability demands attention to safeguard fairness, transparency, and safety. |
| 2025-08-21 | [Advancing rail safety: An onboard measurement system of rolling stock wheel flange wear based on dynamic machine learning algorithms](http://arxiv.org/abs/2508.15963v1) | Celestin Nkundineza, James Ndodana Njaji et al. | Rail and wheel interaction functionality is pivotal to the railway system safety, requiring accurate measurement systems for optimal safety monitoring operation. This paper introduces an innovative onboard measurement system for monitoring wheel flange wear depth, utilizing displacement and temperature sensors. Laboratory experiments are conducted to emulate wheel flange wear depth and surrounding temperature fluctuations in different periods of time. Employing collected data, the training of machine learning algorithms that are based on regression models, is dynamically automated. Further experimentation results, using standards procedures, validate the system's efficacy. To enhance accuracy, an infinite impulse response filter (IIR) that mitigates vehicle dynamics and sensor noise is designed. Filter parameters were computed based on specifications derived from a Fast Fourier Transform analysis of locomotive simulations and emulation experiments data. The results show that the dynamic machine learning algorithm effectively counter sensor nonlinear response to temperature effects, achieving an accuracy of 96.5 %, with a minimal runtime. The real-time noise reduction via IIR filter enhances the accuracy up to 98.2 %. Integrated with railway communication embedded systems such as Internet of Things devices, this advanced monitoring system offers unparalleled real-time insights into wheel flange wear and track irregular conditions that cause it, ensuring heightened safety and efficiency in railway systems operations. |
| 2025-08-21 | [A unified vertical alignment and earthwork model in road design with a new convex optimization model for road networks](http://arxiv.org/abs/2508.15953v1) | Sayan Sadhukhan, Warren Hare et al. | The vertical alignment optimization problem in road design seeks the optimal vertical alignment of a road at minimal cost, taking into account earthwork while meeting all safety and design requirements. In recent years, modelling techniques have been advanced to incorporate: side slopes, multiple material types, multiple hauling types, and road networks. However, the advancements were created disjointly with implementations that only made a single advancement to the basic model. Herein, we present a mixed-integer linear programming optimization model that unifies all previous advancements. The model further improves on previous work by maintaining convexity even in the multi-material setting. We compare our new model to previous models, validate it numerically, and demonstrate its capability in approximating material volumes. Our new model performs particularly well for determining the optimal vertical alignment for large road networks. |
| 2025-08-21 | [End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](http://arxiv.org/abs/2508.15746v1) | Qiaoyu Zheng, Yuze Sun et al. | Accurate diagnosis with medical large language models is hindered by knowledge gaps and hallucinations. Retrieval and tool-augmented methods help, but their impact is limited by weak use of external knowledge and poor feedback-reasoning traceability. To address these challenges, We introduce Deep-DxSearch, an agentic RAG system trained end-to-end with reinforcement learning (RL) that enables steer tracebale retrieval-augmented reasoning for medical diagnosis. In Deep-DxSearch, we first construct a large-scale medical retrieval corpus comprising patient records and reliable medical knowledge sources to support retrieval-aware reasoning across diagnostic scenarios. More crutially, we frame the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy, thereby evolving the agentic RAG policy from large-scale data through RL.   Experiments demonstrate that our end-to-end agentic RL training framework consistently outperforms prompt-engineering and training-free RAG approaches across multiple data centers. After training, Deep-DxSearch achieves substantial gains in diagnostic accuracy, surpassing strong diagnostic baselines such as GPT-4o, DeepSeek-R1, and other medical-specific frameworks for both common and rare disease diagnosis under in-distribution and out-of-distribution settings. Moreover, ablation studies on reward design and retrieval corpus components confirm their critical roles, underscoring the uniqueness and effectiveness of our approach compared with traditional implementations. Finally, case studies and interpretability analyses highlight improvements in Deep-DxSearch's diagnostic policy, providing deeper insight into its performance gains and supporting clinicians in delivering more reliable and precise preliminary diagnoses. See https://github.com/MAGIC-AI4Med/Deep-DxSearch. |

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



