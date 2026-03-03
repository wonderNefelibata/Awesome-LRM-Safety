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
| 2026-03-02 | [Conformal Policy Control](http://arxiv.org/abs/2603.02196v1) | Drew Prinster, Clara Fannjiang et al. | An agent must try new behaviors to explore and improve. In high-stakes environments, an agent that violates safety constraints may cause harm and must be taken offline, curtailing any future interaction. Imitating old behavior is safe, but excessive conservatism discourages exploration. How much behavior change is too much? We show how to use any safe reference policy as a probabilistic regulator for any optimized but untested policy. Conformal calibration on data from the safe policy determines how aggressively the new policy can act, while provably enforcing the user's declared risk tolerance. Unlike conservative optimization methods, we do not assume the user has identified the correct model class nor tuned any hyperparameters. Unlike previous conformal methods, our theory provides finite-sample guarantees even for non-monotonic bounded constraint functions. Our experiments on applications ranging from natural language question answering to biomolecular engineering show that safe exploration is not only possible from the first moment of deployment, but can also improve performance. |
| 2026-03-02 | [From Leaderboard to Deployment: Code Quality Challenges in AV Perception Repositories](http://arxiv.org/abs/2603.02194v1) | Mateus Karvat, Bram Adams et al. | Autonomous vehicle (AV) perception models are typically evaluated solely on benchmark performance metrics, with limited attention to code quality, production readiness and long-term maintainability. This creates a significant gap between research excellence and real-world deployment in safety-critical systems subject to international safety standards. To address this gap, we present the first large-scale empirical study of software quality in AV perception repositories, systematically analyzing 178 unique models from the KITTI and NuScenes 3D Object Detection leaderboards. Using static analysis tools (Pylint, Bandit, and Radon), we evaluated code errors, security vulnerabilities, maintainability, and development practices. Our findings revealed that only 7.3% of the studied repositories meet basic production-readiness criteria, defined as having zero critical errors and no high-severity security vulnerabilities. Security issues are highly concentrated, with the top five issues responsible for almost 80% of occurrences, which prompted us to develop a set of actionable guidelines to prevent them. Additionally, the adoption of Continuous Integration/Continuous Deployment pipelines was correlated with better code maintainability. Our findings highlight that leaderboard performance does not reflect production readiness and that targeted interventions could substantially improve the quality and safety of AV perception code. |
| 2026-03-02 | [Boosting Device Utilization in Control Flow Auditing](http://arxiv.org/abs/2603.02161v1) | Alexandra Lengert, Adam Ilyas Caulfield et al. | Micro-Controller Units (MCUs) are widely used in safety-critical systems, making them attractive targets for attacks. This calls for lightweight defenses that remain effective despite software compromise. Control Flow Auditing (CFAud) is one such mechanism wherein a remote verifier (Vrf) is guaranteed to received evidence about the control flow path taken on a prover (Prv) MCU, even when Prv software is compromised. Despite promising benefits, current CFAud architectures unfortunately require a ``busy-wait'' phase where a hardware-anchored root-of-trust (RoT) in Prv retains execution control to ensure delivery of control flow evidence to Vrf. This drastically reduces the CPU utilization on Prv.   In this work, we addresses this limitation with an architecture for Contention Avoidance in Runtime Auditing with Minimized Execution Latency (CARAMEL). CARAMEL is a hardware-software RoT co-design that enables Prv applications to resume while control flow evidence is transmitted to Vrf. This significantly reduces contention due to transmission delays and improves CPU utilization without giving up on security. Key to CARAMEL is our design of a new RoT with a self-contained (and minimal) dedicated communication interface. CARAMEL's implementation and accompanying evaluation are made open-source. Our results show substantially improved CPU utilization at a modest hardware cost. |
| 2026-03-02 | [Recursive Think-Answer Process for LLMs and VLMs](http://arxiv.org/abs/2603.02099v1) | Byung-Kwan Lee, Youngchae Chee et al. | Think-Answer reasoners such as DeepSeek-R1 have made notable progress by leveraging interpretable internal reasoning. However, despite the frequent presence of self-reflective cues like "Oops!", they remain vulnerable to output errors during single-pass inference. To address this limitation, we propose an efficient Recursive Think-Answer Process (R-TAP) that enables models to engage in iterative reasoning cycles and generate more accurate answers, going beyond conventional single-pass approaches. Central to this approach is a confidence generator that evaluates the certainty of model responses and guides subsequent improvements. By incorporating two complementary rewards-Recursively Confidence Increase Reward and Final Answer Confidence Reward-we show that R-TAP-enhanced models consistently outperform conventional single-pass methods for both large language models (LLMs) and vision-language models (VLMs). Moreover, by analyzing the frequency of "Oops"-like expressions in model responses, we find that R-TAP-applied models exhibit significantly fewer self-reflective patterns, resulting in more stable and faster inference-time reasoning. We hope R-TAP pave the way evolving into efficient and elaborated methods to refine the reasoning processes of future AI. |
| 2026-03-02 | [ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels](http://arxiv.org/abs/2603.02097v1) | Xiang Zheng, Han Li et al. | Large language models (LLMs) are increasingly applied to health management, showing promise across disease prevention, clinical decision-making, and long-term care. However, existing medical benchmarks remain largely static and task-isolated, failing to capture the openness, longitudinal structure, and safety-critical complexity of real-world clinical workflows. We introduce ClinConsensus, a Chinese medical benchmark curated, validated, and quality-controlled by clinical experts. ClinConsensus comprises 2500 open-ended cases spanning the full continuum of care--from prevention and intervention to long-term follow-up--covering 36 medical specialties, 12 common clinical task types, and progressively increasing levels of complexity. To enable reliable evaluation of such complex scenarios, we adopt a rubric-based grading protocol and propose the Clinically Applicable Consistency Score (CACS@k). We further introduce a dual-judge evaluation framework, combining a high-capability LLM-as-judge with a distilled, locally deployable judge model trained via supervised fine-tuning, enabling scalable and reproducible evaluation aligned with physician judgment. Using ClinConsensus, we conduct a comprehensive assessment of several leading LLMs and reveal substantial heterogeneity across task themes, care stages, and medical specialties. While top-performing models achieve comparable overall scores, they differ markedly in reasoning, evidence use, and longitudinal follow-up capabilities, and clinically actionable treatment planning remains a key bottleneck. We release ClinConsensus as an extensible benchmark to support the development and evaluation of medical LLMs that are robust, clinically grounded, and ready for real-world deployment. |
| 2026-03-02 | [A Resource-Rational Principle for Modeling Visual Attention Control](http://arxiv.org/abs/2603.02056v1) | Yunpeng Bai | Understanding how people allocate visual attention is central to Human-Computer Interaction (HCI), yet existing computational models of attention are often either descriptive, task-specific, or difficult to interpret. My dissertation develops a resource-rational, simulation-based framework for modeling visual attention as a sequential decision-making process under perceptual, memory, and time constraints. I formalize visual tasks, such as reading and multitasking, as bounded-optimal control problems using Partially Observable Markov Decision Processes, enabling eye-movement behaviors such as fixation and attention switching to emerge from rational adaptation rather than being hand-coded or purely data-driven. These models are instantiated in simulation environments spanning traditional text reading and reading-while-walking with smart glasses, where they reproduce classic empirical effects, explain observed trade-offs between comprehension and safety, and generate novel predictions under time pressure and interface variation. Collectively, this work contributes a unified computational account of visual attention, offering new tools for theory-driven and resource-efficient HCI design. |
| 2026-03-02 | [Teen Vigilance: Navigating Risky Social Interactions on Discord](http://arxiv.org/abs/2603.02052v1) | Elena Koung, Yunhan Liu et al. | Teenagers are avid users of Discord, a fast growing platform for synchronous communication where they often interact with strangers. Because Discord combines private DMs, semi-private voice channels, and public servers in one place, it creates a hybrid environment that can produce complex and underexplored safety risks for teenagers. Drawing on 16 interviews with teenage Discord users, this study examines their strategies for navigating risky social interactions in the platform. Our findings reveal that when teenagers encounter risks during social interactions, they exercise vigilance by evaluating suspicious interactions before forming friendships, using safety tools, and engaging in controlled risk-taking to safeguard their privacy and security. At the community level, they mitigate risks through selective participation in servers, a practice supported by vigilant governance structures. We discuss how vigilance enables teenagers to act during risky encounters to protect themselves, advancing understanding of teenagers' agency in risk navigation and informing teen-centered designs for safer online environments. |
| 2026-03-02 | [CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies](http://arxiv.org/abs/2603.02004v1) | Gershom Seneviratne, Jianyu An et al. | Visuomotor navigation policies have shown strong perception-action coupling for embodied agents, yet they often struggle with safe navigation and dynamic obstacle avoidance in complex real-world environments. We introduce CHOP, a novel approach that leverages Counterfactual Human Preference Labels to align visuomotor navigation policies towards human intuition of safety and obstacle avoidance in navigation. In CHOP, for each visual observation, the robot's executed trajectory is included among a set of counterfactual navigation trajectories: alternative trajectories the robot could have followed under identical conditions. Human annotators provide pairwise preference labels over these trajectories based on anticipated outcomes such as collision risk and path efficiency. These aggregated preferences are then used to fine-tune visuomotor navigation policies, aligning their behavior with human preferences in navigation. Experiments on the SCAND dataset show that visuomotor navigation policies fine-tuned with CHOP reduce near-collision events by 49.7%, decrease deviation from human-preferred trajectories by 45.0%, and increase average obstacle clearance by 19.8% on average across multiple state-of-the-art models, compared to their pretrained baselines. These improvements transfer to real-world deployments on a Ghost Robotics Vision60 quadruped, where CHOP-aligned policies improve average goal success rates by 24.4%, increase minimum obstacle clearance by 6.8%, reduce collision and intervention events by 45.7%, and improve normalized path completion by 38.6% on average across navigation scenarios, compared to their pretrained baselines. Our results highlight the value of counterfactual preference supervision in bridging the gap between large-scale visuomotor policies and human-aligned, safety-aware embodied navigation. |
| 2026-03-02 | [Quantifying Uncertainty in Void Swelling Prediction: A Conformal Prediction Framework for Reactor Safety Margins](http://arxiv.org/abs/2603.01981v1) | Minhee Kim, Yong Yang | Irradiation-induced void swelling is a critical degradation mechanism for structural materials in nuclear reactors, dictating component operational lifespan and safety. While recent machine learning (ML) approaches have improved the accuracy of swelling rate predictions, they often fail to account for the inherent stochasticity of radiation damage, providing point estimates without rigorous uncertainty quantification. This lack of probabilistic context limits their applications in materials qualification, reactor licensing and risk assessment. In this work, we develop a framework that integrates ensemble ML models with Conformal Prediction (CP) to generate statistically calibrated prediction intervals. Unlike standard error estimation or Bayesian methods that often rely on rigid distributional assumptions, this approach specifically addresses the physical heteroscedasticity of swelling data, where variance transitions from the nucleation-dominated incubation regime to the growth-dominated steady-state regime. We demonstrate that log-transformed conformal prediction inference provides valid empirical coverage consistent with target confidence levels even in sparse data regimes. This framework offers a pathway to replace overly conservative upper-bound curves with Probabilistic Risk Assessment (PRA) tools for high-dose reactor core internals. |
| 2026-03-02 | [MobileMold: A Smartphone-Based Microscopy Dataset for Food Mold Detection](http://arxiv.org/abs/2603.01944v1) | Dinh Nam Pham, Leonard Prokisch et al. | Smartphone clip-on microscopes turn everyday devices into low-cost, portable imaging systems that can even reveal fungal structures at the microscopic level, enabling mold inspection beyond unaided visual checks. In this paper, we introduce MobileMold, an open smartphone-based microscopy dataset for food mold detection and food classification. MobileMold contains 4,941 handheld microscopy images spanning 11 food types, 4 smartphones, 3 microscopes, and diverse real-world conditions. Beyond the dataset release, we establish baselines for (i) mold detection and (ii) food-type classification, including a multi-task setting that predicts both attributes. Across multiple pretrained deep learning architectures and augmentation strategies, we obtain near-ceiling performance (accuracy = 0.9954, F1 = 0.9954, MCC = 0.9907), validating the utility of our dataset for detecting food spoilage. To increase transparency, we complement our evaluation with saliency-based visual explanations highlighting mold regions associated with the model's predictions. MobileMold aims to contribute to research on accessible food-safety sensing, mobile imaging, and exploring the potential of smartphones enhanced with attachments. |
| 2026-03-02 | [Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots](http://arxiv.org/abs/2603.01942v1) | Huw Day, Adrianna Jezierska et al. | Large Language Models have intensified the scale and strategic manipulation of political discourse on social media, leading to conflict escalation. The existing literature largely focuses on platform-led moderation as a countermeasure. In this paper, we propose a user-centric view of "jailbreaking" as an emergent, non-violent de-escalation practice. Online users engage with suspected LLM-powered accounts to circumvent large language model safeguards, exposing automated behaviour and disrupting the circulation of misleading narratives. |
| 2026-03-02 | [LaST-VLA: Thinking in Latent Spatio-Temporal Space for Vision-Language-Action in Autonomous Driving](http://arxiv.org/abs/2603.01928v1) | Yuechen Luo, Fang Li et al. | While Vision-Language-Action (VLA) models have revolutionized autonomous driving by unifying perception and planning, their reliance on explicit textual Chain-of-Thought (CoT) leads to semantic-perceptual decoupling and perceptual-symbolic conflicts. Recent shifts toward latent reasoning attempt to bypass these bottlenecks by thinking in continuous hidden space. However, without explicit intermediate constraints, standard latent CoT often operates as a physics-agnostic representation. To address this, we propose the Latent Spatio-Temporal VLA (LaST-VLA), a framework shifting the reasoning paradigm from discrete symbolic processing into a physically grounded Latent Spatio-Temporal CoT. By implementing a dual-feature alignment mechanism, we distill geometric constraints from 3D foundation models and dynamic foresight from world models directly into the latent space. Coupled with a progressive SFT training strategy that transitions from feature alignment to trajectory generation, and refined via Reinforcement Learning with Group Relative Policy Optimization (GRPO) to ensure safety and rule compliance. \method~setting a new record on NAVSIM v1 (91.3 PDMS) and NAVSIM v2 (87.1 EPDMS), while excelling in spatial-temporal reasoning on SURDS and NuDynamics benchmarks. |
| 2026-03-02 | [Radiation safety challenges in plasma accelerators](http://arxiv.org/abs/2603.01927v1) | S. Bohlen, M. Kirchen et al. | Plasma accelerators are rapidly evolving toward user-relevant machines with increasing repetition rates, particle energies and average beam powers. Despite their compact size, the operational characteristics of plasma accelerators are comparable to those of radio-frequency linacs, involving the continuous generation and dumping of electron bunches. However, beam properties and loss patterns can differ substantially from those of conventional accelerators, leading to radiation safety considerations dominated by high peak charges and distributed beam losses relevant for both personnel protection and machine integrity. Using established scaling laws, we show that significant dose rates already occur at electron energies of only a few tens of MeV, underscoring the relevance of radiation protection even for comparatively low-energy plasma accelerators. Based on a combination of Monte Carlo and particle-in-cell simulations, supported by radiation measurements from plasma accelerator experiments at DESY, we analyze typical radiation fields with a particular focus on radiation generated close to the plasma source. These findings highlight the need for dedicated shielding and beam-dump concepts tailored to plasma accelerators, especially in view of increasing average beam powers and future application-oriented operation. |
| 2026-03-02 | [Real Money, Fake Models: Deceptive Model Claims in Shadow APIs](http://arxiv.org/abs/2603.01919v1) | Yage Zhang, Yukun Jiang et al. | Access to frontier large language models (LLMs), such as GPT-5 and Gemini-2.5, is often hindered by high pricing, payment barriers, and regional restrictions. These limitations drive the proliferation of $\textit{shadow APIs}$, third-party services that claim to provide access to official model services without regional limitations via indirect access. Despite their widespread use, it remains unclear whether shadow APIs deliver outputs consistent with those of the official APIs, raising concerns about the reliability of downstream applications and the validity of research findings that depend on them. In this paper, we present the first systematic audit between official LLM APIs and corresponding shadow APIs. We first identify 17 shadow APIs that have been utilized in 187 academic papers, with the most popular one reaching 5,966 citations and 58,639 GitHub stars by December 6, 2025. Through multidimensional auditing of three representative shadow APIs across utility, safety, and model verification, we uncover both indirect and direct evidence of deception practices in shadow APIs. Specifically, we reveal performance divergence reaching up to $47.21\%$, significant unpredictability in safety behaviors, and identity verification failures in $45.83\%$ of fingerprint tests. These deceptive practices critically undermine the reproducibility and validity of scientific research, harm the interests of shadow API users, and damage the reputation of official model providers. |
| 2026-03-02 | [PAC Finite-Time Safety Guarantees for Stochastic Systems with Unknown Disturbance Distributions](http://arxiv.org/abs/2603.01918v1) | Taoran Wu, Dominik Wagner et al. | We investigate the problem of establishing finite-time probabilistic safety guarantees for discrete-time stochastic dynamical systems subject to unknown disturbance distributions, using barrier certificate methods. Our approach develops a data-driven safety certification framework that relies only on a finite collection of independent and identically distributed (i.i.d.) disturbance samples. Within this framework, we propose a certification procedure such that, with confidence at least $1-δ$ over the sampled disturbances, if the output of the certification procedure is accepted, the probability that the system remains within a prescribed safe set over a finite horizon is at least $1-ε$. A key challenge lies in formally characterizing the probably approximately correct (PAC) generalization behavior induced by finite samples. To address this, we derive PAC generalization bounds using tools from VC dimension, scenario optimization, and Rademacher complexity. These results illuminate the fundamental trade-offs between sample size, model complexity, and safety tolerance, providing both theoretical insight and practical guidance for designing reliable, data-driven safety certificates in discrete-time stochastic systems. |
| 2026-03-02 | [SaferPath: Hierarchical Visual Navigation with Learned Guidance and Safety-Constrained Control](http://arxiv.org/abs/2603.01898v1) | Lingjie Zhang, Zeyu Jiang et al. | Visual navigation is a core capability for mobile robots, yet end-to-end learning-based methods often struggle with generalization and safety in unseen, cluttered, or narrow environments. These limitations are especially pronounced in dense indoor settings, where collisions are likely and end-to-end models frequently fail. To address this, we propose SaferPath, a hierarchical visual navigation framework that leverages learned guidance from existing end-to-end models and refines it through a safety-constrained optimization-control module. SaferPath transforms visual observations into a traversable-area map and refines guidance trajectories using Model Predictive Stein Variational Evolution Strategy (MP-SVES), efficiently generating safe trajectories in only a few iterations. The refined trajectories are tracked by an MPC controller, ensuring robust navigation in complex environments. Extensive experiments in scenarios with unseen obstacles, dense unstructured spaces, and narrow corridors demonstrate that SaferPath consistently improves success rates and reduces collisions, outperforming representative baselines such as ViNT and NoMaD, and enabling safe navigation in challenging real-world settings. |
| 2026-03-02 | [Co-Evolutionary Multi-Modal Alignment via Structured Adversarial Evolution](http://arxiv.org/abs/2603.01784v1) | Guoxin Shi, Haoyu Wang et al. | Adversarial behavior plays a central role in aligning large language models with human values. However, existing alignment methods largely rely on static adversarial settings, which fundamentally limit robustness, particularly in multimodal settings with a larger attack surface. In this work, we move beyond static adversarial supervision and introduce co-evolutionary alignment with evolving attacks, instantiated by CEMMA (Co-Evolutionary Multi-Modal Alignment), an automated and adaptive framework for multimodal safety alignment. We introduce an Evolutionary Attacker that decomposes adversarial prompts into method templates and harmful intents. By employing genetic operators, including mutation, crossover, and differential evolution, it enables simple seed attacks to inherit the structural efficacy of sophisticated jailbreaks. The Adaptive Defender is iteratively updated on the synthesized hard negatives, forming a closed-loop process that adapts alignment to evolving attacks. Experiments show that the Evolutionary Attacker substantially increases red-teaming jailbreak attack success rate (ASR), while the Adaptive Defender improves robustness and generalization across benchmarks with higher data efficiency, without inducing excessive benign refusal, and remains compatible with inference-time defenses such as AdaShield. |
| 2026-03-02 | [A Safety-Aware Shared Autonomy Framework with BarrierIK Using Control Barrier Functions](http://arxiv.org/abs/2603.01705v1) | Berk Guler, Kay Pompetzki et al. | Shared autonomy blends operator intent with autonomous assistance. In cluttered environments, linear blending can produce unsafe commands even when each source is individually collision-free. Many existing approaches model obstacle avoidance through potentials or cost terms, which only enforce safety as a soft constraint. In contrast, safety-critical control requires hard guarantees. We investigate the use of control barrier functions (CBFs) at the inverse kinematics (IK) layer of shared autonomy, targeting post-blend safety while preserving task performance. Our approach is evaluated in simulation on representative cluttered environments and in a VR teleoperation study comparing pure teleoperation with shared autonomy. Across conditions, employing CBFs at the IK layer reduces violation time and increases minimum clearance while maintaining task performance. In the user study, participants reported higher perceived safety and trust, lower interference, and an overall preference for shared autonomy with our safety filter. Additional materials available at https://berkguler.github.io/barrierik. |
| 2026-03-02 | [Contract-based Agentic Intent Framework for Network Slicing in O-RAN](http://arxiv.org/abs/2603.01663v1) | Fransiscus Asisi Bimo, Chun-Kai Lai et al. | Intent-based networking aims to simplify network operation by translating operator intents into a collection of policies, configurations, and control actions. However, this translation process relies on heuristics and loose coupling. It often results in unpredictable behavior and ambiguous safety standards. This paper presents a Contract-based Agentic Intent Framework (CAIF) for the radio access network (RAN). The proposed framework employs a closed-loop agentic pipeline that systematically audits user objectives against formal RAN constraints prior to actuation. The proposed CAIF decouples probabilistic intent extraction from strictly governed policy execution to enable the enforcement of deterministic safety guarantees. We use network slicing as a representative use case to demonstrate the design flow and validate the effectiveness of the proposed approach on an O-RAN testbed. Experimental results show that the closed-loop agentic pipeline of the proposed CAIF can effectively eliminate harmful intent executions observed in direct-actuation baseline approaches. |
| 2026-03-02 | [LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence](http://arxiv.org/abs/2603.01651v1) | Anka Chandrahas Tummepalli, Preethu Rose Anish | Understanding and predicting judicial outcomes demands nuanced analysis of legal documents. Traditional approaches treat judgments and proceedings as unstructured text, limiting the effectiveness of large language models (LLMs) in tasks such as summarization, argument generation, and judgment prediction. We propose LexChronos, an agentic framework that iteratively extracts structured event timelines from Supreme Court of India judgments. LexChronos employs a dual-agent architecture: a LoRA-instruct-tuned extraction agent identifies candidate events, while a pre-trained feedback agent scores and refines them through a confidence-driven loop. To address the scarcity of Indian legal event datasets, we construct a synthetic corpus of 2000 samples using reverse-engineering techniques with DeepSeek-R1 and GPT-4, generating gold-standard event annotations. Our pipeline achieves a BERT-based F1 score of 0.8751 against this synthetic ground truth. In downstream evaluations on legal text summarization, GPT-4 preferred structured timelines over unstructured baselines in 75% of cases, demonstrating improved comprehension and reasoning in Indian jurisprudence. This work lays a foundation for future legal AI applications in the Indian context, such as precedent mapping, argument synthesis, and predictive judgment modelling, by harnessing structured representations of legal events. |

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



