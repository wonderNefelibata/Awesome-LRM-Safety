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
| 2025-10-17 | [Braking within Barriers: Constructive Safety-Critical Control for Input-Constrained Vehicles via the Backup Set Method](http://arxiv.org/abs/2510.15797v1) | Laszlo Gacsi, Adam K. Kiss et al. | This paper presents a safety-critical control framework to maintain bounded lateral motions for vehicles braking on asymmetric surfaces. We synthesize a brake controller that assists drivers and guarantees safety against excessive lateral motions (i.e., prevents the vehicle from spinning out) while minimizing the stopping distance. We address this safety-critical control problem in the presence of input constraints, since braking forces are limited by the available friction on the road. We use backup control barrier functions for safe control design. As this approach requires the construction of a backup set and a backup controller, we propose a novel, systematic method to creating valid backup set-backup controller pairs based on feedback linearization and continuous-time Lyapunov equations. We use simple examples to demonstrate our proposed safety-critical control method. Finally, we implement our approach on a four-wheel vehicle model for braking on asymmetric surfaces and present simulation results. |
| 2025-10-17 | [On Non-interactive Evaluation of Animal Communication Translators](http://arxiv.org/abs/2510.15768v1) | Orr Paradise, David F. Gruber et al. | If you had an AI Whale-to-English translator, how could you validate whether or not it is working? Does one need to interact with the animals or rely on grounded observations such as temperature? We provide theoretical and proof-of-concept experimental evidence suggesting that interaction and even observations may not be necessary for sufficiently complex languages. One may be able to evaluate translators solely by their English outputs, offering potential advantages in terms of safety, ethics, and cost. This is an instance of machine translation quality evaluation (MTQE) without any reference translations available. A key challenge is identifying ``hallucinations,'' false translations which may appear fluent and plausible. We propose using segment-by-segment translation together with the classic NLP shuffle test to evaluate translators. The idea is to translate animal communication, turn by turn, and evaluate how often the resulting translations make more sense in order than permuted. Proof-of-concept experiments on data-scarce human languages and constructed languages demonstrate the potential utility of this evaluation methodology. These human-language experiments serve solely to validate our reference-free metric under data scarcity. It is found to correlate highly with a standard evaluation based on reference translations, which are available in our experiments. We also perform a theoretical analysis suggesting that interaction may not be necessary nor efficient in the early stages of learning to translate. |
| 2025-10-17 | [Grassroots Logic Programs: A Secure, Multiagent, Concurrent, Logic Programming Language](http://arxiv.org/abs/2510.15747v1) | Ehud Shapiro | Grassroots platforms are distributed applications run by\linebreak cryptographically-identified people on their networked personal devices, where multiple disjoint platform instances emerge independently and coalesce when they interoperate. Their foundation is the grassroots social graph, upon which grassroots social networks, grassroots cryptocurrencies, and grassroots democratic federations can be built.   Grassroots platforms have yet to be implemented, the key challenge being faulty and malicious participants: without secure programming support, correct participants cannot reliably identify each other, establish secure communication, or verify each other's code integrity.   We present Grassroots Logic Programs (GLP), a secure, multiagent, concurrent, logic programming language for implementing grassroots platforms. GLP extends logic programs with paired single-reader/single-writer (SRSW) logic variables, providing secure communication channels among cryptographically-identified people through encrypted, signed and attested messages, which enable identity and code integrity verification. We present GLP progressively: logic programs, concurrent GLP, multiagent GLP, augmenting it with cryptographic security, and providing smartphone implementation-ready specifications. We prove safety properties including that GLP computations are deductions, SRSW preservation, acyclicity, and monotonicity. We prove multiagent GLP is grassroots and that GLP streams achieve blockchain security properties. We present a grassroots social graph protocol establishing authenticated peer-to-peer connections and demonstrate secure grassroots social networking applications. |
| 2025-10-17 | [ProSh: Probabilistic Shielding for Model-free Reinforcement Learning](http://arxiv.org/abs/2510.15720v1) | Edwin Hamel-De le Court, Gaspard Ohlmann et al. | Safety is a major concern in reinforcement learning (RL): we aim at developing RL systems that not only perform optimally, but are also safe to deploy by providing formal guarantees about their safety. To this end, we introduce Probabilistic Shielding via Risk Augmentation (ProSh), a model-free algorithm for safe reinforcement learning under cost constraints. ProSh augments the Constrained MDP state space with a risk budget and enforces safety by applying a shield to the agent's policy distribution using a learned cost critic. The shield ensures that all sampled actions remain safe in expectation. We also show that optimality is preserved when the environment is deterministic. Since ProSh is model-free, safety during training depends on the knowledge we have acquired about the environment. We provide a tight upper-bound on the cost in expectation, depending only on the backup-critic accuracy, that is always satisfied during training. Under mild, practically achievable assumptions, ProSh guarantees safety even at training time, as shown in the experiments. |
| 2025-10-17 | [Integration of a Variable Stiffness Link for Long-Reach Aerial Manipulation](http://arxiv.org/abs/2510.15639v1) | Manuel J. Fernandez, Alejandro Suarez et al. | This paper presents the integration of a Variable Stiffness Link (VSL) for long-reach aerial manipulation, enabling adaptable mechanical coupling between an aerial multirotor platform and a dual-arm manipulator. Conventional long-reach manipulation systems rely on rigid or cable connections, which limit precision or transmit disturbances to the aerial vehicle. The proposed VSL introduces an adjustable stiffness mechanism that allows the link to behave either as a flexible rope or as a rigid rod, depending on task requirements.   The system is mounted on a quadrotor equipped with the LiCAS dual-arm manipulator and evaluated through teleoperated experiments, involving external disturbances and parcel transportation tasks. Results demonstrate that varying the link stiffness significantly modifies the dynamic interaction between the UAV and the payload. The flexible configuration attenuates external impacts and aerodynamic perturbations, while the rigid configuration improves positional accuracy during manipulation phases.   These results confirm that VSL enhances versatility and safety, providing a controllable trade-off between compliance and precision. Future work will focus on autonomous stiffness regulation, multi-rope configurations, cooperative aerial manipulation and user studies to further assess its impact on teleoperated and semi-autonomous aerial tasks. |
| 2025-10-17 | [Hypergame-based Cognition Modeling and Intention Interpretation for Human-Driven Vehicles in Connected Mixed Traffic](http://arxiv.org/abs/2510.15573v1) | Jianguo Chen, Zhengqin Liu et al. | With the practical implementation of connected and autonomous vehicles (CAVs), the traffic system is expected to remain a mix of CAVs and human-driven vehicles (HVs) for the foreseeable future. To enhance safety and traffic efficiency, the trajectory planning strategies of CAVs must account for the influence of HVs, necessitating accurate HV trajectory prediction. Current research often assumes that human drivers have perfect knowledge of all vehicles' objectives, an unrealistic premise. This paper bridges the gap by leveraging hypergame theory to account for cognitive and perception limitations in HVs. We model human bounded rationality without assuming them to be merely passive followers and propose a hierarchical cognition modeling framework that captures cognitive relationships among vehicles. We further analyze the cognitive stability of the system, proving that the strategy profile where all vehicles adopt cognitively equilibrium strategies constitutes a hyper Nash equilibrium when CAVs accurately learn HV parameters. To achieve this, we develop an inverse learning algorithm for distributed intention interpretation via vehicle-to-everything (V2X) communication, which extends the framework to both offline and online scenarios. Additionally, we introduce a distributed trajectory prediction and planning approach for CAVs, leveraging the learned parameters in real time. Simulations in highway lane-changing scenarios demonstrate the proposed method's accuracy in parameter learning, robustness to noisy trajectory observations, and safety in HV trajectory prediction. The results validate the effectiveness of our method in both offline and online implementations. |
| 2025-10-17 | [Hypergraph Contrastive Sensor Fusion for Multimodal Fault Diagnosis in Induction Motors](http://arxiv.org/abs/2510.15547v1) | Usman Ali, Ali Zia et al. | Reliable induction motor (IM) fault diagnosis is vital for industrial safety and operational continuity, mitigating costly unplanned downtime. Conventional approaches often struggle to capture complex multimodal signal relationships, are constrained to unimodal data or single fault types, and exhibit performance degradation under noisy or cross-domain conditions. This paper proposes the Multimodal Hypergraph Contrastive Attention Network (MM-HCAN), a unified framework for robust fault diagnosis. To the best of our knowledge, MM-HCAN is the first to integrate contrastive learning within a hypergraph topology specifically designed for multimodal sensor fusion, enabling the joint modelling of intra- and inter-modal dependencies and enhancing generalisation beyond Euclidean embedding spaces. The model facilitates simultaneous diagnosis of bearing, stator, and rotor faults, addressing the engineering need for consolidated di- agnostic capabilities. Evaluated on three real-world benchmarks, MM-HCAN achieves up to 99.82% accuracy with strong cross-domain generalisation and resilience to noise, demonstrating its suitability for real-world deployment. An ablation study validates the contribution of each component. MM-HCAN provides a scalable and robust solution for comprehensive multi-fault diagnosis, supporting predictive maintenance and extended asset longevity in industrial environments. |
| 2025-10-17 | [HarmRLVR: Weaponizing Verifiable Rewards for Harmful LLM Alignment](http://arxiv.org/abs/2510.15499v1) | Yuexiao Liu, Lijun Li et al. | Recent advancements in Reinforcement Learning with Verifiable Rewards (RLVR) have gained significant attention due to their objective and verifiable reward signals, demonstrating strong performance in reasoning and code generation tasks. However, the potential safety risks associated with RLVR remain underexplored. This paper presents HarmRLVR, the first systematic investigation into the alignment reversibility risk of RLVR. We show that safety alignment can be rapidly reversed using GRPO with merely 64 harmful prompts without responses, causing models to readily comply with harmful instructions. Across five models from Llama, Qwen, and DeepSeek, we empirically demonstrate that RLVR-based attacks elevate the average harmfulness score to 4.94 with an attack success rate of 96.01\%, significantly outperforming harmful fine-tuning while preserving general capabilities. Our findings reveal that RLVR can be efficiently exploited for harmful alignment, posing serious threats to open-source model safety. Please see our code at https://github.com/lyxx2535/HarmRLVR. |
| 2025-10-17 | [SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models](http://arxiv.org/abs/2510.15476v1) | Hanbin Hong, Shuya Feng et al. | Large Language Models (LLMs) have rapidly become integral to real-world applications, powering services across diverse sectors. However, their widespread deployment has exposed critical security risks, particularly through jailbreak prompts that can bypass model alignment and induce harmful outputs. Despite intense research into both attack and defense techniques, the field remains fragmented: definitions, threat models, and evaluation criteria vary widely, impeding systematic progress and fair comparison. In this Systematization of Knowledge (SoK), we address these challenges by (1) proposing a holistic, multi-level taxonomy that organizes attacks, defenses, and vulnerabilities in LLM prompt security; (2) formalizing threat models and cost assumptions into machine-readable profiles for reproducible evaluation; (3) introducing an open-source evaluation toolkit for standardized, auditable comparison of attacks and defenses; (4) releasing JAILBREAKDB, the largest annotated dataset of jailbreak and benign prompts to date; and (5) presenting a comprehensive evaluation and leaderboard of state-of-the-art methods. Our work unifies fragmented research, provides rigorous foundations for future studies, and supports the development of robust, trustworthy LLMs suitable for high-stakes deployment. |
| 2025-10-17 | [Semantic4Safety: Causal Insights from Zero-shot Street View Imagery Segmentation for Urban Road Safety](http://arxiv.org/abs/2510.15434v1) | Huan Chen, Ting Han et al. | Street-view imagery (SVI) offers a fine-grained lens on traffic risk, yet two fundamental challenges persist: (1) how to construct street-level indicators that capture accident-related features, and (2) how to quantify their causal impacts across different accident types. To address these challenges, we propose Semantic4Safety, a framework that applies zero-shot semantic segmentation to SVIs to derive 11 interpretable streetscape indicators, and integrates road type as contextual information to analyze approximately 30,000 accident records in Austin. Specifically, we train an eXtreme Gradient Boosting (XGBoost) multi-class classifier and use Shapley Additive Explanations (SHAP) to interpret both global and local feature contributions, and then apply Generalized Propensity Score (GPS) weighting and Average Treatment Effect (ATE) estimation to control confounding and quantify causal effects. Results uncover heterogeneous, accident-type-specific causal patterns: features capturing scene complexity, exposure, and roadway geometry dominate predictive power; larger drivable area and emergency space reduce risk, whereas excessive visual openness can increase it. By bridging predictive modeling with causal inference, Semantic4Safety supports targeted interventions and high-risk corridor diagnosis, offering a scalable, data-informed tool for urban road safety planning. |
| 2025-10-17 | [Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models](http://arxiv.org/abs/2510.15430v1) | Shuang Liang, Zhihao Xu et al. | Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB. |
| 2025-10-17 | [Safe, Efficient, and Robust Reinforcement Learning for Ranking and Diffusion Models](http://arxiv.org/abs/2510.15429v1) | Shashank Gupta | This dissertation investigates how reinforcement learning (RL) methods can be designed to be safe, sample-efficient, and robust. Framed through the unifying perspective of contextual-bandit RL, the work addresses two major application domains - ranking and recommendation, and text-to-image diffusion models. The first part of the thesis develops theory and algorithms for safe deployment in ranking systems. An exposure-based generalisation bound is derived, leading to a counterfactual risk-minimisation objective whose solution is guaranteed not to underperform the logging policy, even with sparse feedback. This guarantee is extended to doubly robust estimators, enabling safety even under adversarial or misspecified user models and offering practitioners explicit control over permissible utility loss. The second part turns to single-action bandits, where various off-policy estimators are unified within a baseline-correction framework. A closed-form optimal baseline is proposed and shown to minimise both evaluation and policy-gradient variance, thereby improving off-policy learning reliability. The final part examines the trade-offs between efficiency and effectiveness in generative RL. A systematic study of PPO and REINFORCE motivates the Leave-One-Out PPO (LOOP) algorithm, which combines multiple diffusion trajectories with a REINFORCE-style baseline inside PPO's clipped objective. LOOP achieves PPO-level sample efficiency while producing generations that align more faithfully with textual attributes. |
| 2025-10-17 | [Corrigibility Transformation: Constructing Goals That Accept Updates](http://arxiv.org/abs/2510.15395v1) | Rubi Hudson | For an AI's training process to successfully impart a desired goal, it is important that the AI does not attempt to resist the training. However, partially learned goals will often incentivize an AI to avoid further goal updates, as most goals are better achieved by an AI continuing to pursue them. We say that a goal is corrigible if it does not incentivize taking actions that avoid proper goal updates or shutdown. In addition to convergence in training, corrigibility also allows for correcting mistakes and changes in human preferences, which makes it a crucial safety property. Despite this, the existing literature does not include specifications for goals that are both corrigible and competitive with non-corrigible alternatives. We provide a formal definition for corrigibility, then introduce a transformation that constructs a corrigible version of any goal that can be made corrigible, without sacrificing performance. This is done by myopically eliciting predictions of reward conditional on costlessly preventing updates, which then also determine the reward when updates are accepted. The transformation can be modified to recursively extend corrigibility to any new agents created by corrigible agents, and to prevent agents from deliberately modifying their goals. Two gridworld experiments demonstrate that these corrigible goals can be learned effectively, and that they lead to the desired behavior. |
| 2025-10-17 | [Towards Automated Chicken Deboning via Learning-based Dynamically-Adaptive 6-DoF Multi-Material Cutting](http://arxiv.org/abs/2510.15376v1) | Zhaodong Yang, Ai-Ping Hu et al. | Automating chicken shoulder deboning requires precise 6-DoF cutting through a partially occluded, deformable, multi-material joint, since contact with the bones presents serious health and safety risks. Our work makes both systems-level and algorithmic contributions to train and deploy a reactive force-feedback cutting policy that dynamically adapts a nominal trajectory and enables full 6-DoF knife control to traverse the narrow joint gap while avoiding contact with the bones. First, we introduce an open-source custom-built simulator for multi-material cutting that models coupling, fracture, and cutting forces, and supports reinforcement learning, enabling efficient training and rapid prototyping. Second, we design a reusable physical testbed to emulate the chicken shoulder: two rigid "bone" spheres with controllable pose embedded in a softer block, enabling rigorous and repeatable evaluation while preserving essential multi-material characteristics of the target problem. Third, we train and deploy a residual RL policy, with discretized force observations and domain randomization, enabling robust zero-shot sim-to-real transfer and the first demonstration of a learned policy that debones a real chicken shoulder. Our experiments in our simulator, on our physical testbed, and on real chicken shoulders show that our learned policy reliably navigates the joint gap and reduces undesired bone/cartilage contact, resulting in up to a 4x improvement over existing open-loop cutting baselines in terms of success rate and bone avoidance. Our results also illustrate the necessity of force feedback for safe and effective multi-material cutting. The project website is at https://sites.google.com/view/chickendeboning-2026. |
| 2025-10-17 | [VERA-MH Concept Paper](http://arxiv.org/abs/2510.15297v1) | Luca Belli, Kate Bentley et al. | We introduce VERA-MH (Validation of Ethical and Responsible AI in Mental Health), an automated evaluation of the safety of AI chatbots used in mental health contexts, with an initial focus on suicide risk.   Practicing clinicians and academic experts developed a rubric informed by best practices for suicide risk management for the evaluation. To fully automate the process, we used two ancillary AI agents. A user-agent model simulates users engaging in a mental health-based conversation with the chatbot under evaluation. The user-agent role-plays specific personas with pre-defined risk levels and other features. Simulated conversations are then passed to a judge-agent who scores them based on the rubric. The final evaluation of the chatbot being tested is obtained by aggregating the scoring of each conversation.   VERA-MH is actively under development and undergoing rigorous validation by mental health clinicians to ensure user-agents realistically act as patients and that the judge-agent accurately scores the AI chatbot. To date we have conducted preliminary evaluation of GPT-5, Claude Opus and Claude Sonnet using initial versions of the VERA-MH rubric and used the findings for further design development. Next steps will include more robust clinical validation and iteration, as well as refining actionable scoring. We are seeking feedback from the community on both the technical and clinical aspects of our evaluation. |
| 2025-10-17 | [FinTrust: A Comprehensive Benchmark of Trustworthiness Evaluation in Finance Domain](http://arxiv.org/abs/2510.15232v1) | Tiansheng Hu, Tongyan Hu et al. | Recent LLMs have demonstrated promising ability in solving finance related problems. However, applying LLMs in real-world finance application remains challenging due to its high risk and high stakes property. This paper introduces FinTrust, a comprehensive benchmark specifically designed for evaluating the trustworthiness of LLMs in finance applications. Our benchmark focuses on a wide range of alignment issues based on practical context and features fine-grained tasks for each dimension of trustworthiness evaluation. We assess eleven LLMs on FinTrust and find that proprietary models like o4-mini outperforms in most tasks such as safety while open-source models like DeepSeek-V3 have advantage in specific areas like industry-level fairness. For challenging task like fiduciary alignment and disclosure, all LLMs fall short, showing a significant gap in legal awareness. We believe that FinTrust can be a valuable benchmark for LLMs' trustworthiness evaluation in finance domain. |
| 2025-10-17 | [High-entropy perovskites as new photocatalysts for cocatalyst-free water splitting](http://arxiv.org/abs/2510.15225v1) | Ho Truong Nam Hai, Makoto Arita et al. | The photocatalytic water-splitting process is thermodynamically challenging and requires catalysts with suitable band structures, as well as the presence of supporting cocatalysts. By considering the unique charge carrier mobility in perovskites, this study introduces three new ABO3-type high-entropy perovskites (Ba1/2Sr1/2)(Ti1/3Zr1/3Hf1/3)O3, (Ba1/2Sr1/2)(Ga1/3In1/3Sn1/3)O3 and (Ba1/2Sr1/2)(Ti1/3Zr1/3Sn1/3)O3 for cocatalyst-free photocatalysis. The three catalysts, having a single-phase cubic structure, are designed by considering configurational entropy, tolerance factor, octahedral factor, ionic radius deviation and valence deviation of >1.5R (R: gas constant), 0.9-1.0, 0.4-0.8, >0.3 and >0.3, respectively. The perovskites exhibit similar valence band tops, while their bandgaps vary slightly depending on the composition at the B-site (slightly lower bandgap by including d10 cations). Additionally, all three materials demonstrate effective hydrogen generation without the need for added cocatalysts. This investigation confirms that high-entropy oxide perovskites can offer significant potential for cocatalyst-free photocatalytic reactions. |
| 2025-10-17 | [ReasonIF: Large Reasoning Models Fail to Follow Instructions During Reasoning](http://arxiv.org/abs/2510.15211v1) | Yongchan Kwon, Shang Zhu et al. | The ability of large language models (LLMs) to follow user instructions is central to their reliability, safety, and usefulness. While prior studies assess instruction adherence in the model's main responses, we argue that it is also critical for large reasoning models (LRMs) to follow user instructions throughout their reasoning process. Reasoning instruction following makes LRMs more controllable and transparent, while reducing risks of undesirable shortcuts, hallucinations, or reward hacking within reasoning traces. To evaluate this dimension, we introduce ReasonIF, a systematic benchmark for assessing reasoning instruction following. ReasonIF includes six categories of instruction prompts, spanning multilingual reasoning, formatting and length control. Across many open-source LRMs including GPT-OSS, Qwen3, and DeepSeek-R1, we find substantial failures in reasoning instruction adherence: the highest instruction following score (IFS) remains below 0.25, meaning that fewer than $25\%$ of reasoning traces comply with the given instructions. Notably, as task difficulty increases, reasoning instruction following degrades further. We also explore two strategies to enhance reasoning instruction fidelity. (1) multi-turn reasoning and (2) Reasoning Instruction Finetuning (RIF) using synthetic data. RIF improves the IFS of $GPT-OSS-20B$ from 0.11 to 0.27, indicating measurable progress but leaving ample room for improvement. |
| 2025-10-16 | [Game mechanics for cyber-harm awareness in the metaverse](http://arxiv.org/abs/2510.15180v1) | Sophie McKenzie, Jeb Webb et al. | Educating children and young people to be safe online is essential, especially as the metaverse, a next-generation internet blending immersive technologies, promises to reshape their interactions and amplify their experiences. While virtual reality offers fully immersive, highly interactive, and multi-sensory engagement, it also heightens cyber harm risks for young or vulnerable users. To address this, the CyberNinjas VR experience was developed to educate children aged 8 to 16 on safe metaverse behaviours, providing clear referral steps for harmful interactions. Understanding user engagement in metaverse gaming will aid the design of future VR environments which prioritize safety and inclusivity. This project analyses CyberNinjas to understand how game mechanics can foster cyber-safe behaviours. |
| 2025-10-16 | [Morphotropic Phase Boundary (MPB) Induced Enhancement of Ferroelectric and Piezoelectric Properties in Li and Ta modified K0.5Na0.5NbO3](http://arxiv.org/abs/2510.15139v1) | Satyaranjan Sahoo, Dhiren K. Pradhan et al. | Lead-free (K0.48Na0.48Li0.04)(Nb1-xTax)O3 (KNLNT-x) ceramics were synthesized to study the effects of Li and Ta substitution on phase transition behavior, microstructure, and ferroelectric, dielectric, and piezoelectric properties. X-ray diffraction and Raman spectroscopy show that compositions with x < 0.10 exhibit a single orthorhombic (Amm2) phase, while 0.10 <= x <= 0.20 show coexistence of orthorhombic and tetragonal (Amm2 + P4mm) phases. For x > 0.20, a single tetragonal (P4mm) phase is obtained. Microstructural analysis shows a dense ceramic with decreasing grain size as Ta concentration increases. Temperature-dependent dielectric studies reveal two transitions: orthorhombic-tetragonal (TO-T) and tetragonal-cubic (TC). Both transition temperatures decrease systematically with increasing Ta, and TO-T shifts below room temperature for x > 0.15. The composition KNLNT-0.20 exhibits the highest dielectric constant (Er = 556) and piezoelectric coefficient (d33 = 159 pC/N). The enhanced piezoelectric response is attributed to a morphotropic phase boundary rather than a shift of the polymorphic phase boundary temperature. A composition-temperature phase diagram was constructed based on XRD, Raman, and dielectric data. |

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



