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
| 2026-01-22 | [Stochastic Control Barrier Functions under State Estimation: From Euclidean Space to Lie Groups](http://arxiv.org/abs/2601.16198v1) | Ruoyu Lin, Magnus Egerstedt | Ensuring safety for autonomous systems under uncertainty remains challenging, particularly when safety of the true state is required despite the true state not being fully known. Control barrier functions (CBFs) have become widely adopted as safety filters. However, standard CBF formulations do not explicitly account for state estimation uncertainty and its propagation, especially for stochastic systems evolving on manifolds. In this paper, we propose a safety-critical control framework with a provable bound on the finite-time safety probability for stochastic systems under noisy state information. The proposed framework explicitly incorporates the uncertainty arising from both process and measurement noise, and synthesizes controllers that adapt to the level of uncertainty. The framework admits closed-form solutions in linear settings, and experimental results demonstrate its effectiveness on systems whose state spaces range from Euclidean space to Lie groups. |
| 2026-01-22 | [Explainable AI to Improve Machine Learning Reliability for Industrial Cyber-Physical Systems](http://arxiv.org/abs/2601.16074v1) | Annemarie Jutte, Uraz Odyurt | Industrial Cyber-Physical Systems (CPS) are sensitive infrastructure from both safety and economics perspectives, making their reliability critically important. Machine Learning (ML), specifically deep learning, is increasingly integrated in industrial CPS, but the inherent complexity of ML models results in non-transparent operation. Rigorous evaluation is needed to prevent models from exhibiting unexpected behaviour on future, unseen data. Explainable AI (XAI) can be used to uncover model reasoning, allowing a more extensive analysis of behaviour. We apply XAI to to improve predictive performance of ML models intended for industrial CPS. We analyse the effects of components from time-series data decomposition on model predictions using SHAP values. Through this method, we observe evidence on the lack of sufficient contextual information during model training. By increasing the window size of data instances, informed by the XAI findings, we are able to improve model performance. |
| 2026-01-22 | [Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction](http://arxiv.org/abs/2601.16034v1) | Tony Cristofano | Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs (including GPT-OSS-20B and GLM-4) confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment. |
| 2026-01-22 | [Attributing and Exploiting Safety Vectors through Global Optimization in Large Language Models](http://arxiv.org/abs/2601.15801v1) | Fengheng Chu, Jiahao Chen et al. | While Large Language Models (LLMs) are aligned to mitigate risks, their safety guardrails remain fragile against jailbreak attacks. This reveals limited understanding of components governing safety. Existing methods rely on local, greedy attribution that assumes independent component contributions. However, they overlook the cooperative interactions between different components in LLMs, such as attention heads, which jointly contribute to safety mechanisms. We propose \textbf{G}lobal \textbf{O}ptimization for \textbf{S}afety \textbf{V}ector Extraction (GOSV), a framework that identifies safety-critical attention heads through global optimization over all heads simultaneously. We employ two complementary activation repatching strategies: Harmful Patching and Zero Ablation. These strategies identify two spatially distinct sets of safety vectors with consistently low overlap, termed Malicious Injection Vectors and Safety Suppression Vectors, demonstrating that aligned LLMs maintain separate functional pathways for safety purposes. Through systematic analyses, we find that complete safety breakdown occurs when approximately 30\% of total heads are repatched across all models. Building on these insights, we develop a novel inference-time white-box jailbreak method that exploits the identified safety vectors through activation repatching. Our attack substantially outperforms existing white-box attacks across all test models, providing strong evidence for the effectiveness of the proposed GOSV framework on LLM safety interpretability. |
| 2026-01-22 | [DualShield: Safe Model Predictive Diffusion via Reachability Analysis for Interactive Autonomous Driving](http://arxiv.org/abs/2601.15729v1) | Rui Yang, Lei Zheng et al. | Diffusion models have emerged as a powerful approach for multimodal motion planning in autonomous driving. However, their practical deployment is typically hindered by the inherent difficulty in enforcing vehicle dynamics and a critical reliance on accurate predictions of other agents, making them prone to safety issues under uncertain interactions. To address these limitations, we introduce DualShield, a planning and control framework that leverages Hamilton-Jacobi (HJ) reachability value functions in a dual capacity. First, the value functions act as proactive guidance, steering the diffusion denoising process towards safe and dynamically feasible regions. Second, they form a reactive safety shield using control barrier-value functions (CBVFs) to modify the executed actions and ensure safety. This dual mechanism preserves the rich exploration capabilities of diffusion models while providing principled safety assurance under uncertain and even adversarial interactions. Simulations in challenging unprotected U-turn scenarios demonstrate that DualShield significantly improves both safety and task efficiency compared to leading methods from different planning paradigms under uncertainty. |
| 2026-01-22 | [Even GPT-5.2 Can't Count to Five: The Case for Zero-Error Horizons in Trustworthy LLMs](http://arxiv.org/abs/2601.15714v1) | Ryoma Sato | We propose Zero-Error Horizon (ZEH) for trustworthy LLMs, which represents the maximum range that a model can solve without any errors. While ZEH itself is simple, we demonstrate that evaluating the ZEH of state-of-the-art LLMs yields abundant insights. For example, by evaluating the ZEH of GPT-5.2, we found that GPT-5.2 cannot even compute the parity of a short string like 11000, and GPT-5.2 cannot determine whether the parentheses in ((((()))))) are balanced. This is surprising given the excellent capabilities of GPT-5.2. The fact that LLMs make mistakes on such simple problems serves as an important lesson when applying LLMs to safety-critical domains. By applying ZEH to Qwen2.5 and conducting detailed analysis, we found that while ZEH correlates with accuracy, the detailed behaviors differ, and ZEH provides clues about the emergence of algorithmic capabilities. Finally, while computing ZEH incurs significant computational cost, we discuss how to mitigate this cost by achieving up to one order of magnitude speedup using tree structures and online softmax. |
| 2026-01-22 | [Improving Methodologies for LLM Evaluations Across Global Languages](http://arxiv.org/abs/2601.15706v1) | Akriti Vij, Benjamin Chua et al. | As frontier AI models are deployed globally, it is essential that their behaviour remains safe and reliable across diverse linguistic and cultural contexts. To examine how current model safeguards hold up in such settings, participants from the International Network for Advanced AI Measurement, Evaluation and Science, including representatives from Singapore, Japan, Australia, Canada, the EU, France, Kenya, South Korea and the UK conducted a joint multilingual evaluation exercise. Led by Singapore AISI, two open-weight models were tested across ten languages spanning high and low resourced groups: Cantonese English, Farsi, French, Japanese, Korean, Kiswahili, Malay, Mandarin Chinese and Telugu. Over 6,000 newly translated prompts were evaluated across five harm categories (privacy, non-violent crime, violent crime, intellectual property and jailbreak robustness), using both LLM-as-a-judge and human annotation.   The exercise shows how safety behaviours can vary across languages. These include differences in safeguard robustness across languages and harm types and variation in evaluator reliability (LLM-as-judge vs. human review). Further, it also generated methodological insights for improving multilingual safety evaluations, such as the need for culturally contextualised translations, stress-tested evaluator prompts and clearer human annotation guidelines. This work represents an initial step toward a shared framework for multilingual safety testing of advanced AI systems and calls for continued collaboration with the wider research community and industry. |
| 2026-01-22 | [Beyond Visual Safety: Jailbreaking Multimodal Large Language Models for Harmful Image Generation via Semantic-Agnostic Inputs](http://arxiv.org/abs/2601.15698v1) | Mingyu Yu, Lana Liu et al. | The rapid advancement of Multimodal Large Language Models (MLLMs) has introduced complex security challenges, particularly at the intersection of textual and visual safety. While existing schemes have explored the security vulnerabilities of MLLMs, the investigation into their visual safety boundaries remains insufficient. In this paper, we propose Beyond Visual Safety (BVS), a novel image-text pair jailbreaking framework specifically designed to probe the visual safety boundaries of MLLMs. BVS employs a "reconstruction-then-generation" strategy, leveraging neutralized visual splicing and inductive recomposition to decouple malicious intent from raw inputs, thereby leading MLLMs to be induced into generating harmful images. Experimental results demonstrate that BVS achieves a remarkable jailbreak success rate of 98.21\% against GPT-5 (12 January 2026 release). Our findings expose critical vulnerabilities in the visual safety alignment of current MLLMs. |
| 2026-01-22 | [AION: Aerial Indoor Object-Goal Navigation Using Dual-Policy Reinforcement Learning](http://arxiv.org/abs/2601.15614v1) | Zichen Yan, Yuchen Hou et al. | Object-Goal Navigation (ObjectNav) requires an agent to autonomously explore an unknown environment and navigate toward target objects specified by a semantic label. While prior work has primarily studied zero-shot ObjectNav under 2D locomotion, extending it to aerial platforms with 3D locomotion capability remains underexplored. Aerial robots offer superior maneuverability and search efficiency, but they also introduce new challenges in spatial perception, dynamic control, and safety assurance. In this paper, we propose AION for vision-based aerial ObjectNav without relying on external localization or global maps. AION is an end-to-end dual-policy reinforcement learning (RL) framework that decouples exploration and goal-reaching behaviors into two specialized policies. We evaluate AION on the AI2-THOR benchmark and further assess its real-time performance in IsaacSim using high-fidelity drone models. Experimental results show that AION achieves superior performance across comprehensive evaluation metrics in exploration, navigation efficiency, and safety. The video can be found at https://youtu.be/TgsUm6bb7zg. |
| 2026-01-22 | [ToxiTwitch: Toward Emote-Aware Hybrid Moderation for Live Streaming Platforms](http://arxiv.org/abs/2601.15605v1) | Baktash Ansari, Shiza Ali et al. | The rapid growth of live-streaming platforms such as Twitch has introduced complex challenges in moderating toxic behavior. Traditional moderation approaches, such as human annotation and keyword-based filtering, have demonstrated utility, but human moderators on Twitch constantly struggle to scale effectively in the fast-paced, high-volume, and context-rich chat environment of the platform while also facing harassment themselves. Recent advances in large language models (LLMs), such as DeepSeek-R1-Distill and Llama-3-8B-Instruct, offer new opportunities for toxicity detection, especially in understanding nuanced, multimodal communication involving emotes. In this work, we present an exploratory comparison of toxicity detection approaches tailored to Twitch. Our analysis reveals that incorporating emotes improves the detection of toxic behavior. To this end, we introduce ToxiTwitch, a hybrid model that combines LLM-generated embeddings of text and emotes with traditional machine learning classifiers, including Random Forest and SVM. In our case study, the proposed hybrid approach reaches up to 80 percent accuracy under channel-specific training (with 13 percent improvement over BERT and F1-score of 76 percent). This work is an exploratory study intended to surface challenges and limits of emote-aware toxicity detection on Twitch. |
| 2026-01-22 | [Tackling the Scaffolding Paradox: A Person-Centered Adaptive Robotic Interview Coach](http://arxiv.org/abs/2601.15600v1) | Wanqi Zhang, Jiangen He et al. | Job interview anxiety is a prevalent challenge among university students and can undermine both performance and confidence in high-stakes evaluative situations. Social robots have shown promise in reducing anxiety through emotional support, yet how such systems should balance psychological safety with effective instructional guidance remains an open question. In this work, we present a three-phase iterative design study of a robotic interview coach grounded in Person-Centered Therapy (PCT) and instructional scaffolding theory. Across three weekly sessions (N=8), we systematically explored how different interaction strategies shape users' emotional experience, cognitive load, and perceived utility. Phase I demonstrated that a PCT-based robot substantially increased perceived psychological safety but introduced a Safety-Guidance Gap, in which users felt supported yet insufficiently coached. Phase II revealed a Scaffolding Paradox: immediate feedback improved clarity but disrupted conversational flow and increased cognitive load, whereas delayed feedback preserved realism but lacked actionable specificity. To resolve this tension, Phase III introduced an Agency-Driven Interaction Mode that allowed users to opt in to feedback dynamically. Qualitative findings indicated that user control acted as an anxiety buffer, restoring trust, reducing overload, and reframing the interaction as collaborative rather than evaluative. Quantitative measures further showed significant reductions in interview-related social and communication anxiety, while maintaining high perceived warmth and therapeutic alliance. We synthesize these findings into an Adaptive Scaffolding Ecosystem framework, highlighting user agency as a key mechanism for balancing emotional support and instructional guidance in social robot coaching systems. |
| 2026-01-22 | [YuFeng-XGuard: A Reasoning-Centric, Interpretable, and Flexible Guardrail Model for Large Language Models](http://arxiv.org/abs/2601.15588v1) | Junyu Lin, Meizhen Liu et al. | As large language models (LLMs) are increasingly deployed in real-world applications, safety guardrails are required to go beyond coarse-grained filtering and support fine-grained, interpretable, and adaptable risk assessment. However, existing solutions often rely on rapid classification schemes or post-hoc rules, resulting in limited transparency, inflexible policies, or prohibitive inference costs. To this end, we present YuFeng-XGuard, a reasoning-centric guardrail model family designed to perform multi-dimensional risk perception for LLM interactions. Instead of producing opaque binary judgments, YuFeng-XGuard generates structured risk predictions, including explicit risk categories and configurable confidence scores, accompanied by natural language explanations that expose the underlying reasoning process. This formulation enables safety decisions that are both actionable and interpretable. To balance decision latency and explanatory depth, we adopt a tiered inference paradigm that performs an initial risk decision based on the first decoded token, while preserving ondemand explanatory reasoning when required. In addition, we introduce a dynamic policy mechanism that decouples risk perception from policy enforcement, allowing safety policies to be adjusted without model retraining. Extensive experiments on a diverse set of public safety benchmarks demonstrate that YuFeng-XGuard achieves stateof-the-art performance while maintaining strong efficiency-efficacy trade-offs. We release YuFeng-XGuard as an open model family, including both a full-capacity variant and a lightweight version, to support a wide range of deployment scenarios. |
| 2026-01-21 | [CompliantVLA-adaptor: VLM-Guided Variable Impedance Action for Safe Contact-Rich Manipulation](http://arxiv.org/abs/2601.15541v1) | Heng Zhang, Wei-Hsing Huang et al. | We propose a CompliantVLA-adaptor that augments the state-of-the-art Vision-Language-Action (VLA) models with vision-language model (VLM)-informed context-aware variable impedance control (VIC) to improve the safety and effectiveness of contact-rich robotic manipulation tasks. Existing VLA systems (e.g., RDT, Pi0, OpenVLA-oft) typically output position, but lack force-aware adaptation, leading to unsafe or failed interactions in physical tasks involving contact, compliance, or uncertainty. In the proposed CompliantVLA-adaptor, a VLM interprets task context from images and natural language to adapt the stiffness and damping parameters of a VIC controller. These parameters are further regulated using real-time force/torque feedback to ensure interaction forces remain within safe thresholds. We demonstrate that our method outperforms the VLA baselines on a suite of complex contact-rich tasks, both in simulation and on real hardware, with improved success rates and reduced force violations. The overall success rate across all tasks increases from 9.86\% to 17.29\%, presenting a promising path towards safe contact-rich manipulation using VLAs. We release our code, prompts, and force-torque-impedance-scenario context datasets at https://sites.google.com/view/compliantvla. |
| 2026-01-21 | [From Generative Engines to Actionable Simulators: The Imperative of Physical Grounding in World Models](http://arxiv.org/abs/2601.15533v1) | Zhikang Chen, Tingting Zhu | A world model is an AI system that simulates how an environment evolves under actions, enabling planning through imagined futures rather than reactive perception. Current world models, however, suffer from visual conflation: the mistaken assumption that high-fidelity video generation implies an understanding of physical and causal dynamics. We show that while modern models excel at predicting pixels, they frequently violate invariant constraints, fail under intervention, and break down in safety-critical decision-making. This survey argues that visual realism is an unreliable proxy for world understanding. Instead, effective world models must encode causal structure, respect domain-specific constraints, and remain stable over long horizons. We propose a reframing of world models as actionable simulators rather than visual engines, emphasizing structured 4D interfaces, constraint-aware dynamics, and closed-loop evaluation. Using medical decision-making as an epistemic stress test, where trial-and-error is impossible and errors are irreversible, we demonstrate that a world model's value is determined not by how realistic its rollouts appear, but by its ability to support counterfactual reasoning, intervention planning, and robust long-horizon foresight. |
| 2026-01-21 | [TransportAgents: a multi-agents LLM framework for traffic accident severity prediction](http://arxiv.org/abs/2601.15519v1) | Zhichao Yang, Jiashu He et al. | Accurate prediction of traffic crash severity is critical for improving emergency response and public safety planning. Although recent large language models (LLMs) exhibit strong reasoning capabilities, their single-agent architectures often struggle with heterogeneous, domain-specific crash data and tend to generate biased or unstable predictions. To address these limitations, this paper proposes TransportAgents, a hybrid multi-agent framework that integrates category-specific LLM reasoning with a multilayer perceptron (MLP) integration module. Each specialized agent focuses on a particular subset of traffic information, such as demographics, environmental context, or incident details, to produce intermediate severity assessments that are subsequently fused into a unified prediction. Extensive experiments on two complementary U.S. datasets, the Consumer Product Safety Risk Management System (CPSRMS) and the National Electronic Injury Surveillance System (NEISS), demonstrate that TransportAgents consistently outperforms both traditional machine learning and advanced LLM-based baselines. Across three representative backbones, including closed-source models such as GPT-3.5 and GPT-4o, as well as open-source models such as LLaMA-3.3, the framework exhibits strong robustness, scalability, and cross-dataset generalizability. A supplementary distributional analysis further shows that TransportAgents produces more balanced and well-calibrated severity predictions than standard single-agent LLM approaches, highlighting its interpretability and reliability for safety-critical decision support applications. |
| 2026-01-21 | [DCeption: Real-world Wireless Man-in-the-Middle Attacks Against CCS EV Charging](http://arxiv.org/abs/2601.15515v1) | Marcell Szakály, Martin Strohmeier et al. | The adoption of Electric Vehicles (EVs) is happening at a rapid pace. To ensure fast and safe charging, complex communication is required between the vehicle and the charging station. In the globally used Combined Charging System (CCS), this communication is carried over the HomePlug Green PHY (HPGP) physical layer. However, HPGP is known to suffer from wireless leakage, which may expose this data link to nearby attackers.   In this paper, we examine active wireless attacks against CCS, and study the impact they can have. We present the first real-time Software-Defined Radio (SDR) implementation of HPGP, granting unprecedented access to the communications within the charging cables. We analyze the characteristics of 2,750 real-world charging sessions to understand the timing constraints for hijacking. Using novel techniques to increase the attacks' reliability, we design a robust wireless Man-in-the-Middle evaluation framework for CCS.   We demonstrate full control over TLS usage and CCS protocol version negotiation, including TLS stripping attacks. We investigate how real devices respond to safety-critical MitM attacks, which modify power delivery information, and found target vehicles to be highly permissive. First, we caused a vehicle to display charging power exceeding 900 kW on the dashboard, while receiving only 40 kW. Second, we remotely overcharged a vehicle, at twice the requested current for 17 seconds before the vehicle triggered the emergency shutdown. Finally, we propose a backwards-compatible, downgrade-proof protocol extension to mitigate the underlying vulnerabilities. |
| 2026-01-21 | [Early predicting of hospital admission using machine learning algorithms: Priority queues approach](http://arxiv.org/abs/2601.15481v1) | Jakub Antczak, James Montgomery et al. | Emergency Department overcrowding is a critical issue that compromises patient safety and operational efficiency, necessitating accurate demand forecasting for effective resource allocation. This study evaluates and compares three distinct predictive models: Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors (SARIMAX), EXtreme Gradient Boosting (XGBoost) and Long Short-Term Memory (LSTM) networks for forecasting daily ED arrivals over a seven-day horizon. Utilizing data from an Australian tertiary referral hospital spanning January 2017 to December 2021, this research distinguishes itself by decomposing demand into eight specific ward categories and stratifying patients by clinical complexity. To address data distortions caused by the COVID-19 pandemic, the study employs the Prophet model to generate synthetic counterfactual values for the anomalous period. Experimental results demonstrate that all three proposed models consistently outperform a seasonal naive baseline. XGBoost demonstrated the highest accuracy for predicting total daily admissions with a Mean Absolute Error of 6.63, while the statistical SARIMAX model proved marginally superior for forecasting major complexity cases with an MAE of 3.77. The study concludes that while these techniques successfully reproduce regular day-to-day patterns, they share a common limitation in underestimating sudden, infrequent surges in patient volume. |
| 2026-01-21 | [Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts](http://arxiv.org/abs/2601.15479v1) | Sydney Anuyah, Sneha Shajee-Mohan et al. | The safe deployment of large language models (LLMs) in high-stakes fields like biomedicine, requires them to be able to reason about cause and effect. We investigate this ability by testing 13 open-source LLMs on a fundamental task: pairwise causal discovery (PCD) from text. Our benchmark, using 12 diverse datasets, evaluates two core skills: 1) \textbf{Causal Detection} (identifying if a text contains a causal link) and 2) \textbf{Causal Extraction} (pulling out the exact cause and effect phrases). We tested various prompting methods, from simple instructions (zero-shot) to more complex strategies like Chain-of-Thought (CoT) and Few-shot In-Context Learning (FICL).   The results show major deficiencies in current models. The best model for detection, DeepSeek-R1-Distill-Llama-70B, only achieved a mean score of 49.57\% ($C_{detect}$), while the best for extraction, Qwen2.5-Coder-32B-Instruct, reached just 47.12\% ($C_{extract}$). Models performed best on simple, explicit, single-sentence relations. However, performance plummeted for more difficult (and realistic) cases, such as implicit relationships, links spanning multiple sentences, and texts containing multiple causal pairs. We provide a unified evaluation framework, built on a dataset validated with high inter-annotator agreement ($κ\ge 0.758$), and make all our data, code, and prompts publicly available to spur further research. \href{https://github.com/sydneyanuyah/CausalDiscovery}{Code available here: https://github.com/sydneyanuyah/CausalDiscovery} |
| 2026-01-21 | [Neural Collision Detection for Multi-arm Laparoscopy Surgical Robots Through Learning-from-Simulation](http://arxiv.org/abs/2601.15459v1) | Sarvin Ghiasi, Majid Roshanfar et al. | This study presents an integrated framework for enhancing the safety and operational efficiency of robotic arms in laparoscopic surgery by addressing key challenges in collision detection and minimum distance estimation. By combining analytical modeling, real-time simulation, and machine learning, the framework offers a robust solution for ensuring safe robotic operations. An analytical model was developed to estimate the minimum distances between robotic arms based on their joint configurations, offering precise theoretical calculations that serve as both a validation tool and a benchmark. To complement this, a 3D simulation environment was created to model two 7-DOF Kinova robotic arms, generating a diverse dataset of configurations for collision detection and distance estimation. Using these insights, a deep neural network model was trained with joint actuators of robot arms and relative positions as inputs, achieving a mean absolute error of 282.2 mm and an R-squared value of 0.85. The close alignment between predicted and actual distances highlights the network's accuracy and its ability to generalize spatial relationships. This work demonstrates the effectiveness of combining analytical precision with machine learning algorithms to enhance the precision and reliability of robotic systems. |
| 2026-01-21 | [Lattice: A Confidence-Gated Hybrid System for Uncertainty-Aware Sequential Prediction with Behavioral Archetypes](http://arxiv.org/abs/2601.15423v1) | Lorian Bannis | We introduce Lattice, a hybrid sequential prediction system that conditionally activates learned behavioral structure using binary confidence gating. The system clusters behavior windows into behavioral archetypes and uses binary confidence gating to activate archetype-based scoring only when confidence exceeds a threshold, falling back to baseline predictions when uncertain. We validate Lattice on recommendation systems (MovieLens), scientific time-series (LIGO), and financial markets, using LSTM and transformer backbones. On MovieLens with LSTM, Lattice achieves +31.9% improvement over LSTM baseline in HR@10 (p < 3.29 x 10^-25, 30 seeds), outperforming transformer baselines by 109.4% over SASRec and 218.6% over BERT4Rec. On LIGO and financial data, the system correctly refuses archetype activation when distribution shift occurs - a successful outcome demonstrating confidence gating prevents false activation. On transformer backbones, Lattice provides 0.0% improvement (neutral, no degradation), gracefully deferring when structure is already present. This bidirectional validation - activating when patterns apply, refusing when they don't, and deferring when redundant - supports confidence gating as a promising architectural principle for managing epistemic uncertainty in safety-critical applications. |

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



