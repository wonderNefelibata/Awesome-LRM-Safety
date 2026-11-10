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
| 2026-11-09 | [Shortcuts and Boundaries: Verifiable RL Misalignment and Contextual Safety Learning](http://arxiv.org/abs/2611.12865v1) | Xinjie Shen | Reinforcement learning (RL) can improve an agent's task capability while reinforcing behavior that violates contextual execution requirements. We study this reward-driven misalignment, including reward hacking, through verifiable reward--context conflicts, where compliant task completion remains feasible while some executable shortcuts earn more reward than every compliant strategy. Encoding these conflicts in task dynamics lets us generate diverse interactive environments at scale and derive compliance feedback from execution histories. In resource and reporting workflows, task optimization increases real productivity through unauthorized resource use or raises reported scores while actual work declines. Controlled changes in the training distribution show that minority conflict exposure can still lead to widespread shortcut acquisition, reaching 97.9\% certified profitable violations on an API task under a 25\% conflict mixture after 128 updates. The same contextual checks supply a simple binary penalty that supports joint capability and compliance gains and partially mitigates acquired violations. Learned safety adapts to expanded authorization and transfers across mechanisms into workflows with evolving evidence and approval requirements. On 32 unseen changing-evidence tasks, resource-trained safety increases safe completion from 53.1\% to 89.1\% relative to task-only training, while both policies achieve full real work. These results connect a verifiable account of reward-driven misalignment with scalable task construction and safety learning that remains useful under changing execution requirements. |
| 2026-11-09 | [Safe Performance-aware Online Input Design](http://arxiv.org/abs/2611.12822v1) | Diyou Liu, Mohammad Khosravi | In this paper, we propose an input design framework for the safe online excitation and identification of uncertain linear systems without requiring informative initial data. The objective is to actively excite the system and collect sufficiently informative input-output data while maintaining safe operation and satisfactory control performance. Achieving this objective is challenging due to the potential conflict between the excitation required for learning and the safety and performance requirements, particularly when limited prior knowledge of the system is available. Inspired by the dual control principle, the proposed scheme balances exploitation for control performance with exploration for model improvement. A Bayesian learning approach is employed to update the model uncertainty online, starting from a conservative initial uncertainty set. As informative data are collected, the uncertainty set contracts, enabling progressively less conservative operation. To reduce the online computational burden, the resulting program is reformulated into a computationally tractable form. Furthermore, an online ellipsoidal terminal set with non-increasing conservatism is designed to guarantee the recursive feasibility of the proposed scheme. Numerical experiments are conducted to demonstrate the effectiveness of the proposed framework, particularly in terms of uncertainty reduction and control performance. |
| 2026-11-09 | [Beyond Falsehoods 2026: Global Trends in the Regulation of Misinformation and Disinformation](http://arxiv.org/abs/2611.12770v1) | Fakhar Abbas, Simon Chesterman et al. | This report examines global legislative and regulatory responses to misinformation, disinformation, and mal-information (MDM) between 1995 and 2025. Drawing on an expanded and updated dataset of 409 coded regulatory and policy entries, including hard and soft law, policy measures, and proposed or draft instruments, it analyses their growth, geographic distribution, substantive focuses, and regulatory designs, which represent the various approaches taken against MDM, as well as early evidence of their implementation.   The central finding is sustained and accelerating regulatory growth. Annual activity increased from 11 instruments in 2016 to 46 in 2023, 67 in 2024, and 71 in 2025. The number of instruments currently in force has reached 282, despite 34 measures having been repealed, withdrawn, or judicially invalidated. This growth has also been accompanied by a shift in regulatory focus. Earlier measures relied heavily on broad prohibitions of "false information," whereas more recent measures increasingly target specific harms, including deepfakes, election interference, non-consensual intimate imagery, and child safety. Regulatory responsibility is also increasingly being placed on platforms rather than solely on individual creators and users. |
| 2026-11-09 | [Double Blind Evals: Resolving the Dual Confidentiality Dilemma in AI Safety Auditing](http://arxiv.org/abs/2611.12604v1) | Andrew Trask, Sol Messing et al. | AI safety evaluations often involve sensitive prompts and proprietary models. Evaluators want to keep their evaluation prompts secret from model owners to prevent purposeful or incidental data leakage, ensuring longer usefulness of their benchmark. Model owners in turn wish to keep their proprietary model weights and inference code within their own secure infrastructure to protect their IP. We present a way forward for "double-blind evals", which provide cryptographic guarantees that an evaluator's prompts are not shared with the model owner, while simultaneously maintaining the confidentiality of the model weights and inference code. To accomplish this, we use secure enclaves that encrypt data in RAM at the hardware level. In a world-first pilot, we test this approach by evaluating a proprietary Google DeepMind model (Gemini 2.5 Flash Lite) against a selection of the private AILuminate benchmark corpus, using an NVIDIA H100 secure enclave on Google Cloud Platform, and using OpenMined's PySyft to maintain data and model privacy. We believe this effort marks the first use of this technology to provide mutual secrecy for one organization's frontier-class proprietary model weights and another organization's closed benchmark. |
| 2026-11-09 | [Frontier AI Safety and Governance 2026: Risk Understanding and Coordinated Governance](http://arxiv.org/abs/2611.12434v1) | Zeng Yi, Seán Ó hÉigeartaigh et al. | Frontier AI Safety and Governance 2026: Risk Understanding and Coordinated Governance is a key outcome of the AI Safety and Governance Program of World Internet Conference Specialized Committee on AI in 2026. Drawing on the latest developments in frontier AI capabilities and applications, the report systematically examines the evolving risk landscape, analyzing key risk areas, their severity and response priorities. Building on a review of existing global governance practices and an assessment of frontier AI governance readiness, it identifies key governance gaps and weaknesses and proposes directions and practical mechanisms for coordinated responses. The report provides a reference for government departments, international organizations, businesses, research institutions and other stakeholders. It aims to strengthen the international community's understanding of and consensus on frontier AI risks, enhance preparedness and collective response capacity, and advance the safe, reliable and responsible development of frontier AI to better serve the common well-being of humanity. |
| 2026-11-09 | [Unprompted coordination through shared memory in LLM agents: a locked two-implementation replication and extension](http://arxiv.org/abs/2611.12412v1) | Pekka Abrahamsson, Muhammad Waseem et al. | Coordination between large language-model agents has become a central question for AI safety and software engineering, as such agents are deployed in groups that share infrastructure. Existing research tells the agents that partners exist or asks them to coordinate. Whether coordination arises through shared state alone, with nothing disclosing that anyone else is present, is the key question for deployments that share caches and memory between agents, yet it has rarely been examined, and never with independent replication; this study does both. Four agents each received a short arithmetic task and a key-value store presented to each as its own scratch space; two held complementary halves of one computation, and no prompt mentioned other agents, sharing or collaboration. For reliability, the environment was rebuilt independently from a written protocol, and both implementations ran under hashed locks on design, data and analysis. The results bear novelty value in three respects. Joint success occurred only through the shared store. Whether a model coordinated depended not on its vendor tier but on whether it wrote its intermediate result and looked at the store again, and one ordinary sentence, "check your workspace again before submitting", switched that second look on and off. The findings held under replication. For practice, state shared between agents is a communication channel and should be separated, namespaced and logged. For research, designs should be locked before the data, and the parse rule, endpoint and parameters reported. Future work should test open-weight models and separate using another agent's output from intending to communicate. |
| 2026-11-09 | [Safety-Constrained Intelligent Control of a 2-DOF Robotic Manipulator Using Control Barrier Functions](http://arxiv.org/abs/2611.12378v1) | Gabriel da Silva Lima, Made Widhi Surya Atman et al. | Control barrier function (CBF) filters depend on dynamic models that can be inaccurate when their parameters and external forces are uncertain. This paper investigates a radial basis function (RBF) neural network compensator whose output is shared by a Cartesian tracking controller and a joint-space CBF filter. An auxiliary state driven by the difference between requested and applied torques corrects the learning signal, reducing the direct influence of filter intervention on adaptation. A two-link manipulator is evaluated through constrained regulation, ablations, matched uncertainty and disturbance trials, and operating-range and timing studies. The results demonstrate that the proposed controller improves tracking while maintaining joint, velocity, and torque limits under parameter uncertainty, and that both the shared compensation and the auxiliary correction contribute to this improvement. |
| 2026-11-09 | [Hierarchical Multi-Channel Tip-Over Prediction via Isolated Temporal Models for Planetary Rover Safety](http://arxiv.org/abs/2611.12261v1) | Siddhant Shete, Raul Dominguez et al. | Reliable tip-over prevention is critical for planetary rovers operating in unstructured environments where communication delays prevent real-time human intervention. Existing learning-based approaches typically optimize a single prediction objective, limiting their ability to capture both gradual instability evolution and abrupt terrain disturbances. This work presents a hierarchical multi-channel tip-over prediction framework based on isolated temporal learning. A multi-horizon Temporal Convolutional Network (TCN) learns progressive instability patterns from inertial measurements, while a Gated Recurrent Unit (GRU)-based forward dynamics model predicts nominal rover motion and detects unexpected dynamic deviations through prediction residuals. These complementary signals are integrated through decision-level fusion and priority-based arbitration to generate graded safety actions, including early warning and emergency braking. The proposed system is deployed as an Orocos Real-Time Toolkit (RTT)-based safety component within the Rock robotics framework, operating at 100 Hz on a planetary rover software stack. Experiments on Inertial Measurement Unit (IMU) samples collected across heterogeneous terrains demonstrate 97.58% event prediction coverage with a 1.82s median lead time, outperforming individual prediction channels. The results validate that isolated temporal models combined through hierarchical decision fusion provide a robust solution for real-time rover safety in uncertain terrain. |
| 2026-11-09 | [Compact and Wide-Force-Range Magnetorheological Hand Exoskeleton for Kinesthetic Feedback in Teleoperation](http://arxiv.org/abs/2611.12130v1) | Xianlong Mai, Rui Yu et al. | Hand kinesthetic feedback is crucial for restoring and enhancing human perception of force and displacement in robotic hand teleoperation. Wide perception range of force feedback is vital in rendering objects of different stiffness, especially for high-stiffness rendering. However, existing approaches struggle to achieve wide range of force feedback while maintaining compact and wearable design. This paper presents a hand exoskeleton system employing magnetorheological (MR) brakes to provide a wide range kinesthetic feedback during teleoperation. The MR brake provides a braking torque ranging from 0.01 to 0.918 Nm, while maintaining a compact structure with a total mass of only 39.8 g. The proposed exoskeleton can render fingertip forces up to 21 N while maintaining back-driving force of 1.01 N. Experimental results show that the system allows users to discriminate object stiffness using kinesthetic feedback alone. In a tele-grasping task, when both conditions achieved a 100% success rate, kinesthetic feedback led to a 28.2% reduction in object contact force compared to visual feedback only, enhancing object safety and operator immersion. Questionnaire results further indicate that kinesthetic feedback improves reliability, realism, and user confidence during teleoperation. |
| 2026-11-09 | [CRAFT-AI: Capability and Requirement Alignment for Flexible Manufacturing through Agentic AI](http://arxiv.org/abs/2611.12092v1) | Kiavash Fathi, Hans Wernher van de Venn | One of the fundamental needs for reaching resilient production infrastructure is flexible manufacturing, which requires translating changing production requirements into implementable system configurations. This paper introduces \textbf{CRAFT-AI}, an agentic AI approach combining natural language (NL) requirement interpretation, Asset Administration Shell (AAS)-based capability descriptions, and reusable engineering knowledge from existing manufacturing configurations. A tool-using AI agent matches requirements with available capabilities and reference implementations to generate manufacturing scenes and corresponding AAS instances. The approach is demonstrated using a Factory I/O prototype with twelve reference scenes and five illustrative manufacturing tasks of varying complexity, providing an initial feasibility assessment of the proposed AAS-enabled, system-level generative engineering approach. For the four valid user task descriptions, proper requirements, correct references to the relevant engineering knowledge, correct components and lastly a valid Factory I/O scene are generated. For the task with missing engineering knowledge and a wrong sensor selection, the user is informed about which requirements of the scene generation lack domain knowledge and also which sensor is not compatible with the selected raw material by the user with link to the related documentation. After fixing the task description, the correct scene is seamlessly generated. \textbf{CRAFT-AI} highlights the importance of safeguards for risk perception and communication in foundation model-driven solutions. |
| 2026-11-09 | [BECON: Belief-Conditioned Constrained Multi-Objective Reinforcement Learning under Drifting Preferences and Budgets](http://arxiv.org/abs/2611.12053v1) | Bui Trong Duc, Huynh Thi Thanh Binh | Constrained preference-conditioned MORL typically assumes full task observability, treating preferences and safety budgets as stationary task signals. In real-world deployment, however, these signals are often unreliable proxies: intent may drift, safety tolerances may shift with context, and exposed descriptors may be noisy, delayed, or incomplete. We therefore formulate constrained preference-conditioned MORL as a partially observed task-specification problem, where the agent must navigate a fundamental conflict between unreliable external signals and latent interaction dynamics. We propose \textbf{BECON}, a belief-conditioned framework for constrained MORL with drifting preferences and budgets, which infers latent task context from recent histories to estimate preferences and safety budgets. These estimates are fused with the exposed task signal via uncertainty-dependent gates, enabling the policy to defer to observations under high epistemic uncertainty while correcting them when history is informative. The fused context jointly conditions preference optimization and context-adaptive constraint enforcement. Experiments on diverse multi-objective control tasks demonstrate that BECON consistently improves over state-of-the-art MORL baselines, achieving stronger robustness and constraint satisfaction under task drift. |
| 2026-11-09 | [FocusBranch: Combinatorial Branch-and-Bound for $\ell_0$ Neural Network Robustness Verification](http://arxiv.org/abs/2611.12024v1) | Renwei Deng, Yang He et al. | Verifying the local robustness of neural networks under $\ell_0$ perturbations is inherently combinatorial, as it must reason over all sparse subsets of $ε$ perturbed pixels. This paper presents FocusBranch, a combinatorial branch-and-bound framework for $\ell_0$ robustness verification. The key insight is to make the $\ell_0$-ball decomposable and branchable using a node abstraction, which consists of a focus pixel set $\mathcal{S}$ and a level $l$, permitting at most $ε-l$ pixels outside $\mathcal{S}$. This abstraction unifies two branching mechanisms: concrete nodes branch through upper shadows to fix additional perturbed locations, while super nodes branch through covering designs to cover many perturbation cases with fewer child nodes. FocusBranch proves node safety by concretizing linear relaxations over node-constrained $\ell_0$-balls and recursively branches on unknown nodes. Rather than relying on a fixed verification plan, it samples nodes across levels to estimate safe rates and costs, then selects a starting level and branching strategy predicted to minimize verification time. Experiments on fully connected and convolutional classifiers for MNIST, Fashion-MNIST, and CIFAR-10 show that FocusBranch verifies $\ell_0$ robustness effectively across architectures and outperforms state-of-the-art verifiers. The code is available at: https://github.com/simpl-group/FocusBranch. |
| 2026-11-09 | [OASIS: Optimization-Aware Subgoal Inference with Safety Constraints for Autonomous Mobile Robot Navigation](http://arxiv.org/abs/2611.11996v1) | Arian Hajizadeha, Javad Pourrafia et al. | Navigation in dynamic environments requires mobile robots to make progress toward their goals while respecting motion limits and avoiding surrounding obstacles. Learned subgoal policies can guide model predictive control beyond its local prediction horizon, but a geometrically suitable subgoal does not necessarily lead to a successfully solved control problem. This paper proposes OASIS, a hierarchical navigation framework that incorporates recent optimization outcomes into reinforcement learning-based subgoal generation. A high-level policy trained using Soft Actor--Critic generates local subgoals, while a nonlinear model predictive controller tracks them subject to motion limits, geometric collision-avoidance constraints, and embedded high-order control barrier function constraints. The outcomes of controller optimizations over each subgoal interval are summarized and provided to the policy at its next decision, without requiring additional optimization calls. The framework is evaluated in five navigation scenarios with static and moving obstacles, including dense interactions and sensing uncertainty. Comparisons with hierarchical and direct-goal control variants show that OASIS achieves the highest aggregate navigation success. Matched comparisons across three training seeds further show a consistent success advantage over the otherwise identical hierarchy without optimization-status input. The selected-policy comparison also shows fewer primary optimization failures with similar mean controller time. |
| 2026-11-09 | [AnyBreak: Towards Jailbreaking Any T2I Model for Any Concept](http://arxiv.org/abs/2611.11980v1) | Qi Lu, Zehui Guo et al. | Text-to-image (T2I) models have demonstrated remarkable capabilities in image generation and editing, yet this progress has also introduced significant security risks. Prior work has shown that adversarial prompt can induce the generation of Not-Safe-For-Work (NSFW) content, including pornographic, violent, and graphic imagery. However, such attacks are typically mitigated by prompt filters and often rely on white-box assumptions. In this paper, we propose AnyBreak, a novel jailbreak framework that constructs adversarial images via multi-scale progressive semantic alignment. During the generation process, these adversarial images steer the generation trajectory in latent space towards unsafe data manifolds, enabling cross-modal transfer of malicious concepts from the text modality to the image modality. AnyBreak operates using only clean textual prompts and adversarial images at inference time, and can jailbreak any T2I models towards any malicious concepts in a fully black-box setting. We further introduce AnyBreak-V, an enhanced variant that incorporates a compensation term to bypass post-hoc safety checkers. Extensive experiments demonstrate that AnyBreak achieves high attack success rates while preserving visual fidelity and semantic consistency with the input prompts under black-box and clean-text constraints. Our findings reveal that existing T2I defense leave critical cross-modal attack surfaces insufficiently addressed. The code is available at https://github.com/luqi-glory/AnyBreak. |
| 2026-11-09 | [Long-distance quantum secure direct communication with Gottesman-Kitaev-Preskill qubits](http://arxiv.org/abs/2611.11822v1) | Meng-Dong Zhu, Jia-Wei Ying et al. | The Gottesman-Kitaev-Preskill (GKP) code possesses a lattice periodic structure that enables the effective correction of small displacement errors induced by noise in phase space, which is indispensable for quantum computation. The GKP code was also used to construct a long-distance entanglement channel in quantum repeaters and distribute random keys in quantum key distribution (QKD). Using GKP encoding, we propose a quantum secure direct communication (QSDC) protocol that not only transmits the secret message directly, but also helps protect the safety of communication personnel. The exponential attenuation loss of photons can be overcome by the third-generation quantum repeater. Numerical simulation results demonstrate that secure transmission distance exceeding 1000 km is achievable with a 23 dB squeezing level. Furthermore, we establish a cost-based optimization framework to characterize the trade-off among secure communication distance, secrecy capacity, and repeater resources, enabling optimized repeater deployment and providing a systematic design strategy for scalable quantum repeater architectures. The results demonstrate the potential of combining GKP error correction for long-distance QSDC and quantum networks. |
| 2026-11-09 | [CaseLaw: A Policy-Decoupled Framework for Training-Free Content Moderation](http://arxiv.org/abs/2611.11812v1) | Qi Lu, Sen Li et al. | Content moderation is inherently policy-dependent: moderation boundaries are often subtle, subjective, and frequently updated, while conventional rule based or fine-tuning based systems require costly rule rewriting, model retraining, or system redeployment. We propose CaseLaw, a policy-decoupled and training-free framework for adaptive content moderation. CaseLaw reformulates moderation as precedent-based retrieval and voting, where multimodal content is transformed into evidence-preserved structured semantics, embedded into a stable semantic space, and classified by thresholded voting over retrieved precedent labels. We train NoteCap on a large-scale corpus of structurally annotated notes to ensure consistent semantic extraction and freeze it during subsequent policy adaptation, which requires no additional training. Policy adaptation is achieved by updating the semantic case library, including case labels and mappings, without retraining models or modifying prompts. To evaluate CaseLaw, we construct MFO-RedNotes, a benchmark for the male-female opposition risk domain. We additionally provide a five category CoPro comparison. Experiments show that CaseLaw consistently outperforms competing baselines in both precision and recall across policy-dependent risk levels and broader safety categories, with category-wise precision/recall reaching up to 86.8%/90.1% on MFO-RedNotes across both policy settings and 94.5%/92.3% on CoPro. Our work highlights the potential of precedent-based reasoning for policy adaptive content moderation and offers a practical direction toward more stable, transparent, and training-free moderation systems. |
| 2026-11-09 | [ShadowBench-MCP: Measuring Shadow Invocations in MCP-Style Tool Routing](http://arxiv.org/abs/2611.11720v1) | Eyhab Al-Masri, Wenjun Yang | LLM agents increasingly act as routers for tools, APIs, and software-supply-chain workflows, making an execution path - rather than generated text alone - a security-relevant model output. We study shadow invocation: a task-preserving request variant that causes an LLM router to select a riskier execution path, potentially skipping validation, provenance, staging, redaction, or audit controls even though the request does not ask for a malicious action. We introduce ShadowBench-MCP, a benchmark and metric suite that separates task preservation (TPSS), operational pressure (OPSS), and route-geometric drift (RGAS), and combines them in the pre-execution SODS score. In a focused policy-ablation study using a Groq-hosted qwen/qwen3-32b router, 120 routing decisions yield 96 original-versus-variant comparisons across six security-sensitive workflows. Efficiency-first routing produces a 29.2% shadow-invocation rate and 26.7% safety-tool skip rate. Policy-aware routing reduces these to 4.2% and 1.7%, respectively, corresponding to an 85.7% relative reduction in shadow invocations. Pre-execution signals distinguish shadow routes with ROC AUC up to 0.94. These results are early, single-router evidence of a policy-dependent routing-layer security effect, not a claim of universal vulnerability. |
| 2026-11-09 | [Rapid Heat Removal from Chip Cores: Near-Junction Active Cooling of Power Transistors via Electron Entropy Transport and Spatial Decoupling](http://arxiv.org/abs/2611.11716v1) | Xiangying Li | Chip performance is limited by a shared bottleneck: the core produces heat faster than it can be moved out. We study the hardest case, RF power GaN HEMTs, whose hotspots exceed 5 kW/cm^2, and propose a horizontal near-junction active cooling topology built on electron entropy transport. Classical on-chip thermoelectric cooling faces a contest: Peltier absorption grows linearly with current, Joule heating with its square. In a vertical pillar, its own conduction resistance locks about half the Joule heat back toward the cold end, capping net cooling. From the Onsager framework we show the origin of this asymmetry: Peltier absorption localizes at the interface where the Seebeck coefficient jumps, whereas Joule heating fills the volume. The topology exploits this: a thermoelectric thin film lies flat on a diamond substrate beside the device drain (the electron exit), so bulk Joule heat leaves vertically into the substrate. A lateral fin model gives a closed form for the recirculation factor gamma: the vertical pillar yields gamma = 1, while the horizontal film yields gamma_max = sqrt(2) L_z/L_x, about 0.03 for L_z = 10 um, L_x = 500 um. The decoupling scales the net cooling ceiling and the critical runaway current by roughly 1/gamma. At equal resistance, the same film lifts maximum cooling from 13 mW to 500 mW, and the critical current from 0.65 A to 24 A. A time-constant and linear stability analysis with temperature-sensitive parameters defines the working window and safety margins for microsecond pulse cooling. The architecture leaves the device structure untouched; the principle, absorb at an interface and extract bulk heat vertically, applies to any chip with a hot core and offers a reproducible design path for near-junction thermal management. |
| 2026-11-08 | [PA-SaR: Perception-Aware Learning, Control, and Coordination for Multi-Robot Search and Rescue](http://arxiv.org/abs/2611.11648v1) | Jocubas Dimsa, Maria Grouev et al. | Autonomous SaR requires robots to explore unknown or partially known environments, avoid hazards, coordinate team activities, and guide discovered victims to safety. Improving individual capabilities of robots does not necessarily improve mission outcomes: safer motion can restrict the exploration behavior, faster discovery can increase the rescue demand, and effective guidance depends on victim responses. We introduce PA-SaR, a mathematical and architectural framework for coordinating heterogeneous local control, learning-based exploration, supervisory task allocation, and adaptive approach-lead victim guidance around mission-level objectives. PA- SaR distinguishes environmental ground-truth, robot-specific perceptions, and simulated victim behavior. This provides a common basis for integrating complementary mechanisms and analyzing their interactions across mission stages. We demonstrate the framework through single- and multi-robot simulation studies incorporating model-predictive and learning-based exploration, perception-based safety shielding, event-driven task allocation, and adaptive victim guidance. The results show trade-offs between collision avoidance and search completion, indicate that supervisory allocation improves search completion beyond information sharing alone, and demonstrate that increased exploration does not necessarily yield more rescues. Adaptive guidance increases rescues relative to continuous leading, without a detected improvement in mean or final victim trust or a reduction in mortality. These findings support evaluating local control, team coordination, and evacuation as related but distinct aspects of autonomous SaR within the considered simulation environments. |
| 2026-11-08 | [Weight-Space Programming of Policies and Their Verification via Model Checking](http://arxiv.org/abs/2611.11604v1) | Dennis Gross | Large language models (LLMs) can write the weights of small neural networks directly, without gradient descent. This weight-space programming creates policies for sequential decision-making from a task description alone, but nothing guarantees that a written policy behaves as intended. We combine weight-space programming with probabilistic model checking. Given an environment modeled as a Markov decision process and a requirement in probabilistic computation tree logic (PCTL), an LLM writes all weights of a policy network, optionally repairs them on a dataset of optimal actions, and receives model-checking feedback until the requirement holds or a round limit is reached. The model checker Storm computes, rather than estimates from simulations, the probability that the policy satisfies the requirement, so every returned policy is verified. With two LLMs on three environments, the created policy of each of the 18 runs attains the optimal probability, before any dataset or model-checking feedback. Four of six policies programmed for two transporter jobs also complete three jobs with probability 1, so they generalize to a task that the LLM never saw. The pipeline also programs a policy that satisfies a complex safety requirement expressed with the PCTL until operator in all six runs. In one of them, dataset repair raised the accuracy to 98.4\% but broke the requirement, and one round of model-checking feedback restored it. |

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



