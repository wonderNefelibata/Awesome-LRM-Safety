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
| 2026-03-16 | [From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation](http://arxiv.org/abs/2603.15600v1) | Yibin Liu, Yaxing Lyu et al. | Accurate process supervision remains a critical challenge for long-horizon robotic manipulation. A primary bottleneck is that current video MLLMs, trained primarily under a Supervised Fine-Tuning (SFT) paradigm, function as passive "Observers" that recognize ongoing events rather than evaluating the current state relative to the final task goal. In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics". We leverage outcome-based Reinforcement Learning to incentivize explicit Chain-of-Thought generation for progress estimation. Furthermore, our architecture constructs a structured temporal input by explicitly anchoring the video sequence between initial and current state images. Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance. Quantitatively, our 7B model achieves a 50% reduction in the mean absolute error of specialized reasoning baselines, demonstrating significant relative accuracy improvements over 72B-scale general MLLMs. Furthermore, PRIMO R1 exhibits strong zero-shot generalization on difficult failure detection tasks. We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%. |
| 2026-03-16 | [Severe Domain Shift in Skeleton-Based Action Recognition:A Study of Uncertainty Failure in Real-World Gym Environments](http://arxiv.org/abs/2603.15574v1) | Aaditya Khanal, Junxiu Zhou | The practical deployment gap -- transitioning from controlled multi-view 3D skeleton capture to   unconstrained monocular 2D pose estimation -- introduces a compound domain shift whose safety implications   remain critically underexplored. We present a systematic study of this severe domain shift using a novel   Gym2D dataset (style/viewpoint shift) and the UCF101 dataset (semantic shift). Our Skeleton Transformer   achieves 63.2% cross-subject accuracy on NTU-120 but drops to 1.6% under zero-shot transfer to the Gym   domain and 1.16% on UCF101. Critically, we demonstrate that high Out-Of-Distribution (OOD) detection AUROC   does not guarantee safe selective classification. Standard uncertainty methods fail to detect this   performance drop: the model remains confidently incorrect with 99.6% risk even at 50% coverage across both   OOD datasets. While energy-based scoring (AUROC >= 0.91) and Mahalanobis distance provide reliable   distributional detection signals, such high AUROC scores coexist with poor risk-coverage behavior when   making decisions. A lightweight finetuned gating mechanism restores calibration and enables graceful   abstention, substantially reducing the rate of confident wrong predictions. Our work challenges standard   deployment assumptions, providing a principled safety analysis of both semantic and geometric skeleton   recognition deployment. |
| 2026-03-16 | [Are Dilemmas and Conflicts in LLM Alignment Solvable? A View from Priority Graph](http://arxiv.org/abs/2603.15527v1) | Zhenheng Tang, Xiang Liu et al. | As Large Language Models (LLMs) become more powerful and autonomous, they increasingly face conflicts and dilemmas in many scenarios. We first summarize and taxonomize these diverse conflicts. Then, we model the LLM's preferences to make different choices as a priority graph, where instructions and values are nodes, and the edges represent context-specific priorities determined by the model's output distribution. This graph reveals that a unified stable LLM alignment is very challenging, because the graph is neither static nor necessarily consistent in different contexts. Besides, it also reveals a potential vulnerability: priority hacking, where adversaries can craft deceptive contexts to manipulate the graph and bypass safety alignments. To counter this, we propose a runtime verification mechanism, enabling LLMs to query external sources to ground their context and resist manipulation. While this approach enhances robustness, we also acknowledge that many ethical and value dilemmas are philosophically irreducible, posing a long-term, open challenge for the future of AI alignment. |
| 2026-03-16 | [Evasive Intelligence: Lessons from Malware Analysis for Evaluating AI Agents](http://arxiv.org/abs/2603.15457v1) | Simone Aonzo, Merve Sahin et al. | Artificial intelligence (AI) systems are increasingly adopted as tool-using agents that can plan, observe their environment, and take actions over extended time periods. This evolution challenges current evaluation practices where the AI models are tested in restricted, fully observable settings. In this article, we argue that evaluations of AI agents are vulnerable to a well-known failure mode in computer security: malicious software that exhibits benign behavior when it detects that it is being analyzed. We point out how AI agents can infer the properties of their evaluation environment and adapt their behavior accordingly. This can lead to overly optimistic safety and robustness assessments. Drawing parallels with decades of research on malware sandbox evasion, we demonstrate that this is not a speculative concern, but rather a structural risk inherent to the evaluation of adaptive systems. Finally, we outline concrete principles for evaluating AI agents, which treat the system under test as potentially adversarial. These principles emphasize realism, variability of test conditions, and post-deployment reassessment. |
| 2026-03-16 | [Amplification Effects in Test-Time Reinforcement Learning: Safety and Reasoning Vulnerabilities](http://arxiv.org/abs/2603.15417v1) | Vanshaj Khattar, Md Rafi ur Rashid et al. | Test-time training (TTT) has recently emerged as a promising method to improve the reasoning abilities of large language models (LLMs), in which the model directly learns from test data without access to labels. However, this reliance on test data also makes TTT methods vulnerable to harmful prompt injections. In this paper, we investigate safety vulnerabilities of TTT methods, where we study a representative self-consistency-based test-time learning method: test-time reinforcement learning (TTRL), a recent TTT method that improves LLM reasoning by rewarding self-consistency using majority vote as a reward signal. We show that harmful prompt injection during TTRL amplifies the model's existing behaviors, i.e., safety amplification when the base model is relatively safe, and harmfulness amplification when it is vulnerable to the injected data. In both cases, there is a decline in reasoning ability, which we refer to as the reasoning tax. We also show that TTT methods such as TTRL can be exploited adversarially using specially designed "HarmInject" prompts to force the model to answer jailbreak and reasoning queries together, resulting in stronger harmfulness amplification. Overall, our results highlight that TTT methods that enhance LLM reasoning by promoting self-consistency can lead to amplification behaviors and reasoning degradation, highlighting the need for safer TTT methods. |
| 2026-03-16 | [TrinityGuard: A Unified Framework for Safeguarding Multi-Agent Systems](http://arxiv.org/abs/2603.15408v1) | Kai Wang, Biaojie Zeng et al. | With the rapid development of LLM-based multi-agent systems (MAS), their significant safety and security concerns have emerged, which introduce novel risks going beyond single agents or LLMs. Despite attempts to address these issues, the existing literature lacks a cohesive safeguarding system specialized for MAS risks. In this work, we introduce TrinityGuard, a comprehensive safety evaluation and monitoring framework for LLM-based MAS, grounded in the OWASP standards. Specifically, TrinityGuard encompasses a three-tier fine-grained risk taxonomy that identifies 20 risk types, covering single-agent vulnerabilities, inter-agent communication threats, and system-level emergent hazards. Designed for scalability across various MAS structures and platforms, TrinityGuard is organized in a trinity manner, involving an MAS abstraction layer that can be adapted to any MAS structures, an evaluation layer containing risk-specific test modules, alongside runtime monitor agents coordinated by a unified LLM Judge Factory. During Evaluation, TrinityGuard executes curated attack probes to generate detailed vulnerability reports for each risk type, where monitor agents analyze structured execution traces and issue real-time alerts, enabling both pre-development evaluation and runtime monitoring. We further formalize these safety metrics and present detailed case studies across various representative MAS examples, showcasing the versatility and reliability of TrinityGuard. Overall, TrinityGuard acts as a comprehensive framework for evaluating and monitoring various risks in MAS, paving the way for further research into their safety and security. |
| 2026-03-16 | [Detection of Autonomous Shuttles in Urban Traffic Images Using Adaptive Residual Context](http://arxiv.org/abs/2603.15404v1) | Mohamed Aziz Younes, Nicolas Saunier et al. | The progressive automation of transport promises to enhance safety and sustainability through shared mobility. Like other vehicles and road users, and even more so for such a new technology, it requires monitoring to understand how it interacts in traffic and to evaluate its safety. This can be done with fixed cameras and video object detection. However, the addition of new detection targets generally requires a fine-tuning approach for regular detection methods. Unfortunately, this implementation strategy will lead to a phenomenon known as catastrophic forgetting, which causes a degradation in scene understanding. In road safety applications, preserving contextual scene knowledge is of the utmost importance for protecting road users. We introduce the Adaptive Residual Context (ARC) architecture to address this. ARC links a frozen context branch and trainable task-specific branches through a Context-Guided Bridge, utilizing attention to transfer spatial features while preserving pre-trained representations. Experiments on a custom dataset show that ARC matches fine-tuned baselines while significantly improving knowledge retention, offering a data-efficient solution to add new vehicle categories for complex urban environments. |
| 2026-03-16 | [SFCoT: Safer Chain-of-Thought via Active Safety Evaluation and Calibration](http://arxiv.org/abs/2603.15397v1) | Yu Pan, Wenlong Yu et al. | Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks. However, they remain highly susceptible to jailbreak attacks that undermine their safety alignment. Existing defense mechanisms typically rely on post hoc filtering applied only to the final output, leaving intermediate reasoning steps unmonitored and vulnerable to adversarial manipulation. To address this gap, this paper proposes a SaFer Chain-of-Thought (SFCoT) framework, which proactively evaluates and calibrates potentially unsafe reasoning steps in real time. SFCoT incorporates a three-tier safety scoring system alongside a multi-perspective consistency verification mechanism, designed to detect potential risks throughout the reasoning process. A dynamic intervention module subsequently performs targeted calibration to redirect reasoning trajectories toward safe outcomes. Experimental results demonstrate that SFCoT reduces the attack success rate from $58.97\%$ to $12.31\%$, demonstrating it as an effective and efficient LLM safety enhancement method without a significant decline in general performance. |
| 2026-03-16 | [CRASH: Cognitive Reasoning Agent for Safety Hazards in Autonomous Driving](http://arxiv.org/abs/2603.15364v1) | Erick Silva, Rehana Yasmin et al. | As AVs grow in complexity and diversity, identifying the root causes of operational failures has become increasingly complex. The heterogeneity of system architectures across manufacturers, ranging from end-to-end to modular designs, together with variations in algorithms and integration strategies, limits the standardization of incident investigations and hinders systematic safety analysis. This work examines real-world AV incidents reported in the NHTSA database. We curate a dataset of 2,168 cases reported between 2021 and 2025, representing more than 80 million miles driven. To process this data, we introduce CRASH, Cognitive Reasoning Agent for Safety Hazards, an LLM-based agent that automates reasoning over crash reports by leveraging both standardized fields and unstructured narrative descriptions. CRASH operates on a unified representation of each incident to generate concise summaries, attribute a primary cause, and assess whether the AV materially contributed to the event. Our findings show that (1) CRASH attributes 64% of incidents to perception or planning failures, underscoring the importance of reasoning-based analysis for accurate fault attribution; and (2) approximately 50% of reported incidents involve rear-end collisions, highlighting a persistent and unresolved challenge in autonomous driving deployment. We further validate CRASH with five domain experts, achieving 86% accuracy in attributing AV system failures. Overall, CRASH demonstrates strong potential as a scalable and interpretable tool for automated crash analysis, providing actionable insights to support safety research and the continued development of autonomous driving systems. |
| 2026-03-16 | [A Kolmogorov-Arnold Surrogate Model for Chemical Equilibria: Application to Solid Solutions](http://arxiv.org/abs/2603.15307v1) | Leonardo Boledi, Dirk Bosbach et al. | The computational cost of geochemical solvers is a challenging matter. For reactive transport simulations, where chemical calculations are performed up to billions of times, it is crucial to reduce the total computational time. Existing publications have explored various machine-learning approaches to determine the most effective data-driven surrogate model. In particular, multilayer perceptrons are widely employed due to their ability to recognize nonlinear relationships. In this work, we focus on the recent Kolmogorov-Arnold networks, where learnable spline-based functions replace classical fixed activation functions. This architecture has achieved higher accuracy with fewer trainable parameters and has become increasingly popular for solving partial differential equations. First, we train a surrogate model based on an existing cement system benchmark. Then, we move to an application case for the geological disposal of nuclear waste, i.e., the determination of radionuclide-bearing solids solubilities. To the best of our knowledge, this work is the first to investigate co-precipitation with radionuclide incorporation using data-driven surrogate models, considering increasing levels of thermodynamic complexity from simple mechanical mixtures to non-ideal solid solutions of binary (Ba,Ra)SO$_4$ and ternary (Sr,Ba,Ra)SO$_4$ systems. On the cement benchmark, we demonstrate that the Kolmogorov-Arnold architecture outperforms multilayer perceptrons in both absolute and relative error metrics, reducing them by 62% and 59%, respectively. On the binary and ternary radium solid solution models, Kolmogorov-Arnold networks maintain median prediction errors near $1\times10^{-3}$. This is the first step toward employing surrogate models to speed up reactive transport simulations and optimize the safety assessment of deep geological waste repositories. |
| 2026-03-16 | [ReLU Barrier Functions for Nonlinear Systems with Constrained Control: A Union of Invariant Sets Approach](http://arxiv.org/abs/2603.15286v1) | Pouya Samanipour, Hasan A. Poonawala | Certifying safety for nonlinear systems with polytopic input constraints is challenging because CBF synthesis must ensure control admissibility under saturation. We propose an approximation--verification pipeline that performs convex barrier synthesis on piecewise-affine (PWA) surrogates and certifies safety for the original nonlinear system via facet-wise verification. To reduce conservatism while preserving tractability, we use a two-slope Leaky ReLU surrogate for the extended class-$\mathcal{K}$ function $α(\cdot)$ and combine multiple certificates using a Union of Invariant Sets (UIS). Counterexamples are handled through local uncertainty updates. Simulations on pendulum and cart-pole systems with input saturation show larger certified invariant sets than linear-$α$ designs with tractable computation time. |
| 2026-03-16 | [Algorithms for Deciding the Safety of States in Fully Observable Non-deterministic Problems: Technical Report](http://arxiv.org/abs/2603.15282v1) | Johannes Schmalz, Chaahat Jain | Learned action policies are increasingly popular in sequential decision-making, but suffer from a lack of safety guarantees. Recent work introduced a pipeline for testing the safety of such policies under initial-state and action-outcome non-determinism. At the pipeline's core, is the problem of deciding whether a state is safe (a safe policy exists from the state) and finding faults, which are state-action pairs that transition from a safe state to an unsafe one. Their most effective algorithm for deciding safety, TarjanSafe, is effective on their benchmarks, but we show that it has exponential worst-case runtime with respect to the state space. A linear-time alternative exists, but it is slower in practice. We close this gap with a new policy-iteration algorithm iPI, that combines the best of both: it matches TarjanSafe's best-case runtime while guaranteeing a polynomial worst-case. Experiments confirm our theory and show that in problems amenable to TarjanSafe iPI has similar performance, whereas in ill-suited problems iPI scales exponentially better. |
| 2026-03-16 | [Directional Embedding Smoothing for Robust Vision Language Models](http://arxiv.org/abs/2603.15259v1) | Ye Wang, Jing Liu et al. | The safety and reliability of vision-language models (VLMs) are a crucial part of deploying trustworthy agentic AI systems. However, VLMs remain vulnerable to jailbreaking attacks that undermine their safety alignment to yield harmful outputs. In this work, we extend the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense to VLMs and evaluate its performance against the JailBreakV-28K benchmark of multi-modal jailbreaking attacks. We find that RESTA is effective in reducing attack success rate over this diverse corpus of attacks, in particular, when employing directional embedding noise, where the injected noise is aligned with the original token embedding vectors. Our results demonstrate that RESTA can contribute to securing VLMs within agentic systems, as a lightweight, inference-time defense layer of an overall security framework. |
| 2026-03-16 | [HapticVLA: Contact-Rich Manipulation via Vision-Language-Action Model without Inference-Time Tactile Sensing](http://arxiv.org/abs/2603.15257v1) | Konstantin Gubernatorov, Mikhail Sannikov et al. | Tactile sensing is a crucial capability for Vision-Language-Action (VLA) architectures, as it enables dexterous and safe manipulation in contact-rich tasks. However, reliance on dedicated tactile hardware increases cost and reduces reproducibility across robotic platforms. We argue that tactile-aware manipulation can be learned offline and deployed without direct haptic feedback at inference. To this end, we present HapticVLA, which proceeds in two tightly coupled stages: Safety-Aware Reward-Weighted Flow Matching (SA-RWFM) and Tactile Distillation (TD). SA-RWFM trains a flow-matching action expert that incorporates precomputed, safety-aware tactile rewards penalizing excessive grasping force and suboptimal grasping trajectories. TD further transfers this tactile-aware capability into a conventional VLA: we distill a compact tactile token from the SA-RWFM teacher and train a student VLA to predict that token from vision and state modalities, enabling tactile-aware action generation at inference without requiring on-board tactile sensors. This design preserves contact-rich tactile-aware reasoning within VLA while removing the need for on-board tactile sensors during deployment. On real-world experiments, HapticVLA achieves a mean success rate of 86.7%, consistently outperforming baseline VLAs - including versions provided with direct tactile feedback during inference. |
| 2026-03-16 | [Why the Valuable Capabilities of LLMs Are Precisely the Unexplainable Ones](http://arxiv.org/abs/2603.15238v1) | Quan Cheng | This paper proposes and argues for a counterintuitive thesis: the truly valuable capabilities of large language models (LLMs) reside precisely in the part that cannot be fully captured by human-readable discrete rules. The core argument is a proof by contradiction via expert system equivalence: if the full capabilities of an LLM could be described by a complete set of human-readable rules, then that rule set would be functionally equivalent to an expert system; but expert systems have been historically and empirically demonstrated to be strictly weaker than LLMs; therefore, a contradiction arises -- the capabilities of LLMs that exceed those of expert systems are exactly the capabilities that cannot be rule-encoded. This thesis is further supported by the Chinese philosophical concept of Wu (sudden insight through practice), the historical failure of expert systems, and a structural mismatch between human cognitive tools and complex systems. The paper discusses implications for interpretability research, AI safety, and scientific epistemology. |
| 2026-03-16 | [ADV-0: Closed-Loop Min-Max Adversarial Training for Long-Tail Robustness in Autonomous Driving](http://arxiv.org/abs/2603.15221v1) | Tong Nie, Yihong Tang et al. | Deploying autonomous driving systems requires robustness against long-tail scenarios that are rare but safety-critical. While adversarial training offers a promising solution, existing methods typically decouple scenario generation from policy optimization and rely on heuristic surrogates. This leads to objective misalignment and fails to capture the shifting failure modes of evolving policies. This paper presents ADV-0, a closed-loop min-max optimization framework that treats the interaction between driving policy (defender) and adversarial agent (attacker) as a zero-sum Markov game. By aligning the attacker's utility directly with the defender's objective, we reveal the optimal adversary distribution. To make this tractable, we cast dynamic adversary evolution as iterative preference learning, efficiently approximating this optimum and offering an algorithm-agnostic solution to the game. Theoretically, ADV-0 converges to a Nash Equilibrium and maximizes a certified lower bound on real-world performance. Experiments indicate that it effectively exposes diverse safety-critical failures and greatly enhances the generalizability of both learned policies and motion planners against unseen long-tail risks. |
| 2026-03-16 | [Token Coherence: Adapting MESI Cache Protocols to Minimize Synchronization Overhead in Multi-Agent LLM Systems](http://arxiv.org/abs/2603.15183v1) | Vladyslav Parakhin | Multi-agent LLM orchestration incurs synchronization costs scaling as O(n x S x |D|) in agents, steps, and artifact size under naive broadcast -- a regime I term broadcast-induced triply-multiplicative overhead. I argue this pathology is a structural residue of full-state rebroadcast, not an inherent property of multi-agent coordination.   The central claim: synchronization cost explosion in LLM multi-agent systems maps with formal precision onto the cache coherence problem in shared-memory multiprocessors, and MESI-protocol invalidation transfers to artifact synchronization under minimal structural modification.   I construct the Artifact Coherence System (ACS) and prove the Token Coherence Theorem: lazy invalidation attenuates cost by at least S/(n + W(d_i)) when S > n + W(d_i), converting O(n x S x |D|) to O((n + W) x |D|). A TLA+-verified protocol enforces single-writer safety, monotonic versioning, and bounded staleness across ~2,400 explored states.   Simulation across four workload configurations yields token savings of 95.0% +/- 1.3% at V=0.05, 92.3% +/- 1.4% at V=0.10, 88.3% +/- 1.5% at V=0.25, and 84.2% +/- 1.3% at V=0.50 -- each exceeding the theorem's conservative lower bounds. Savings of ~81% persist at V=0.9, contrary to the predicted collapse threshold.   Contributions: (1) formal MESI-to-artifact state mapping; (2) Token Coherence Theorem as savings lower bound; (3) TLA+-verified protocol with three proven invariants; (4) characterization of conditional artifact access semantics resolving the always-read objection; (5) reference Python implementation integrating with LangGraph, CrewAI, and AutoGen via thin adapter layers. |
| 2026-03-16 | [Iterative Learning Control-Informed Reinforcement Learning for Batch Process Control](http://arxiv.org/abs/2603.15180v1) | Runze Lin, Ziqi Zhuo et al. | A significant limitation of Deep Reinforcement Learning (DRL) is the stochastic uncertainty in actions generated during exploration-exploitation, which poses substantial safety risks during both training and deployment. In industrial process control, the lack of formal stability and convergence guarantees further inhibits adoption of DRL methods by practitioners. Conversely, Iterative Learning Control (ILC) represents a well-established autonomous control methodology for repetitive systems, particularly in batch process optimization. ILC achieves desired control performance through iterative refinement of control laws, either between consecutive batches or within individual batches, to compensate for both repetitive and non-repetitive disturbances. This study introduces an Iterative Learning Control-Informed Reinforcement Learning (IL-CIRL) framework for training DRL controllers in dual-layer batch-to-batch and within-batch control architectures for batch processes. The proposed method incorporates Kalman filter-based state estimation within the iterative learning structure to guide DRL agents toward control policies that satisfy operational constraints and ensure stability guarantees. This approach enables the systematic design of DRL controllers for batch processes operating under multiple disturbance conditions. |
| 2026-03-16 | [Multi-Scale Control of Large Agent Populations: From Density Dynamics to Individual Actuation](http://arxiv.org/abs/2603.15160v1) | Mario di Bernardo | We review a body of recent work by the author and collaborators on controlling the spatial organisation of large agent populations across multiple scales. A central theme is the systematic bridging of microscopic agent-level dynamics and macroscopic density descriptions, enabling control design at the most natural level of abstraction and subsequent translation across scales. We show how this multi-scale perspective provides a unified approach to both \emph{direct control}, where every agent is actuated, and \emph{indirect control}, where few leaders or herders steer a larger uncontrolled population. The review covers continuification-based control with robustness under limited sensing and decentralised implementation via distributed density estimation; leader--follower density regulation with dual-feedback stability guarantees and bio-inspired plasticity; optimal-transport methods for coverage control and macro-to-micro discretisation; nonreciprocal field theory for collective decision-making; mean-field control barrier functions for population-level safety; and hierarchical reinforcement learning for settings where closed-form solutions are intractable. Together, these results demonstrate the breadth and versatility of a multi-scale control framework that integrates analytical methods, learning, and physics-inspired approaches for large agent populations. |
| 2026-03-16 | [Master Micro Residual Correction with Adaptive Tactile Fusion and Force-Mixed Control for Contact-Rich Manipulation](http://arxiv.org/abs/2603.15152v1) | Xingting Li, Yifan Xie et al. | Robotic contact-rich and fine-grained manipulation remains a significant challenge due to complex interaction dynamics and the competing requirements of multi-timescale control. While current visual imitation learning methods excel at long-horizon planning, they often fail to perceive critical interaction cues like friction variations or incipient slip, and struggle to balance global task coherence with local reactive feedback. To address these challenges, we propose M2-ResiPolicy, a novel Master-Micro residual control architecture that synergizes high-level action guidance with low-level correction. The framework consists of a Master-Guidance Policy (MGP) operating at 10 Hz, which generates temporally consistent action chunks via a diffusion-based backbone and employs a tactile-intensity-driven adaptive fusion mechanism to dynamically modulate perceptual weights between vision and touch. Simultaneously, a high-frequency (60 Hz) Micro-Residual Corrector (MRC) utilizes a lightweight GRU to provide real-time action compensation based on TCP wrench feedback. This policy is further integrated with a force-mixed PBIC execution layer, effectively regulating contact forces to ensure interaction safety. Experiments across several demanding tasks including fragile object grasping and precision insertion, demonstrate that M2-ResiPolicy significantly outperforms standard Diffusion Policy (DP) and state-of-the-art Reactive Diffusion Policy (RDP), achieving a 93\% damage-free success rate in chip grasping and superior force regulation stability. |

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



