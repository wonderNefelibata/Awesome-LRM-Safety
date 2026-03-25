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
| 2026-03-24 | [MedObvious: Exposing the Medical Moravec's Paradox in VLMs via Clinical Triage](http://arxiv.org/abs/2603.23501v1) | Ufaq Khan, Umair Nawaz et al. | Vision Language Models (VLMs) are increasingly used for tasks like medical report generation and visual question answering. However, fluent diagnostic text does not guarantee safe visual understanding. In clinical practice, interpretation begins with pre-diagnostic sanity checks: verifying that the input is valid to read (correct modality and anatomy, plausible viewpoint and orientation, and no obvious integrity violations). Existing benchmarks largely assume this step is solved, and therefore miss a critical failure mode: a model can produce plausible narratives even when the input is inconsistent or invalid. We introduce MedObvious, a 1,880-task benchmark that isolates input validation as a set-level consistency capability over small multi-panel image sets: the model must identify whether any panel violates expected coherence. MedObvious spans five progressive tiers, from basic orientation/modality mismatches to clinically motivated anatomy/viewpoint verification and triage-style cues, and includes five evaluation formats to test robustness across interfaces. Evaluating 17 different VLMs, we find that sanity checking remains unreliable: several models hallucinate anomalies on normal (negative-control) inputs, performance degrades when scaling to larger image sets, and measured accuracy varies substantially between multiple-choice and open-ended settings. These results show that pre-diagnostic verification remains unsolved for medical VLMs and should be treated as a distinct, safety-critical capability before deployment. |
| 2026-03-24 | [SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning](http://arxiv.org/abs/2603.23483v1) | Haoyu Huang, Jinfa Huang et al. | Agentic multimodal large language models (MLLMs) (e.g., OpenAI o3 and Gemini Agentic Vision) achieve remarkable reasoning capabilities through iterative visual tool invocation. However, the cascaded perception, reasoning, and tool-calling loops introduce significant sequential overhead. This overhead, termed agentic depth, incurs prohibitive latency and seriously limits system-level concurrency. To this end, we propose SpecEyes, an agentic-level speculative acceleration framework that breaks this sequential bottleneck. Our key insight is that a lightweight, tool-free MLLM can serve as a speculative planner to predict the execution trajectory, enabling early termination of expensive tool chains without sacrificing accuracy. To regulate this speculative planning, we introduce a cognitive gating mechanism based on answer separability, which quantifies the model's confidence for self-verification without requiring oracle labels. Furthermore, we design a heterogeneous parallel funnel that exploits the stateless concurrency of the small model to mask the stateful serial execution of the large model, maximizing system throughput. Extensive experiments on V* Bench, HR-Bench, and POPE demonstrate that SpecEyes achieves 1.1-3.35x speedup over the agentic baseline while preserving or even improving accuracy (up to +6.7%), thereby boosting serving throughput under concurrent workloads. |
| 2026-03-24 | [Information-Driven Active Perception for k-step Predictive Safety Monitoring](http://arxiv.org/abs/2603.23450v1) | Sumukha Udupa, Jie Fu | This work studies the synthesis of active perception policies for predictive safety monitoring in partially observable stochastic systems. Operating under strict sensing and communication budgets, the proposed monitor dynamically schedules sensor queries to maximize information gain about the safety of future states. The underlying stochastic dynamics are captured by a labeled hidden Markov model (HMM), with safety requirements defined by a deterministic finite automaton (DFA). To enable active information acquisition, we introduce minimizing k-step Shannon conditional entropy of the safety of future states as a planning objective, under the constraint of a limited sensor query budget. Using observable operators, we derive an efficient algorithm to compute the k-step conditional entropy and analyze key properties of the conditional entropy gradient with respect to policy parameters. We validate the effectiveness of the method for predictive safety monitoring through a dynamic congestion game example. |
| 2026-03-24 | [A Joint Reinforcement Learning Scheduling and Compression Framework for Teleoperated Driving](http://arxiv.org/abs/2603.23387v1) | Giacomo Avanzi, Marco Giordani et al. | Teleoperated driving (TD) is envisioned as a key application of future sixth generation (6G) networks. In this paradigm, connected vehicles transmit sensor-perception data to a remote (software) driver, which returns driving control commands to enhance traffic efficiency and road safety. This scenario imposes to maintain reliable and low-latency communication between the vehicle and the remote driver. To this aim, a promising solution is Predictive Quality of Service (PQoS), which provides mechanisms to estimate possible Quality of Service (QoS) degradation, and trigger timely network corrective actions accordingly. In particular, Reinforcement Learning (RL) agents can be trained to identify the optimal PQoS configuration. In this paper, we develop and implement two integrated RL agents that jointly determine (i) the optimal compression configuration for TD sensor data to balance the trade-off between transmission efficiency and data quality, and (ii) the optimal scheduling configuration to minimize the end-to-end latency by allocating radio resources according to different priority levels. We prove via full-stack ns-3 simulations that our integrated agents can deliver superior performance than any standalone model that only optimizes either compression or scheduling, especially in constrained or congested networks. While these agents can be deployed using either centralized or decentralized learning, we further propose a new meta-learning agent that dynamically selects the most appropriate strategy between the two based on current network conditions and application requirements. |
| 2026-03-24 | [Off-Policy Value-Based Reinforcement Learning for Large Language Models](http://arxiv.org/abs/2603.23355v1) | Peng-Yuan Wang, Ziniu Li et al. | Improving data utilization efficiency is critical for scaling reinforcement learning (RL) for long-horizon tasks where generating trajectories is expensive. However, the dominant RL methods for LLMs are largely on-policy: they update each batch of data only once, discard it, and then collect fresh samples, resulting in poor sample efficiency. In this work, we explore an alternative value-based RL framework for LLMs that naturally enables off-policy learning. We propose ReVal, a Bellman-update-based method that combines stepwise signals capturing internal consistency with trajectory-level signals derived from outcome verification. ReVal naturally supports replay-buffer-based training, allowing efficient reuse of past trajectories. Experiments on standard mathematical reasoning benchmarks show that ReVal not only converges faster but also outperforms GRPO in final performance. On DeepSeek-R1-Distill-1.5B, ReVal improves training efficiency and achieves improvement of 2.7% in AIME24 and 4.5% in out-of-domain benchmark GPQA over GRPO. These results suggest that value-based RL is a practical alternative to policy-based methods for LLM training. |
| 2026-03-24 | [Experimental Insights into the Limiting Mechanism of Vacancy Transport in Sodium Metal Anodes for Solid State Batteries](http://arxiv.org/abs/2603.23340v1) | Ansgar Lowack, Rafael Anton et al. | Ceramic solid-state batteries with sodium (Na) metal electrodes promise enhanced safety and energy density compared to contemporary secondary batteries. However, the critical delamination of the Na metal electrode during discharge - when vacancies accumulate at the Na/ceramic interface - remains to be understood and avoided. The study investigates the underlying mechanism by applying a linear current ramp between two Na metal electrodes separated by a ceramic solid electrolyte to provoke vacancy buildup. Above a critical current density $j_\mathrm{crit}$ the anode voltage no longer increases linearly but in an exponential fashion. Arrhenius analysis of $j_\mathrm{crit}(T)$ for the three solid electrolytes $\mathrm{Na_{1.9}Al_{10.67}Li_{0.33}O_{17}}$, $\mathrm{Na_{3.4}Zr_2Si_{2.4}P_{0.6}O_{12}}$, and $\mathrm{Na_5SmSi_4O_{12}}$ yields an activation energy $E_\mathrm{A}$ of $0.13$ to $0.15\,\mathrm{eV}$. This exceeds the activation energy of $0.053\,\mathrm{eV}$ for the diffusive vacancy migration in bulk Na significantly. Further, $E_\mathrm{A}$ is insensitive to anode microstructure variation. Both observations rule out bulk diffusion as the transport bottleneck. A thin tin-sodium alloy interlayer lowers $E_\mathrm{A}$ to $(0.10\pm0.01)\,\mathrm{eV}$, implicating interfacial thermodynamics as rate-limiting. Sodiophilic, Na-conducting interlayers and low-tension interfaces emerge as key pathways to stable, high-rate Na-SSBs at practical stack pressures. |
| 2026-03-24 | [Not All Tokens Are Created Equal: Query-Efficient Jailbreak Fuzzing for LLMs](http://arxiv.org/abs/2603.23269v1) | Wenyu Chen, Xiangtao Meng et al. | Large Language Models(LLMs) are widely deployed, yet are vulnerable to jailbreak prompts that elicit policy-violating outputs. Although prior studies have uncovered these risks, they typically treat all tokens as equally important during prompt mutation, overlooking the varying contributions of individual tokens to triggering model refusals. Consequently, these attacks introduce substantial redundant searching under query-constrained scenarios, reducing attack efficiency and hindering comprehensive vulnerability assessment. In this work, we conduct a token-level analysis of refusal behavior and observe that token contributions are highly skewed rather than uniform. Moreover, we find strong cross-model consistency in refusal tendencies, enabling the use of a surrogate model to estimate token-level contributions to the target model's refusals. Motivated by these findings, we propose TriageFuzz, a token-aware jailbreak fuzzing framework that adapts the fuzz testing approach with a series of customized designs. TriageFuzz leverages a surrogate model to estimate the contribution of individual tokens to refusal behaviors, enabling the identification of sensitive regions within the prompt. Furthermore, it incorporates a refusal-guided evolutionary strategy that adaptively weights candidate prompts with a lightweight scorer to steer the evolution toward bypassing safety constraints. Extensive experiments on six open-source LLMs and three commercial APIs demonstrate that TriageFuzz achieves comparable attack success rates (ASR) with significantly reduced query costs. Notably, it attains a 90% ASR with over 70% fewer queries compared to baselines. Even under an extremely restrictive budget of 25 queries, TriageFuzz outperforms existing methods, improving ASR by 20-40%. |
| 2026-03-24 | [SafeSeek: Universal Attribution of Safety Circuits in Language Models](http://arxiv.org/abs/2603.23268v1) | Miao Yu, Siyuan Fu et al. | Mechanistic interpretability reveals that safety-critical behaviors (e.g., alignment, jailbreak, backdoor) in Large Language Models (LLMs) are grounded in specialized functional components. However, existing safety attribution methods struggle with generalization and reliability due to their reliance on heuristic, domain-specific metrics and search algorithms. To address this, we propose \ourmethod, a unified safety interpretability framework that identifies functionally complete safety circuits in LLMs via optimization. Unlike methods focusing on isolated heads or neurons, \ourmethod introduces differentiable binary masks to extract multi-granular circuits through gradient descent on safety datasets, while integrates Safety Circuit Tuning to utilize these sparse circuits for efficient safety fine-tuning. We validate \ourmethod in two key scenarios in LLM safety: \textbf{(1) backdoor attacks}, identifying a backdoor circuit with 0.42\% sparsity, whose ablation eradicates the Attack Success Rate (ASR) from 100\% $\to$ 0.4\% while retaining over 99\% general utility; \textbf{(2) safety alignment}, localizing an alignment circuit with 3.03\% heads and 0.79\% neurons, whose removal spikes ASR from 0.8\% $\to$ 96.9\%, whereas excluding this circuit during helpfulness fine-tuning maintains 96.5\% safety retention. |
| 2026-03-24 | [SynForceNet: A Force-Driven Global-Local Latent Representation Framework for Lithium-Ion Battery Fault Diagnosis](http://arxiv.org/abs/2603.23265v1) | Rongxiu Chen, Yuting Su | Online safety fault diagnosis is essential for lithium-ion batteries in electric vehicles(EVs), particularly under complex and rare safety-critical conditions in real-world operation. In this work, we develop an online battery fault diagnosis network based on a deep anomaly detection framework combining kernel one-class classification and minimum-volume estimation. Mechanical constraints and spike-timing-dependent plasticity(STDP)-based dynamic representations are introduced to improve complex fault characterization and enable a more compact normal-state boundary. The proposed method is validated using 8.6 million valid data points collected from 20 EVs. Compared with several advanced baseline methods, it achieves average improvements of 7.59% in TPR, 27.92% in PPV, 18.28% in F1 score, and 23.68% in AUC. In addition, we analyze the spatial separation of fault representations before and after modeling, and further enhance framework robustness by learning the manifold structure in the latent space. The results also suggest the possible presence of shared causal structures across different fault types, highlighting the promise of integrating deep learning with physical constraints and neural dynamics for battery safety diagnosis. |
| 2026-03-24 | [Path Planning and Reinforcement Learning-Driven Control of On-Orbit Free-Flying Multi-Arm Robots](http://arxiv.org/abs/2603.23182v1) | Álvaro Belmonte-Baeza, José Luis Ramón et al. | This paper presents a hybrid approach that integrates trajectory optimization (TO) and reinforcement learning (RL) for motion planning and control of free-flying multi-arm robots in on-orbit servicing scenarios. The proposed system integrates TO for generating feasible, efficient paths while accounting for dynamic and kinematic constraints, and RL for adaptive trajectory tracking under uncertainties. The multi-arm robot design, equipped with thrusters for precise body control, enables redundancy and stability in complex space operations. TO optimizes arm motions and thruster forces, reducing reliance on the arms for stabilization and enhancing maneuverability. RL further refines this by leveraging model-free control to adapt to dynamic interactions and disturbances. The experimental results validated through comprehensive simulations demonstrate the effectiveness and robustness of the proposed hybrid approach. Two case studies are explored: surface motion with initial contact and a free-floating scenario requiring surface approximation. In both cases, the hybrid method outperforms traditional strategies. In particular, the thrusters notably enhance motion smoothness, safety, and operational efficiency. The RL policy effectively tracks TO-generated trajectories, handling high-dimensional action spaces and dynamic mismatches. This integration of TO and RL combines the strengths of precise, task-specific planning with robust adaptability, ensuring high performance in the uncertain and dynamic conditions characteristic of space environments. By addressing challenges such as motion coupling, environmental disturbances, and dynamic control requirements, this framework establishes a strong foundation for advancing the autonomy and effectiveness of space robotic systems. |
| 2026-03-24 | [Robust Safety Monitoring of Language Models via Activation Watermarking](http://arxiv.org/abs/2603.23171v1) | Toluwani Aremu, Daniil Ognev et al. | Large language models (LLMs) can be misused to reveal sensitive information, such as weapon-making instructions or writing malware. LLM providers rely on $\emph{monitoring}$ to detect and flag unsafe behavior during inference. An open security challenge is $\emph{adaptive}$ adversaries who craft attacks that simultaneously (i) evade detection while (ii) eliciting unsafe behavior. Adaptive attackers are a major concern as LLM providers cannot patch their security mechanisms, since they are unaware of how their models are being misused. We cast $\emph{robust}$ LLM monitoring as a security game, where adversaries who know about the monitor try to extract sensitive information, while a provider must accurately detect these adversarial queries at low false positive rates. Our work (i) shows that existing LLM monitors are vulnerable to adaptive attackers and (ii) designs improved defenses through $\emph{activation watermarking}$ by carefully introducing uncertainty for the attacker during inference. We find that $\emph{activation watermarking}$ outperforms guard baselines by up to $52\%$ under adaptive attackers who know the monitoring algorithm but not the secret key. |
| 2026-03-24 | [Describe-Then-Act: Proactive Agent Steering via Distilled Language-Action World Models](http://arxiv.org/abs/2603.23149v1) | Massimiliano Pappa, Luca Romani et al. | Deploying safety-critical agents requires anticipating the consequences of actions before they are executed. While world models offer a paradigm for this proactive foresight, current approaches relying on visual simulation incur prohibitive latencies, often exceeding several seconds per step. In this work, we challenge the assumption that visual processing is necessary for failure prevention. We show that a trained policy's latent state, combined with its planned actions, already encodes sufficient information to anticipate action outcomes, making visual simulation redundant for failure prevention. To this end, we introduce DILLO (DIstiLLed Language-ActiOn World Model), a fast steering layer that shifts the paradigm from "simulate-then-act" to "describe-then-act." DILLO is trained via cross-modal distillation, where a privileged Vision Language Model teacher annotates offline trajectories and a latent-conditioned Large Language Model student learns to predict semantic outcomes. This creates a text-only inference path, bypassing heavy visual generation entirely, achieving a 14x speedup over baselines. Experiments on MetaWorld and LIBERO demonstrate that DILLO produces high-fidelity descriptions of the next state and is able to steer the policy, improving episode success rate by up to 15 pp and 9.3 pp on average across tasks. |
| 2026-03-24 | [SMSP: A Plug-and-Play Strategy of Multi-Scale Perception for MLLMs to Perceive Visual Illusions](http://arxiv.org/abs/2603.23118v1) | Jinzhe Tu, Ruilei Guo et al. | Recent works have shown that Multimodal Large Language Models (MLLMs) are highly vulnerable to hidden-pattern visual illusions, where the hidden content is imperceptible to models but obvious to humans. This deficiency highlights a perceptual misalignment between current MLLMs and humans, and also introduces potential safety concerns. To systematically investigate this failure, we introduce IlluChar, a comprehensive and challenging illusion dataset, and uncover a key underlying mechanism for the models' failure: high-frequency attention bias, where the models are easily distracted by high-frequency background textures in illusion images, causing them to overlook hidden patterns. To address the issue, we propose the Strategy of Multi-Scale Perception (SMSP), a plug-and-play framework that aligns with human visual perceptual strategies. By suppressing distracting high-frequency backgrounds, SMSP generates images closer to human perception. Our experiments demonstrate that SMSP significantly improves the performance of all evaluated MLLMs on illusion images, for instance, increasing the accuracy of Qwen3-VL-8B-Instruct from 13.0% to 84.0%. Our work provides novel insights into MLLMs' visual perception, and offers a practical and robust solution to enhance it. Our code is publicly available at https://github.com/Tujz2023/SMSP. |
| 2026-03-24 | [Fault-Tolerant Design and Multi-Objective Model Checking for Real-Time Deep Reinforcement Learning Systems](http://arxiv.org/abs/2603.23113v1) | Guoxin Su, Thomas Robinson et al. | Deep reinforcement learning (DRL) has emerged as a powerful paradigm for solving complex decision-making problems. However, DRL-based systems still face significant dependability challenges particularly in real-time environments due to the simulation-to-reality gap, out-of-distribution observations, and the critical impact of latency. Latency-induced faults, in particular, can lead to unsafe or unstable behaviour, yet existing fault-tolerance approaches to DRL systems lack formal methods to rigorously analyse and optimise performance and safety simultaneously in real-time settings. To address this, we propose a formal framework for designing and analysing real-time switching mechanisms between DRL agents and alternative controllers. Our approach leverages Timed Automata (TAs) for explicit switch logic design, which is then syntactically converted to a Markov Decision Process (MDP) for formal analysis. We develop a novel convex query technique for multi-objective model checking, enabling the optimisation of soft performance objectives while ensuring hard safety constraints for MDPs. Furthermore, we present MOPMC, a GPU-accelerated software tool implementing this technique, demonstrating superior scalability in both model size and objective numbers. |
| 2026-03-24 | [When Language Models Lose Their Mind: The Consequences of Brain Misalignment](http://arxiv.org/abs/2603.23091v1) | Gabriele Merlin, Mariya Toneva | While brain-aligned large language models (LLMs) have garnered attention for their potential as cognitive models and for potential for enhanced safety and trustworthiness in AI, the role of this brain alignment for linguistic competence remains uncertain. In this work, we investigate the functional implications of brain alignment by introducing brain-misaligned models--LLMs intentionally trained to predict brain activity poorly while maintaining high language modeling performance. We evaluate these models on over 200 downstream tasks encompassing diverse linguistic domains, including semantics, syntax, discourse, reasoning, and morphology. By comparing brain-misaligned models with well-matched brain-aligned counterparts, we isolate the specific impact of brain alignment on language understanding. Our experiments reveal that brain misalignment substantially impairs downstream performance, highlighting the critical role of brain alignment in achieving robust linguistic competence. These findings underscore the importance of brain alignment in LLMs and offer novel insights into the relationship between neural representations and linguistic processing. |
| 2026-03-24 | [DariMis: Harm-Aware Modeling for Dari Misinformation Detection on YouTube](http://arxiv.org/abs/2603.22977v1) | Jawid Ahmad Baktash, Mosa Ebrahimi et al. | Dari, the primary language of Afghanistan, is spoken by tens of millions of people yet remains largely absent from the misinformation detection literature. We address this gap with DariMis, the first manually annotated dataset of 9,224 Dari-language YouTube videos, labeled across two dimensions: Information Type (Misinformation, Partly True, True) and Harm Level (Low, Medium, High). A central empirical finding is that these dimensions are structurally coupled, not independent: 55.9 percent of Misinformation carries at least Medium harm potential, compared with only 1.0 percent of True content. This enables Information Type classifiers to function as implicit harm-triage filters in content moderation pipelines.   We further propose a pair-input encoding strategy that represents the video title and description as separate BERT segment inputs, explicitly modeling the semantic relationship between headline claims and body content, a key signal of misleading information. An ablation study against single-field concatenation shows that pair-input encoding yields a 7.0 percentage point gain in Misinformation recall (60.1 percent to 67.1 percent), the safety-critical minority class, despite modest overall macro F1 differences (0.09 percentage points). We benchmark a Dari/Farsi-specialized model (ParsBERT) against XLM-RoBERTa-base; ParsBERT achieves the best test performance with accuracy of 76.60 percent and macro F1 of 72.77 percent. Bootstrap 95 percent confidence intervals are reported for all metrics, and we discuss both the practical significance and statistical limitations of the results. |
| 2026-03-24 | [From Morality Installation in LLMs to LLMs in Morality-as-a-System](http://arxiv.org/abs/2603.22944v1) | Gunter Bombaerts | Work on morality in large language models (LLMs) has progressed via constitutional AI, reinforcement learning from human feedback (RLHF) and systematic benchmarking, yet it still lacks tools to connect internal moral representations to regulatory obligations, to design cultural plurality across the full development stack, and to monitor how moral properties drift over the lifecycle of a deployed system. These difficulties reflect a shared root. Morality is installed in a model at training time. I propose instead a morality-as-a-system framework, grounded in Niklas Luhmann's social systems theory, that treats LLM morality as a dynamic, emergent property of a sociotechnical system. Moral behaviour in a deployed LLM is not fixed at training. It is continuously reproduced through interactions among seven structurally coupled components spanning the neural substrate, training data, alignment procedures, system prompts, moderation, runtime dynamics, and user interface. This is a conceptual framework paper, not an empirical study. It philosophically reframes three known challenges, the interpretability-governance gap, the cross-component plurality problem, and the absence of lifecycle monitoring, as structural coupling failures that the installation paradigm cannot diagnose. For technical researchers, it explores three illustrative hypotheses about cross-component representational inconsistency, representation-level drift as an early safety signal, and the governance advantage of lifecycle monitoring. For philosophers and governance specialists, it offers a vocabulary for specifying substrate-level monitoring obligations within existing governance frameworks. The morality-as-a-system framework does not displace elements such as constitutional AI or RLHF it embeds them within a larger temporal and structural account and specifies the additional infrastructure those methods require. |
| 2026-03-24 | [Balancing Safety and Efficiency in Aircraft Health Diagnosis: A Task Decomposition Framework with Heterogeneous Long-Micro Scale Cascading and Knowledge Distillation-based Interpretability](http://arxiv.org/abs/2603.22885v1) | Xinhang Chen, Zhihuan Wei et al. | Whole-aircraft diagnosis for general aviation faces threefold challenges: data uncertainty, task heterogeneity, and computational inefficiency. Existing end-to-end approaches uniformly model health discrimination and fault characterization, overlooking intrinsic receptive field conflicts between global context modeling and local feature extraction, while incurring prohibitive training costs under severe class imbalance. To address these, this study proposes the Diagnosis Decomposition Framework (DDF), explicitly decoupling diagnosis into Anomaly Detection (AD) and Fault Classification (FC) subtasks via the Long-Micro Scale Diagnostician (LMSD). Employing a "long-range global screening and micro-scale local precise diagnosis" strategy, LMSD utilizes Convolutional Tokenizer with Multi-Head Self-Attention (ConvTokMHSA) for global operational pattern discrimination and Multi-Micro Kernel Network (MMK Net) for local fault feature extraction. Decoupled training separates "large-sample lightweight" and "small-sample complex" optimization pathways, significantly reducing computational overhead. Concurrently, Keyness Extraction Layer (KEL) via knowledge distillation furnishes physically traceable explanations for two-stage decisions, materializing interpretability-by-design. Experiments on the NGAFID real-world aviation dataset demonstrate approximately 4-8% improvement in Multi-Class Weighted Penalty Metric (MCWPM) over baselines with substantially reduced training time, validating comprehensive advantages in task adaptability, interpretability, and efficiency. This provides a deployable methodology for general aviation health management. |
| 2026-03-24 | [TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](http://arxiv.org/abs/2603.22882v1) | Chunxiao Li, Lijun Li et al. | The rapid advancement of Vision-Language Models (VLMs) has brought their safety vulnerabilities into sharp focus. However, existing red teaming methods are fundamentally constrained by an inherent linear exploration paradigm, confining them to optimizing within a predefined strategy set and preventing the discovery of novel, diverse exploits. To transcend this limitation, we introduce TreeTeaming, an automated red teaming framework that reframes strategy exploration from static testing to a dynamic, evolutionary discovery process. At its core lies a strategic Orchestrator, powered by a Large Language Model (LLM), which autonomously decides whether to evolve promising attack paths or explore diverse strategic branches, thereby dynamically constructing and expanding a strategy tree. A multimodal actuator is then tasked with executing these complex strategies. In the experiments across 12 prominent VLMs, TreeTeaming achieves state-of-the-art attack success rates on 11 models, outperforming existing methods and reaching up to 87.60\% on GPT-4o. The framework also demonstrates superior strategic diversity over the union of previously public jailbreak strategies. Furthermore, the generated attacks exhibit an average toxicity reduction of 23.09\%, showcasing their stealth and subtlety. Our work introduces a new paradigm for automated vulnerability discovery, underscoring the necessity of proactive exploration beyond static heuristics to secure frontier AI models. |
| 2026-03-24 | [Agent-Sentry: Bounding LLM Agents via Execution Provenance](http://arxiv.org/abs/2603.22868v1) | Rohan Sequeira, Stavros Damianakis et al. | Agentic computing systems, which autonomously spawn new functionalities based on natural language instructions, are becoming increasingly prevalent. While immensely capable, these systems raise serious security, privacy, and safety concerns. Fundamentally, the full set of functionalities offered by these systems, combined with their probabilistic execution flows, is not known beforehand. Given this lack of characterization, it is non-trivial to validate whether a system has successfully carried out the user's intended task or instead executed irrelevant actions, potentially as a consequence of compromise. In this paper, we propose Agent-Sentry, a framework that attempts to bound agentic systems to address this problem. Our key insight is that agentic systems are designed for specific use cases and therefore need not expose unbounded or unspecified functionalities. Once bounded, these systems become easier to scrutinize. Agent-Sentry operationalizes this insight by uncovering frequent functionalities offered by an agentic system, along with their execution traces, to construct behavioral bounds. It then learns a policy from these traces and blocks tool calls that deviate from learned behaviors or that misalign with user intent. Our evaluation shows that Agent-Sentry helps prevent over 90\% of attacks that attempt to trigger out-of-bounds executions, while preserving up to 98\% of system utility. |

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



