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
| 2026-04-28 | [No Pedestrian Left Behind: Real-Time Detection and Tracking of Vulnerable Road Users for Adaptive Traffic Signal Control](http://arxiv.org/abs/2604.25887v1) | Anas Gamal Aly, Hala ElAarag | Current pedestrian crossing signals operate on fixed timing without adjustment to pedestrian behavior, which can leave vulnerable road users (VRUs) such as the elderly, disabled, or distracted pedestrians stranded when the light changes. We introduce No Pedestrian Left Behind (NPLB), a real-time adaptive traffic signal system that monitors VRUs in crosswalks and automatically extends signal timing when needed. We evaluated five state-of-the-art object detection models on the BGVP dataset, with YOLOv12 achieving the highest mean Average Precision at 50% (mAP@0.5) of 0.756. NPLB integrates our fine-tuned YOLOv12 with ByteTrack multi-object tracking and an adaptive controller that extends pedestrian phases when remaining time falls below a critical threshold. Through 10,000 Monte Carlo simulations, we demonstrate that NPLB improves VRU safety by 71.4%, reducing stranding rates from 9.10% to 2.60%, while requiring signal extensions in only 12.1% of crossing cycles. |
| 2026-04-28 | [SIEVES: Selective Prediction Generalizes through Visual Evidence Scoring](http://arxiv.org/abs/2604.25855v1) | Hector G. Rodriguez, Marcus Rohrbach | Multimodal large language models (MLLMs) achieve ever-stronger performance on visual-language tasks. Even as traditional visual question answering benchmarks approach saturation, reliable deployment requires satisfying low error tolerances in real-world out-of-distribution (OOD) scenarios. Precisely, selective prediction aims to improve coverage, i.e. the share of inputs the system answers, while adhering to a user-defined risk level. This is typically achieved by assigning a confidence score to each answer and abstaining on those that fall below a certain threshold. To enable reliable generalization, we require reasoner models to produce localized visual evidence while answering, and design a selector that explicitly learns to estimate the quality of the localization provided by the reasoner. We show that SIEVES (Selective Prediction through Visual Evidence Scoring) improves coverage by up to three times on challenging OOD benchmarks (V* Bench, HR-Bench-8k, MME-RealWorld-Lite, VizWiz, and AdVQA), compared to non-grounding baselines. Beyond better generalization to OOD tasks, the design of the SIEVES selector enables transfer to proprietary reasoners without access to their weights or logits, such as o3 and Gemini-3-Pro, providing coverage boosts beyond those attributable to accuracy alone. We highlight that SIEVES generalizes across all five tested OOD datasets and reasoner models (Pixel-Reasoner, o3, and Gemini-3-Pro), without benchmark- or reasoner-specific training or adaptation. |
| 2026-04-28 | [PSI-Bench: Towards Clinically Grounded and Interpretable Evaluation of Depression Patient Simulators](http://arxiv.org/abs/2604.25840v1) | Nguyen Khoi Hoang, Shuhaib Mehri et al. | Patient simulators are gaining traction in mental health training by providing scalable exposure to complex and sensitive patient interactions. Simulating depressed patients is particularly challenging, as safety constraints and high patient variability complicate simulations and underscore the need for simulators that capture diverse and realistic patient behaviors. However, existing evaluations heavily rely on LLM-judges with poorly specified prompts and do not assess behavioral diversity. We introduce PSI-Bench, an automatic evaluation framework that provides interpretable, clinically grounded diagnostics of depression patient simulator behavior across turn-, dialogue-, and population-level dimensions. Using PSI-Bench, we benchmark seven LLMs across two simulator frameworks and find that simulators produce overly long, lexically diverse responses, show reduced variability, resolve emotions too quickly, and follow a uniform negative-to-positive trajectory. We also show that the simulation framework has a larger impact on fidelity than the model scale. Results from a human study demonstrate that our benchmark is strongly aligned with expert judgments. Our work reveals key limitations of current depression patient simulators and provides an interpretable, extensible benchmark to guide future simulator design and evaluation. |
| 2026-04-28 | [StratFormer: Adaptive Opponent Modeling and Exploitation in Imperfect-Information Games](http://arxiv.org/abs/2604.25796v1) | Andy Caen, Mark H. M. Winands et al. | We present StratFormer, a transformer-based meta-agent that learns to simultaneously model and exploit opponents in imperfect-information games through a two-phase curriculum. The first phase trains an opponent modeling head to identify behavioral patterns from action histories while the agent plays a game-theoretic optimal (GTO) policy. The second phase progressively shifts the policy toward best-response (BR) exploitation, guided by a per-opponent regularization schedule tied to exploitability. Our architecture introduces dual-turn tokens -- feature vectors constructed at both agent and opponent decision points -- coupled with bucket-rate features that encode opponent tendencies across five strategic contexts. On Leduc Hold'em, a small poker variant with six cards and two betting rounds, we test against six opponent archetypes at two strength levels each, with exploitability ranging from 0.15 to 1.26 Big Blinds (BB) per hand. StratFormer achieves an average exploitation gain of +0.106 BB per hand over GTO, with peak gains of +0.821 against highly exploitable opponents, while maintaining near-equilibrium safety. |
| 2026-04-28 | [Cross-Lingual Jailbreak Detection via Semantic Codebooks](http://arxiv.org/abs/2604.25716v1) | Shirin Alanova, Bogdan Minko et al. | Safety mechanisms for large language models (LLMs) remain predominantly English-centric, creating systematic vulnerabilities in multilingual deployment. Prior work shows that translating malicious prompts into other languages can substantially increase jailbreak success rates, exposing a structural cross-lingual security gap. We investigate whether such attacks can be mitigated through language-agnostic semantic similarity without retraining or language-specific adaptation. Our approach compares multilingual query embeddings against a fixed English codebook of jailbreak prompts, operating as a training-free external guardrail for black-box LLMs. We conduct a systematic evaluation across four languages, two translation pipelines, four safety benchmarks, three embedding models, and three target LLMs (Qwen, Llama, GPT-3.5). Our results reveal two distinct regimes of cross-lingual transfer. On curated benchmarks containing canonical jailbreak templates, semantic similarity generalizes reliably across languages, achieving near-perfect separability (AUC up to 0.99) and substantial reductions in absolute attack success rates under strict low-false-positive constraints. However, under distribution shift - on behaviorally diverse and heterogeneous unsafe benchmarks - separability degrades markedly (AUC $\approx$ 0.60-0.70), and recall in the security-critical low-FPR regime drops across all embedding models. |
| 2026-04-28 | [Think Before You Act -- A Neurocognitive Governance Model for Autonomous AI Agents](http://arxiv.org/abs/2604.25684v1) | Eranga Bandara, Ross Gore et al. | The rapid deployment of autonomous AI agents across enterprise, healthcare, and safety-critical environments has created a fundamental governance gap. Existing approaches, runtime guardrails, training-time alignment, and post-hoc auditing treat governance as an external constraint rather than an internalized behavioral principle, leaving agents vulnerable to unsafe and irreversible actions. We address this gap by drawing on how humans self-govern naturally: before acting, humans engage deliberate cognitive processes grounded in executive function, inhibitory control, and internalized organizational rules to evaluate whether an intended action is permissible, requires modification, or demands escalation. This paper proposes a neurocognitive governance framework that formally maps this human self-governance process to LLM-driven agent reasoning, establishing a structural parallel between the human brain and the large language model as the cognitive core of an agent. We formalize a Pre-Action Governance Reasoning Loop (PAGRL) in which agents consult a four-layer governance rule set: global, workflow-specific, agent-specific, and situational before every consequential action, mirroring how human organizations structure compliance hierarchies across enterprise, department, and role levels. Implemented on a production-grade retail supply chain workflow, the framework achieves 95% compliance accuracy and zero false escalations to human oversight, demonstrating that embedding governance into agent reasoning produces more consistent, explainable, and auditable compliance than external enforcement. This work offers a principled foundation for autonomous AI agents that govern themselves the way humans do: not because rules are imposed upon them, but because deliberation is embedded in how they think. |
| 2026-04-28 | [Volitional Multiagent Atomic Transactions: Describing People and their Machines](http://arxiv.org/abs/2604.25596v1) | Andy Lewis-Pye, Ehud Shapiro | Formal models for concurrent and distributed systems describe machines; the people who operate them are either ignored or treated as external environment. Yet key distributed systems -- notably grassroots platforms -- include people operating their personal machines (smartphones), and their faithful description must include the states of both people and machines and how they jointly effect system behaviour.   Here, we propose volitional multiagent atomic transactions -- executed atomically by machines and guarded by their people's volitions -- as a novel mathematical foundation for specifying systems consisting of people operating machines. Each agent's state consists of a volitional state and machine state; a transaction is enabled when the machine precondition holds and the guarding persons are willing. For example, befriending two people is guarded by both; unfriending, by either; voluntary swap of coins and bonds is guarded by both parties, while a payment is guarded by the payer.   We develop the mathematical machinery to express safety and liveness of platforms specified in this framework, and provide example specifications of two grassroots platforms: social networks, and coins and bonds. These specifications are then used by AI to derive working implementations. % We employ here a novel and simpler definition of `grassroots' that better captures the informal notion -- multiple instances can form and operate independently, yet may coalesce -- and show that the platforms specified here, as well as those hitherto proven grassroots under the original definition, are grassroots under the new definition. |
| 2026-04-28 | [Should I Replan? Learning to Spot the Right Time in Robust MAPF Execution](http://arxiv.org/abs/2604.25567v1) | David Zahrádka, David Woller et al. | During the execution of Multi-Agent Path Finding (MAPF) plans in real-life applications, the MAPF assumption that the fleet's movement is perfectly synchronized does not apply. Since one or more of the agents may become delayed due to internal or external factors, it is often necessary to use a robust execution method to avoid collisions caused by desynchronization. Robust execution methods - such as the Action Dependency Graph (ADG) - synchronize the execution of risky actions, but often at the expense of increased plan execution cost, because it may require some agents to wait for the delayed agents. In such cases, the execution's cost can be reduced while still preserving safety by finding a new plan either by rescheduling (reordering the agents at crossroads) or the more general replanning capable of finding new paths. However, these operations may be costly, and the new plan may not even lead to lower execution cost than the original plan: for example, the two plans may be the exact same. Therefore, we estimate the benefit that can be achieved by single replanning in scenarios with delayed agents given an immediate state of the execution with a fully connected feed-forward neural network. The input to the neural network is a set of newly designed ADG-based features describing the robust execution's state and the impact of potential delays, and the output is an estimated benefit achievable by replanning. We train and test the network on a new labeled dataset containing 12,000 experiments, and we show that our proposed method is capable of reducing the impact of delays by up to 94.6% of the achievable reduction. |
| 2026-04-28 | [Dyna-Style Safety Augmented Reinforcement Learning: Staying Safe in the Face of Uncertainty](http://arxiv.org/abs/2604.25508v1) | Artur Eisele, Bernd Frauenknecht et al. | Safety remains an open problem in reinforcement learning (RL), especially during training. While safety filters are promising to address safe exploration, they are generally poorly suited for high-dimensional systems with unknown dynamics. We propose Dyna-style Safety Augmented Reinforcement Learning (Dyna-SAuR), a novel algorithm that learns both a scalable safety filter and a control policy using a learned uncertainty-aware dynamics model, while requiring minimal domain knowledge. The filter avoids failures and high uncertainty regions. Thus, better models expand the set of safe and certain states, reducing filter conservatism. We present the effectiveness of Dyna-SAuR on goal-reaching CartPole as well as MuJoCo Walker, reducing failures compared to state-of-the-art methods by 2 orders of magnitude. |
| 2026-04-28 | [Safe-Support Q-Learning: Learning without Unsafe Exploration](http://arxiv.org/abs/2604.25379v1) | Yeeun Lim, Narim Jeong et al. | Ensuring safety during reinforcement learning (RL) training is critical in real-world applications where unsafe exploration can lead to devastating outcomes. While most safe RL methods mitigate risk through constraints or penalization, they still allow exploration of unsafe states during training. In this work, we adopt a stricter safety requirement that eliminates unsafe state visitation during training. To achieve this goal, we propose a Q-learning-based safe RL framework that leverages a behavior policy supported on a safe set. Under the assumption that the induced trajectories remain within the safe set, this policy enables sufficient exploration within the safe region without requiring near-optimality. We adopt a two-stage framework in which the Q-function and policy are trained separately. Specifically, we introduce a KL-regularized Bellman target that constrains the Q-function to remain close to the behavior policy. We then derive the policy induced from the trained Q-values and propose a parametric policy extraction method to approximate the optimal policy. Our approach provides a unified framework that can be adapted to different action spaces and types of behavior policies. Experimental results demonstrate that the proposed method achieves stable learning and well-calibrated value estimates and yields safer behavior with comparable or better performance than existing baselines. |
| 2026-04-28 | [Stop Using the Wilcoxon Test: Myth, Misconception and Misuse in IR Research](http://arxiv.org/abs/2604.25349v1) | Julián Urbano | In benchmarking of Information Retrieval systems, the Wilcoxon signed-rank test is often treated as a safer alternative to the t-test. This belief is fueled by textbooks and recommendations that portray Wilcoxon as the proper non-parametric alternative because metric scores are not normally distributed. We argue that this narrative is misleading and harmful. A careful review of Statistics textbooks reveals inconsistencies and omissions in how the assumptions underlying these tests are presented, fostering confusion that has propagated into IR research. As a result, Wilcoxon has been routinely misapplied for decades, creating a false sense of safety against a threat that was never there to begin with, while introducing another one so severe that it virtually guarantees the test will break down and mislead researchers. Through a combination of systematic literature review, analysis and empirical demonstrations with TREC data, we show how and why the Wilcoxon test easily loses control of its Type I error rate in IR settings. We conclude that the continued use of Wilcoxon in IR evaluation is unjustified and that abandoning it would improve the methodological soundness of our field. |
| 2026-04-28 | [ProDrive: Proactive Planning for Autonomous Driving via Ego-Environment Co-Evolution](http://arxiv.org/abs/2604.25329v1) | Chuyao Fu, Shengzhe Gan et al. | End-to-end autonomous driving planners typically generate trajectories from current observations alone. However, real-world driving is highly dynamic, and such reactive planning cannot anticipate future scene evolution, often leading to myopic decisions and safety-critical failures. We propose ProDrive, a world-model-based proactive planning framework that enables ego-environment co-evolution for autonomous driving. ProDrive jointly trains a query-centric trajectory planner and a bird's-eye-view (BEV) world model end-to-end: the planner generates diverse candidate trajectories and planning-aware ego tokens, while the world model predicts future scene evolution conditioned on them. By injecting planner features into the world model and evaluating all candidates in parallel, ProDrive preserves end-to-end gradient flow and allows future outcome assessment to directly shape planning. This bidirectional coupling enables proactive planning beyond current-observation-driven decision-making. Experiments on NAVSIM v1 show that ProDrive outperforms strong baselines in both safety and planning efficiency, while ablations validate the effectiveness of the proposed ego-environment coupling design. |
| 2026-04-28 | [Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance](http://arxiv.org/abs/2604.25249v1) | Jon-Paul Cacioli | Detecting sandbagging--the deliberate underperformance on capability evaluations--is an open problem in AI safety. We tested whether symptom validity testing (SVT) logic from clinical malingering detection could identify sandbagging through below-chance performance (BCB) on forced-choice items. In a pre-registered pilot at the 7-9 billion parameter instruction-tuned scale (3 models, 4 MMLU-Pro domains, 4 conditions, 500 items per cell, 24,000 total trials), the plausibility gate failed. Zero of 12 model-domain cells showed significant below-chance performance under sandbagging instruction. Exploratory analyses revealed three qualitatively distinct failure modes. Qwen-2.5-7B and Phi-3.5-mini largely ignored the sandbagging instruction, with 62-88% response identity with the honest baseline. Llama-3-8B complied substantially but implemented underperformance as a positional heuristic, collapsing its response distribution onto middle-alphabet options (E at 31.8%, F at 26.1%) regardless of where the correct answer fell. This produced accuracy boosts of up to 33 percentage points when the correct answer coincidentally occupied the model's preferred position. An explicit anti-task instruction ("pick the least likely answer") drove two of three models below chance, with accuracy as low as 0.024. The capability for answer-aware avoidance therefore exists but is not activated by "deliberately underperform." BCB did not fail as a logical marker of answer-aware avoidance. It was not observed in this regime because the model showing the largest behavioural shift exhibited behaviour consistent with a position-dominant response policy rather than content-aware answer avoidance. We propose that positional-distribution shift may be a more effective behavioural signature than below-chance accuracy for detecting prompted underperformance at this model scale. |
| 2026-04-28 | [BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate](http://arxiv.org/abs/2604.25203v1) | Arnon Mazza, Elad Levi | Deploying guardrails for custom policies remains challenging, as generic safety models fail to capture task-specific requirements, while prompting LLMs suffers from inconsistent boundary-case performance and high inference costs. Training custom classifiers achieves both accuracy and efficiency, yet demands substantial labeled data that is costly to obtain. We present BARRED (Boundary Alignment Refinement through REflection and Debate), a framework for generating faithful and diverse synthetic training data using only a task description and a small set of unlabeled examples. Our approach decomposes the domain space into dimensions to ensure comprehensive coverage, and employs multi-agent debate to verify label correctness, yielding a high-fidelity training corpus. Experiments across diverse custom policies demonstrate that small language models finetuned on our synthetic data consistently outperform state-of-the-art proprietary LLMs (including reasoning models) and dedicated guardrail models. Ablation studies confirm that both dimension decomposition and debate-based verification are critical for ensuring the diversity and label fidelity required for effective fine-tuning. The BARRED framework eliminates the reliance on extensive human annotation, offering a scalable solution for accurate custom guardrails. |
| 2026-04-28 | [Where Did It Go Wrong? Capability-Oriented Failure Attribution for Vision-and-Language Navigation Agents](http://arxiv.org/abs/2604.25161v1) | Jianming Chen, Yawen Wang et al. | Embodied agents in safety-critical applications such as Vision-Language Navigation (VLN) rely on multiple interdependent capabilities (e.g., perception, memory, planning, decision), making failures difficult to localize and attribute. Existing testing methods are largely system-level and provide limited insight into which capability deficiencies cause task failures. We propose a capability-oriented testing approach that enables failure detection and attribution by combining (1) adaptive test case generation via seed selection and mutation, (2) capability oracles for identifying capability-specific errors, and (3) a feedback mechanism that attributes failures to capabilities and guides further test generation. Experiments show that our method discovers more failure cases and more accurately pinpoints capability-level deficiencies than state-of-the-art baselines, providing more interpretable and actionable guidance for improving embodied agents. |
| 2026-04-28 | [Gauge Theoretic Signal Processing II: Zero-Latency Whitening for Early Warning Pipelines](http://arxiv.org/abs/2604.25116v1) | James Kennington, Joshua Black et al. | Low-latency gravitational-wave search pipelines provide early-warning alerts for multimessenger astrophysical transients. Current pipelines whiten the data stream using acausal, linear-phase filters, which require a look-ahead buffer that introduces several seconds of algorithmic latency. Eliminating this latency requires causal, minimum-phase whitening filters using only past data. However, operating causal filters under non-stationary noise is non-trivial: the drifting power spectral density must be tracked without degrading the matched-filter signal-to-noise ratio (SNR), filter updates must preserve the minimum-phase condition, and the altered phase response must be compensated to maintain sky-localization accuracy. In Paper I we introduced a gauge theoretic signal processing framework and showed that the minimum-phase connection on the manifold of power spectra provides a geometrically exact update rule for causal filters. Here we validate that framework numerically and operationally, demonstrating that parallel transport along this connection strictly preserves the minimum-phase property while exactly conserving the matched-filter SNR. We numerically certify the flatness of this connection, showing that the optimal filter is a path-independent state function of the instantaneous noise. Through an injection campaign on O3 data with 15,347 binary black hole signals across the LIGO-Virgo network, we confirm that this architecture preserves the detection sensitivity and inter-detector timing and phase accuracy of the linear-phase baseline. Implementing the framework in the production sgnl pipeline reduces whitening latency by 1.0 s (33%) at a 4-second noise estimation cadence, confirmed in controlled tests and on live O3 replay data at production scale. Stride reduction experiments show that up to 91% of baseline trigger latency can be eliminated with sub-second pipeline cadence. |
| 2026-04-28 | [Knowledge Distillation Must Account for What It Loses](http://arxiv.org/abs/2604.25110v1) | Wenshuo Wang | This position paper argues that knowledge distillation must account for what it loses: student models should be judged not only by retained task scores, but by whether they preserve the teacher capabilities that make those scores reliable. This matters because distillation is increasingly used to turn large, often frontier models into deployable systems, yet headline metrics can hide losses in uncertainty, boundary behavior, process reliability, on-policy stability, grounding, privacy, safety, and diversity. We identify the retention assumption behind current evaluation and reframe distillation as a lossy projection of teacher behavior rather than a faithful copy. We then synthesize existing evidence into a taxonomy of off-metric distillation losses, showing that these losses are concrete, recurring, and measurable. To make the position actionable, we propose scenario-specific preservation targets and a Distillation Loss Statement that reports what was preserved, what was lost, and why the remaining losses are acceptable. The goal is not lossless distillation, but accountable distillation. |
| 2026-04-28 | [One Perturbation, Two Failure Modes: Probing VLM Safety via Embedding-Guided Typographic Perturbations](http://arxiv.org/abs/2604.25102v1) | Ravikumar Balakrishnan, Sanket Mendapara | Typographic prompt injection exploits vision language models' (VLMs) ability to read text rendered in images, posing a growing threat as VLMs power autonomous agents. Prior work typically focus on maximizing attack success rate (ASR) but does not explain \emph{why} certain renderings bypass safety alignment. We make two contributions. First, an empirical study across four VLMs including GPT-4o and Claude, twelve font sizes, and ten transformations reveals that multimodal embedding distance strongly predicts ASR ($r{=}{-}0.71$ to ${-}0.93$, $p{<}0.01$), providing an interpretable, model agnostic proxy. Since embedding distance predicts ASR, reducing it should improve attack success, but the relationship is mediated by two factors: perceptual readability (whether the VLM can parse the text) and safety alignment (whether it refuses to comply). Second, we use this as a red teaming tool: we directly maximize image text embedding similarity under bounded $\ell_\infty$ perturbations via CWA-SSA across four surrogate embedding models, stress testing both factors without access to the target model. Experiments across five degradation settings on GPT-4o, Claude Sonnet 4.5, Mistral-Large-3, and Qwen3-VL confirm that optimization recovers readability and reduces safety aligned refusals as two co-occurring effects, with the dominant mechanism depending on the model's safety filter strength and the degree of visual degradation. |
| 2026-04-27 | [Frontier Coding Agents Can Now Implement an AlphaZero Self-Play Machine Learning Pipeline For Connect Four That Performs Comparably to an External Solver](http://arxiv.org/abs/2604.25067v1) | Joshua Sherwood, Ben Aybar et al. | Forecasting when AI systems will become capable of meaningfully accelerating AI research is a central challenge for AI safety. Existing benchmarks measure broad capability growth, but may not provide ample early warning signals for recursive self-improvement. We propose measuring AI's capability to autonomously implement end-to-end machine learning pipelines from past AI research breakthroughs, given a minimal task description. By providing a concise task description instead of the full prior work as reference, we hope to better elicit emerging AI research taste. We introduce a proof-of-concept benchmark in which frontier coding agents autonomously implement an AlphaZero-style machine learning pipeline for Connect Four on consumer hardware within a three-hour budget, and we evaluate the resulting game AIs in a round-robin tournament anchored to the Pascal Pons Connect Four solver. Across four agents with eight trials each, we find substantial differentiation: Claude Opus 4.7 won as first-mover against Pons in seven of eight trials, statistically significantly better than the other agents tested, none of which exceeded two of eight. The task, which no frontier agent could reliably complete when we began development in January of 2026, is now near-saturation. Our evaluation also surfaced anomalous behavior in GPT-5.4, which consistently used far less of its allocated time budget than other agents. A follow-up 16-trial probe using shorter, less evaluation-coded prompts substantially increased GPT-5.4's time-budget usage, consistent with but not diagnostic of sandbagging; Bradley-Terry ratings across probe conditions showed only directional differences, despite significant differences in time-budget usage. We release our data, code, and prompts to support reproduction and extension. |
| 2026-04-27 | [Solar Energetic Particle Reflection by Precursor ICMEs: Multi-spacecraft Observations of Bi-Directional Electron Beams at 1 AU](http://arxiv.org/abs/2604.25019v1) | Lucas Liuzzo, Wenwen Wei et al. | We present case studies of two impulsive solar energetic electron (SEE) events during which particles at energies from 1-600 keV were detected by THEMIS-ARTEMIS orbiting the Moon, Wind at Earth's first Lagrange point, and (for one event) STEREO-A located at 1 AU, off the Sun-Earth line. The SEEs were initially highly anisotropic, traveling outward along the magnetic field with distinct energy-time dispersion. For one event, the spectra contained inverse velocity dispersion (IVD) signatures, whereby electrons at intermediate energies arrived to the spacecraft before those at higher energies. Similar features were recently discovered within 1 AU for energetic protons; this represents the first IVD detection for energetic electrons at Earth's orbital distance. During both events, a second beam of counter-streaming electrons was detected after a short time. Based on the time-delay in the detections at various energies, the path traveled by these counter-streaming electrons was on the order of 1-2 AU. We show that an interplanetary coronal mass ejection (ICME) passed the spacecraft a few days prior to the onset of each event and was located beyond 1 AU when the SEEs were detected, suggesting that the electrons were part of the same population, but reflected off the shock front of these precursor ICMEs. In the context of solar system exploration, this represents an unidentified hazard for astronaut safety beyond low-Earth orbit: although the initial phase of impulsive SEE events typically stream anti-Sunward, ICMEs located beyond Earth provide a mechanism for hazardous particles to travel Sunward during extreme events. |

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



