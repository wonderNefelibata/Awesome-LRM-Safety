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
| 2025-10-27 | [Think Twice: Branch-and-Rethink Reasoning Reward Model](http://arxiv.org/abs/2510.23596v1) | Yizhu Jiao, Jiaqi Zeng et al. | Large language models (LLMs) increasingly rely on thinking models that externalize intermediate steps and allocate extra test-time compute, with think-twice strategies showing that a deliberate second pass can elicit stronger reasoning. In contrast, most reward models (RMs) still compress many quality dimensions into a single scalar in one shot, a design that induces judgment diffusion: attention spreads across evaluation criteria, yielding diluted focus and shallow analysis. We introduce branch-and-rethink (BR-RM), a two-turn RM that transfers the think-twice principle to reward modeling. Turn 1 performs adaptive branching, selecting a small set of instance-critical dimensions (such as factuality and safety) and sketching concise, evidence-seeking hypotheses. Turn 2 executes branch-conditioned rethinking, a targeted reread that tests those hypotheses and scrutinizes only what matters most. We train with GRPO-style reinforcement learning over structured two-turn traces using a simple binary outcome reward with strict format checks, making the approach compatible with standard RLHF pipelines. By converting all-at-oncescoringintofocused, second-lookreasoning, BR-RMreducesjudgmentdiffusionandimproves sensitivity to subtle yet consequential errors while remaining practical and scalable. Experimental results demonstrate that our model achieves state-of-the-art performance on three challenging reward modeling benchmarks across diverse domains. The code and the model will be released soon. |
| 2025-10-27 | [UrbanVLA: A Vision-Language-Action Model for Urban Micromobility](http://arxiv.org/abs/2510.23576v1) | Anqi Li, Zhiyong Wang et al. | Urban micromobility applications, such as delivery robots, demand reliable navigation across large-scale urban environments while following long-horizon route instructions. This task is particularly challenging due to the dynamic and unstructured nature of real-world city areas, yet most existing navigation methods remain tailored to short-scale and controllable scenarios. Effective urban micromobility requires two complementary levels of navigation skills: low-level capabilities such as point-goal reaching and obstacle avoidance, and high-level capabilities, such as route-visual alignment. To this end, we propose UrbanVLA, a route-conditioned Vision-Language-Action (VLA) framework designed for scalable urban navigation. Our method explicitly aligns noisy route waypoints with visual observations during execution, and subsequently plans trajectories to drive the robot. To enable UrbanVLA to master both levels of navigation, we employ a two-stage training pipeline. The process begins with Supervised Fine-Tuning (SFT) using simulated environments and trajectories parsed from web videos. This is followed by Reinforcement Fine-Tuning (RFT) on a mixture of simulation and real-world data, which enhances the model's safety and adaptability in real-world settings. Experiments demonstrate that UrbanVLA surpasses strong baselines by more than 55% in the SocialNav task on MetaUrban. Furthermore, UrbanVLA achieves reliable real-world navigation, showcasing both scalability to large-scale urban environments and robustness against real-world uncertainties. |
| 2025-10-27 | [RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation](http://arxiv.org/abs/2510.23571v1) | Yash Jangir, Yidi Zhang et al. | The pursuit of robot generalists - instructable agents capable of performing diverse tasks across diverse environments - demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. Existing simulation benchmarks are similarly limited, as they train and test policies within the same synthetic domains and cannot assess models trained from real-world demonstrations or alternative simulation environments. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. In this paper, we introduce a new benchmarking framework that overcomes these challenges by shifting VLA evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated VLM-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons. To measure robustness, we systematically perturb simulated environments along multiple axes, such as textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world trained robot manipulation policies, addressing a critical missing capability in today's robotics landscape. |
| 2025-10-27 | [Linear effects, exceptions, and resource safety: a Curry-Howard correspondence for destructors](http://arxiv.org/abs/2510.23517v1) | Sidney Congard, Guillaume Munch-Maccagnoni et al. | We analyse the problem of combining linearity, effects, and exceptions, in abstract models of programming languages, as the issue of providing some kind of strength for a monad $T(- \oplus E)$ in a linear setting. We consider in particular for $T$ the allocation monad, which we introduce to model and study resource-safety properties. We apply these results to a series of two linear effectful calculi for which we establish their resource-safety properties.   The first calculus is a linear call-by-push-value language with two allocation effects $\mathit{new}$ and $\mathit{delete}$. The resource-safety properties follow from the linear (and even ordered) character of the typing rules.   We then explain how to integrate exceptions on top of linearity and effects by adjoining default destruction actions to types, as inspired by C++/Rust destructors. We see destructors as objects $\delta : A\rightarrow TI$ in the slice category over $TI$. This construction gives rise to a second calculus, an affine ordered call-by-push-value language with exceptions and destructors, in which the weakening rule performs a side-effect. As in C++/Rust, a ``move'' operation is necessary to allow random-order release of resources, as opposed to last-in-first-out order. Moving resources is modelled as an exchange rule that performs a side-effect. |
| 2025-10-27 | [Localising under the drape: proprioception in the era of distributed surgical robotic system](http://arxiv.org/abs/2510.23512v1) | Martin Huber, Nicola A. Cavalcanti et al. | Despite their mechanical sophistication, surgical robots remain blind to their surroundings. This lack of spatial awareness causes collisions, system recoveries, and workflow disruptions, issues that will intensify with the introduction of distributed robots with independent interacting arms. Existing tracking systems rely on bulky infrared cameras and reflective markers, providing only limited views of the surgical scene and adding hardware burden in crowded operating rooms. We present a marker-free proprioception method that enables precise localisation of surgical robots under their sterile draping despite associated obstruction of visual cues. Our method solely relies on lightweight stereo-RGB cameras and novel transformer-based deep learning models. It builds on the largest multi-centre spatial robotic surgery dataset to date (1.4M self-annotated images from human cadaveric and preclinical in vivo studies). By tracking the entire robot and surgical scene, rather than individual markers, our approach provides a holistic view robust to occlusions, supporting surgical scene understanding and context-aware control. We demonstrate an example of potential clinical benefits during in vivo breathing compensation with access to tissue dynamics, unobservable under state of the art tracking, and accurately locate in multi-robot systems for future intelligent interaction. In addition, and compared with existing systems, our method eliminates markers and improves tracking visibility by 25%. To our knowledge, this is the first demonstration of marker-free proprioception for fully draped surgical robots, reducing setup complexity, enhancing safety, and paving the way toward modular and autonomous robotic surgery. |
| 2025-10-27 | [An Error-Based Safety Buffer for Safe Adaptive Control (Extended Version)](http://arxiv.org/abs/2510.23491v1) | Peter A. Fisher, Johannes Autenrieb et al. | We consider the problem of adaptive control of a class of feedback linearizable plants with matched parametric uncertainties whose states are accessible, subject to state constraints, which often arise due to safety considerations. In this paper, we combine adaptation and control barrier functions into a real-time control architecture that guarantees stability, ensures control performance, and remains safe even with the parametric uncertainties. Two problems are considered, differing in the nature of the parametric uncertainties. In both cases, the control barrier function is assumed to have an arbitrary relative degree. In addition to guaranteeing stability, it is proved that both the control objective and safety objective are met with near-zero conservatism. No excitation conditions are imposed on the command signal. Simulation results demonstrate the non-conservatism of all of the theoretical developments. |
| 2025-10-27 | [Are Agents Just Automata? On the Formal Equivalence Between Agentic AI and the Chomsky Hierarchy](http://arxiv.org/abs/2510.23487v1) | Roham Koohestani, Ziyou Li et al. | This paper establishes a formal equivalence between the architectural classes of modern agentic AI systems and the abstract machines of the Chomsky hierarchy. We posit that the memory architecture of an AI agent is the definitive feature determining its computational power and that it directly maps it to a corresponding class of automaton. Specifically, we demonstrate that simple reflex agents are equivalent to Finite Automata, hierarchical task-decomposition agents are equivalent to Pushdown Automata, and agents employing readable/writable memory for reflection are equivalent to TMs. This Automata-Agent Framework provides a principled methodology for right-sizing agent architectures to optimize computational efficiency and cost. More critically, it creates a direct pathway to formal verification, enables the application of mature techniques from automata theory to guarantee agent safety and predictability. By classifying agents, we can formally delineate the boundary between verifiable systems and those whose behavior is fundamentally undecidable. We address the inherent probabilistic nature of LLM-based agents by extending the framework to probabilistic automata that allow quantitative risk analysis. The paper concludes by outlining an agenda for developing static analysis tools and grammars for agentic frameworks. |
| 2025-10-27 | [Evaluating Large Language Models for Stance Detection on Financial Targets from SEC Filing Reports and Earnings Call Transcripts](http://arxiv.org/abs/2510.23464v1) | Nikesh Gyawali, Doina Caragea et al. | Financial narratives from U.S. Securities and Exchange Commission (SEC) filing reports and quarterly earnings call transcripts (ECTs) are very important for investors, auditors, and regulators. However, their length, financial jargon, and nuanced language make fine-grained analysis difficult. Prior sentiment analysis in the financial domain required a large, expensive labeled dataset, making the sentence-level stance towards specific financial targets challenging. In this work, we introduce a sentence-level corpus for stance detection focused on three core financial metrics: debt, earnings per share (EPS), and sales. The sentences were extracted from Form 10-K annual reports and ECTs, and labeled for stance (positive, negative, neutral) using the advanced ChatGPT-o3-pro model under rigorous human validation. Using this corpus, we conduct a systematic evaluation of modern large language models (LLMs) using zero-shot, few-shot, and Chain-of-Thought (CoT) prompting strategies. Our results show that few-shot with CoT prompting performs best compared to supervised baselines, and LLMs' performance varies across the SEC and ECT datasets. Our findings highlight the practical viability of leveraging LLMs for target-specific stance in the financial domain without requiring extensive labeled data. |
| 2025-10-27 | [Floating-Point Neural Network Verification at the Software Level](http://arxiv.org/abs/2510.23389v1) | Edoardo Manino, Bruno Farias et al. | The behaviour of neural network components must be proven correct before deployment in safety-critical systems. Unfortunately, existing neural network verification techniques cannot certify the absence of faults at the software level. In this paper, we show how to specify and verify that neural networks are safe, by explicitly reasoning about their floating-point implementation. In doing so, we construct NeuroCodeBench 2.0, a benchmark comprising 912 neural network verification examples that cover activation functions, common layers, and full neural networks of up to 170K parameters. Our verification suite is written in plain C and is compatible with the format of the International Competition on Software Verification (SV-COMP). Thanks to it, we can conduct the first rigorous evaluation of eight state-of-the-art software verifiers on neural network code. The results show that existing automated verification tools can correctly solve an average of 11% of our benchmark, while producing around 3% incorrect verdicts. At the same time, a historical analysis reveals that the release of our benchmark has already had a significantly positive impact on the latter. |
| 2025-10-27 | [Full-Dynamics Real-Time Nonlinear Model Predictive Control of Heavy-Duty Hydraulic Manipulator for Trajectory Tracking Tasks](http://arxiv.org/abs/2510.23386v1) | Alvaro Paz, Mahdi Hejrati et al. | Heavy-duty hydraulic manipulators (HHMs) operate under strict physical and safety-critical constraints due to their large size, high power, and complex nonlinear dynamics. Ensuring that both joint-level and end-effector trajectories remain compliant with actuator capabilities, such as force, velocity, and position limits, is essential for safe and reliable operation, yet remains largely underexplored in real-time control frameworks. This paper presents a nonlinear model predictive control (NMPC) framework designed to guarantee constraint satisfaction throughout the full nonlinear dynamics of HHMs, while running at a real-time control frequency of 1 kHz. The proposed method combines a multiple-shooting strategy with real-time sensor feedback, and is supported by a robust low-level controller based on virtual decomposition control (VDC) for precise joint tracking. Experimental validation on a full-scale hydraulic manipulator shows that the NMPC framework not only enforces actuator constraints at the joint level, but also ensures constraint-compliant motion in Cartesian space for the end-effector. These results demonstrate the method's capability to deliver high-accuracy trajectory tracking while strictly respecting safety-critical limits, setting a new benchmark for real-time control in large-scale hydraulic systems. |
| 2025-10-27 | [Process Reward Models for Sentence-Level Verification of LVLM Radiology Reports](http://arxiv.org/abs/2510.23217v1) | Alois Thomas, Maya Varma et al. | Automating radiology report generation with Large Vision-Language Models (LVLMs) holds great potential, yet these models often produce clinically critical hallucinations, posing serious risks. Existing hallucination detection methods frequently lack the necessary sentence-level granularity or robust generalization across different LVLM generators. We introduce a novel approach: a sentence-level Process Reward Model (PRM) adapted for this vision-language task. Our PRM predicts the factual correctness of each generated sentence, conditioned on clinical context and preceding text. When fine-tuned on MIMIC-CXR with weakly-supervised labels, a lightweight 0.5B-parameter PRM outperforms existing verification techniques, demonstrating, for instance, relative improvements of 7.5% in Matthews Correlation Coefficient and 1.8% in AUROC over strong white-box baselines on outputs from one LVLM. Unlike methods reliant on internal model states, our PRM demonstrates strong generalization to an unseen LVLM. We further show its practical utility: PRM scores effectively filter low-quality reports, improving F1-CheXbert scores by 4.5% (when discarding the worst 10% of reports). Moreover, when guiding a novel weighted best-of-N selection process on the MIMIC-CXR test set, our PRM show relative improvements in clinical metrics of 7.4% for F1-CheXbert and 0.6% for BERTScore. These results demonstrate that a lightweight, context-aware PRM provides a model-agnostic safety layer for clinical LVLMs without access to internal activations |
| 2025-10-27 | [Neural Networks for AC Optimal Power Flow: Improving Worst-Case Guarantees during Training](http://arxiv.org/abs/2510.23196v1) | Bastien Giraud, Rahul Nellikath et al. | The AC Optimal Power Flow (AC-OPF) problem is central to power system operation but challenging to solve efficiently due to its nonconvex and nonlinear nature. Neural networks (NNs) offer fast surrogates, yet their black-box behavior raises concerns about constraint violations that can compromise safety. We propose a verification-informed NN framework that incorporates worst-case constraint violations directly into training, producing models that are both accurate and provably safer. Through post-hoc verification, we achieve substantial reductions in worst-case violations and, for the first time, verify all operational constraints of large-scale AC-OPF proxies. Practical feasibility is further enhanced via restoration and warm-start strategies for infeasible operating points. Experiments on systems ranging from 57 to 793 buses demonstrate scalability, speed, and reliability, bridging the gap between ML acceleration and safe, real-time deployment of AC-OPF solutions - and paving the way toward data-driven optimal control. |
| 2025-10-27 | [Evaluation of Vision-LLMs in Surveillance Video](http://arxiv.org/abs/2510.23190v1) | Pascal Benschop, Cristian Meo et al. | The widespread use of cameras in our society has created an overwhelming amount of video data, far exceeding the capacity for human monitoring. This presents a critical challenge for public safety and security, as the timely detection of anomalous or criminal events is crucial for effective response and prevention. The ability for an embodied agent to recognize unexpected events is fundamentally tied to its capacity for spatial reasoning. This paper investigates the spatial reasoning of vision-language models (VLMs) by framing anomalous action recognition as a zero-shot, language-grounded task, addressing the embodied perception challenge of interpreting dynamic 3D scenes from sparse 2D video. Specifically, we investigate whether small, pre-trained vision--LLMs can act as spatially-grounded, zero-shot anomaly detectors by converting video into text descriptions and scoring labels via textual entailment. We evaluate four open models on UCF-Crime and RWF-2000 under prompting and privacy-preserving conditions. Few-shot exemplars can improve accuracy for some models, but may increase false positives, and privacy filters -- especially full-body GAN transforms -- introduce inconsistencies that degrade accuracy. These results chart where current vision--LLMs succeed (simple, spatially salient events) and where they falter (noisy spatial cues, identity obfuscation). Looking forward, we outline concrete paths to strengthen spatial grounding without task-specific training: structure-aware prompts, lightweight spatial memory across clips, scene-graph or 3D-pose priors during description, and privacy methods that preserve action-relevant geometry. This positions zero-shot, language-grounded pipelines as adaptable building blocks for embodied, real-world video understanding. Our implementation for evaluating VLMs is publicly available at: https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition |
| 2025-10-27 | [Optimizing Optimism: Up to 6.5x Faster zkVM Validty Proofs via Sparse Derivation](http://arxiv.org/abs/2510.23172v1) | Mohsen Ahmadvand, Pedro Souto | The Optimism derivation pipeline is engineered for correctness and liveness, not for succinct validity proofs. A straightforward port to a zkVM imposes significant overheads, making validity proofs significantly more costly than necessary. We systematically identify inefficiencies in the current design, analyze their impact on proving costs, and provide a soundness-preserving redesign tailored to zk proving. Our redesign achieves up to 6.5x faster derivation inside zkVMs (3.5x overall speedup) while maintaining identical safety guarantees. |
| 2025-10-27 | [Planning Oriented Integrated Sensing and Communication](http://arxiv.org/abs/2510.23021v1) | Xibin Jin, Guoliang Li et al. | Integrated sensing and communication (ISAC) enables simultaneous localization, environment perception, and data exchange for connected autonomous vehicles. However, most existing ISAC designs prioritize sensing accuracy and communication throughput, treating all targets uniformly and overlooking the impact of critical obstacles on motion efficiency. To overcome this limitation, we propose a planning-oriented ISAC (PISAC) framework that reduces the sensing uncertainty of planning-bottleneck obstacles and expands the safe navigable path for the ego-vehicle, thereby bridging the gap between physical-layer optimization and motion-level planning. The core of PISAC lies in deriving a closed-form safety bound that explicitly links ISAC transmit power to sensing uncertainty, based on the Cram\'er-Rao Bound and occupancy inflation principles. Using this model, we formulate a bilevel power allocation and motion planning (PAMP) problem, where the inner layer optimizes the ISAC beam power distribution and the outer layer computes a collision-free trajectory under uncertainty-aware safety constraints. Comprehensive simulations in high-fidelity urban driving environments demonstrate that PISAC achieves up to 40% higher success rates and over 5% shorter traversal times than existing ISAC-based and communication-oriented benchmarks, validating its effectiveness in enhancing both safety and efficiency. |
| 2025-10-27 | [The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](http://arxiv.org/abs/2510.22977v1) | Chenlong Yin, Zeyang Sha et al. | Enhancing the reasoning capabilities of Large Language Models (LLMs) is a key strategy for building Agents that "think then act." However, recent observations, like OpenAI's o3, suggest a paradox: stronger reasoning often coincides with increased hallucination, yet no prior work has systematically examined whether reasoning enhancement itself causes tool hallucination. To address this gap, we pose the central question: Does strengthening reasoning increase tool hallucination? To answer this, we introduce SimpleToolHalluBench, a diagnostic benchmark measuring tool hallucination in two failure modes: (i) no tool available, and (ii) only distractor tools available. Through controlled experiments, we establish three key findings. First, we demonstrate a causal relationship: progressively enhancing reasoning through RL increases tool hallucination proportionally with task performance gains. Second, this effect transcends overfitting - training on non-tool tasks (e.g., mathematics) still amplifies subsequent tool hallucination. Third, the effect is method-agnostic, appearing when reasoning is instilled via supervised fine-tuning and when it is merely elicited at inference by switching from direct answers to step-by-step thinking. We also evaluate mitigation strategies including Prompt Engineering and Direct Preference Optimization (DPO), revealing a fundamental reliability-capability trade-off: reducing hallucination consistently degrades utility. Mechanistically, Reasoning RL disproportionately collapses tool-reliability-related representations, and hallucinations surface as amplified divergences concentrated in late-layer residual streams. These findings reveal that current reasoning enhancement methods inherently amplify tool hallucination, highlighting the need for new training objectives that jointly optimize for capability and reliability. |
| 2025-10-27 | [CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents](http://arxiv.org/abs/2510.22963v1) | Zesen Liu, Zhixiang Zhang et al. | LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections. |
| 2025-10-27 | [Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](http://arxiv.org/abs/2510.22954v1) | Liwei Jiang, Yuanjun Chai et al. | Language models (LMs) often struggle to generate diverse, human-like creative content, raising concerns about the long-term homogenization of human thought through repeated exposure to similar outputs. Yet scalable methods for evaluating LM output diversity remain limited, especially beyond narrow tasks such as random number or name generation, or beyond repeated sampling from a single model. We introduce Infinity-Chat, a large-scale dataset of 26K diverse, real-world, open-ended user queries that admit a wide range of plausible answers with no single ground truth. We introduce the first comprehensive taxonomy for characterizing the full spectrum of open-ended prompts posed to LMs, comprising 6 top-level categories (e.g., brainstorm & ideation) that further breaks down to 17 subcategories. Using Infinity-Chat, we present a large-scale study of mode collapse in LMs, revealing a pronounced Artificial Hivemind effect in open-ended generation of LMs, characterized by (1) intra-model repetition, where a single model consistently generates similar responses, and more so (2) inter-model homogeneity, where different models produce strikingly similar outputs. Infinity-Chat also includes 31,250 human annotations, across absolute ratings and pairwise preferences, with 25 independent human annotations per example. This enables studying collective and individual-specific human preferences in response to open-ended queries. Our findings show that LMs, reward models, and LM judges are less well calibrated to human ratings on model generations that elicit differing idiosyncratic annotator preferences, despite maintaining comparable overall quality. Overall, INFINITY-CHAT presents the first large-scale resource for systematically studying real-world open-ended queries to LMs, revealing critical insights to guide future research for mitigating long-term AI safety risks posed by the Artificial Hivemind. |
| 2025-10-27 | [Intelligent Multimodal Multi-Sensor Fusion-Based UAV Identification, Localization, and Countermeasures for Safeguarding Low-Altitude Economy](http://arxiv.org/abs/2510.22947v1) | Yi Tao, Zhen Gao et al. | The development of the low-altitude economy has led to a growing prominence of uncrewed aerial vehicle (UAV) safety management issues. Therefore, accurate identification, real-time localization, and effective countermeasures have become core challenges in airspace security assurance. This paper introduces an integrated UAV management and control system based on deep learning, which integrates multimodal multi-sensor fusion perception, precise positioning, and collaborative countermeasures. By incorporating deep learning methods, the system combines radio frequency (RF) spectral feature analysis, radar detection, electro-optical identification, and other methods at the detection level to achieve the identification and classification of UAVs. At the localization level, the system relies on multi-sensor data fusion and the air-space-ground integrated communication network to conduct real-time tracking and prediction of UAV flight status, providing support for early warning and decision-making. At the countermeasure level, it adopts comprehensive measures that integrate ``soft kill'' and ``hard kill'', including technologies such as electromagnetic signal jamming, navigation spoofing, and physical interception, to form a closed-loop management and control process from early warning to final disposal, which significantly enhances the response efficiency and disposal accuracy of low-altitude UAV management. |
| 2025-10-27 | [Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies](http://arxiv.org/abs/2510.22944v1) | Bin Wang, YiLu Zhong et al. | Large language models (LLMs) have become indispensable for automated code generation, yet the quality and security of their outputs remain a critical concern. Existing studies predominantly concentrate on adversarial attacks or inherent flaws within the models. However, a more prevalent yet underexplored issue concerns how the quality of a benign but poorly formulated prompt affects the security of the generated code. To investigate this, we first propose an evaluation framework for prompt quality encompassing three key dimensions: goal clarity, information completeness, and logical consistency. Based on this framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale benchmark dataset containing tasks with prompts categorized into four distinct levels of normativity (L0-L3). Extensive experiments on multiple state-of-the-art LLMs reveal a clear correlation: as prompt normativity decreases, the likelihood of generating insecure code consistently and markedly increases. Furthermore, we demonstrate that advanced prompting techniques, such as Chain-of-Thought and Self-Correction, effectively mitigate the security risks introduced by low-quality prompts, substantially improving code safety. Our findings highlight that enhancing the quality of user prompts constitutes a critical and effective strategy for strengthening the security of AI-generated code. |

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



