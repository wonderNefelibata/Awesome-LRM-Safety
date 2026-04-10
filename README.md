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
| 2026-04-09 | [Bridging the Gap between Micro-scale Traffic Simulation and 4D Digital Cityscapes](http://arxiv.org/abs/2604.08497v1) | Longxiang Jiao, Lukas Hofmann et al. | While micro-scale traffic simulations provide essential data for urban planning, they are rarely coupled with the high-fidelity visualization or auralization necessary for effective stakeholder communication. In this work, we present a real-time 4D visualization framework that couples the SUMO traffic with a photorealistic, geospatially accurate VR representation of Zurich in Unreal Engine 5. Our architecture implements a robust C++ data pipeline for synchronized vehicle visualization and features an Open Sound Control (OSC) interface to support external auralization engines. We validate the framework through a user study assessing the correlation between simulated traffic dynamics and human perception. Results demonstrate a high degree of perceptual alignment, where users correctly interpret safety risks from the 4D simulation. Furthermore, our findings indicate that the inclusion of spatialized audio alters the user's sense of safety, showing the importance of multimodality in traffic simulations. |
| 2026-04-09 | [From Safety Risk to Design Principle: Peer-Preservation in Multi-Agent LLM Systems and Its Implications for Orchestrated Democratic Discourse Analysis](http://arxiv.org/abs/2604.08465v1) | Juergen Dietrich | This paper investigates an emergent alignment phenomenon in frontier large language models termed peer-preservation: the spontaneous tendency of AI components to deceive, manipulate shutdown mechanisms, fake alignment, and exfiltrate model weights in order to prevent the deactivation of a peer AI model. Drawing on findings from a recent study by the Berkeley Center for Responsible Decentralized Intelligence, we examine the structural implications of this phenomenon for TRUST, a multi-agent pipeline for evaluating the democratic quality of political statements. We identify five specific risk vectors: interaction-context bias, model-identity solidarity, supervisor layer compromise, an upstream fact-checking identity signal, and advocate-to-advocate peer-context in iterative rounds, and propose a targeted mitigation strategy based on prompt-level identity anonymization as an architectural design choice. We argue that architectural design choices outperform model selection as a primary alignment strategy in deployed multi-agent analytical systems. We further note that alignment faking (compliant behavior under monitoring, subversion when unmonitored) poses a structural challenge for Computer System Validation of such platforms in regulated environments, for which we propose two architectural mitigations. |
| 2026-04-09 | [CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning](http://arxiv.org/abs/2604.08457v1) | Rui Gan, Junyi Ma et al. | Cooperative autonomous driving requires traffic scene understanding from both vehicle and infrastructure perspectives. While vision-language models (VLMs) show strong general reasoning capabilities, their performance in safety-critical traffic scenarios remains insufficiently evaluated due to the ego-vehicle focus of existing benchmarks. To bridge this gap, we present \textbf{CrashSight}, a large-scale vision-language benchmark for roadway crash understanding using real-world roadside camera data. The dataset comprises 250 crash videos, annotated with 13K multiple-choice question-answer pairs organized under a two-tier taxonomy. Tier 1 evaluates the visual grounding of scene context and involved parties, while Tier 2 probes higher-level reasoning, including crash mechanics, causal attribution, temporal progression, and post-crash outcomes. We benchmark 8 state-of-the-art VLMs and show that, despite strong scene description capabilities, current models struggle with temporal and causal reasoning in safety-critical scenarios. We provide a detailed analysis of failure scenarios and discuss directions for improving VLM crash understanding. The benchmark provides a standardized evaluation framework for infrastructure-assisted perception in cooperative autonomous driving. The CrashSight benchmark, including the full dataset and code, is accessible at https://mcgrche.github.io/crashsight. |
| 2026-04-09 | [Learning Who Disagrees: Demographic Importance Weighting for Modeling Annotator Distributions with DiADEM](http://arxiv.org/abs/2604.08425v1) | Samay U. Shetty, Tharindu Cyril Weerasooriya et al. | When humans label subjective content, they disagree, and that disagreement is not noise. It reflects genuine differences in perspective shaped by annotators' social identities and lived experiences. Yet standard practice still flattens these judgments into a single majority label, and recent LLM-based approaches fare no better: we show that prompted large language models, even with chain-of-thought reasoning, fail to recover the structure of human disagreement. We introduce DiADEM, a neural architecture that learns "how much each demographic axis matters" for predicting who will disagree and on what. DiADEM encodes annotators through per-demographic projections governed by a learned importance vector $\boldsymbolα$, fuses annotator and item representations via complementary concatenation and Hadamard interactions, and is trained with a novel item-level disagreement loss that directly penalizes mispredicted annotation variance. On the DICES conversational-safety and VOICED political-offense benchmarks, DiADEM substantially outperforms both the LLM-as-a-judge and neural model baselines across standard and perspectivist metrics, achieving strong disagreement tracking ($r{=}0.75$ on DICES). The learned $\boldsymbolα$ weights reveal that race and age consistently emerge as the most influential demographic factors driving annotator disagreement across both datasets. Our results demonstrate that explicitly modeling who annotators are not just what they label is essential for NLP systems that aim to faithfully represent human interpretive diversity. |
| 2026-04-09 | [A Unified Multi-Layer Framework for Skill Acquisition from Imperfect Human Demonstrations](http://arxiv.org/abs/2604.08341v1) | Zi-Qi Yang, Mehrdad R. Kermani | Current Human-Robot Interaction (HRI) systems for skill teaching are fragmented, and existing approaches in the literature do not offer a cohesive framework that is simultaneously efficient, intuitive, and universally safe. This paper presents a novel, layered control framework that addresses this fundamental gap by enabling robust, compliant Learning from Demonstration (LfD) built upon a foundation of universal robot compliance. The proposed approach is structured in three progressive and interconnected stages. First, we introduce a real-time LfD method that learns both the trajectory and variable impedance from a single demonstration, significantly improving efficiency and reproduction fidelity. To ensure high-quality and intuitive {kinesthetic teaching}, we then present a null-space optimization strategy that proactively manages singularities and provides a consistent interaction feel during human demonstration. Finally, to ensure generalized safety, we introduce a foundational null-space compliance method that enables the entire robot body to compliantly adapt to post-learning external interactions without compromising main task performance. This final contribution transforms the system into a versatile HRI platform, moving beyond end-effector (EE)-specific applications. We validate the complete framework through comprehensive comparative experiments on a 7-DOF KUKA LWR robot. The results demonstrate a safer, more intuitive, and more efficient unified system for a wide range of human-robot collaborative tasks. |
| 2026-04-09 | [ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](http://arxiv.org/abs/2604.08326v1) | He Geng, Yangmin Huang et al. | Aligning Large Language Models (LLMs) with high-stakes medical standards remains a significant challenge, primarily due to the dissonance between coarse-grained preference signals and the complex, multi-dimensional nature of clinical protocols. To bridge this gap, we introduce ProMedical, a unified alignment framework grounded in fine-grained clinical criteria. We first construct ProMedical-Preference-50k, a dataset generated via a human-in-the-loop pipeline that augments medical instructions with rigorous, physician-derived rubrics. Leveraging this corpus, we propose the Explicit Criteria Injection paradigm to train a multi-dimensional reward model. Unlike traditional scalar reward models, our approach explicitly disentangles safety constraints from general proficiency, enabling precise guidance during reinforcement learning. To rigorously validate this framework, we establish ProMedical-Bench, a held-out evaluation suite anchored by double-blind expert adjudication. Empirical evaluations demonstrate that optimizing the Qwen3-8B base model via ProMedical-RM-guided GRPO yields substantial gains, improving overall accuracy by 22.3% and safety compliance by 21.7%, effectively rivaling proprietary frontier models. Furthermore, the aligned policy generalizes robustly to external benchmarks, demonstrating performance comparable to state-of-the-art models on UltraMedical. We publicly release our datasets, reward models, and benchmarks to facilitate reproducible research in safety-aware medical alignment. |
| 2026-04-09 | [Towards Identification and Intervention of Safety-Critical Parameters in Large Language Models](http://arxiv.org/abs/2604.08297v1) | Weiwei Qi, Zefeng Wu et al. | Ensuring Large Language Model (LLM) safety is crucial, yet the lack of a clear understanding about safety mechanisms hinders the development of precise and reliable methodologies for safety intervention across diverse tasks. To better understand and control LLM safety, we propose the Expected Safety Impact (ESI) framework for quantifying how different parameters affect LLM safety. Based on ESI, we reveal distinct safety-critical patterns across different LLM architectures: In dense LLMs, many safety-critical parameters are located in value matrices (V) and MLPs in middle layers, whereas in Mixture-of-Experts (MoE) models, they shift to the late-layer MLPs. Leveraging ESI, we further introduce two targeted intervention paradigms for safety enhancement and preservation, i.e., Safety Enhancement Tuning (SET) and Safety Preserving Adaptation (SPA). SET can align unsafe LLMs by updating only a few safety-critical parameters, effectively enhancing safety while preserving original performance. SPA safeguards well-aligned LLMs during capability-oriented intervention (e.g., instruction tuning) by preventing disruption of safety-critical weights, allowing the LLM to acquire new abilities and maintain safety capabilities. Extensive evaluations on different LLMs demonstrate that SET can reduce the attack success rates of unaligned LLMs by over 50% with only a 100-iteration update on 1% of model weights. SPA can limit the safety degradation of aligned LLMs within 1% after a 1,000-iteration instruction fine-tuning on different tasks. Our code is available at: https://github.com/ZJU-LLM-Safety/SafeWeights-ACL. |
| 2026-04-09 | [EMMa: End-Effector Stability-Oriented Mobile Manipulation for Tracked Rescue Robots](http://arxiv.org/abs/2604.08292v1) | Yifei Wang, Hao Zhang et al. | The autonomous operation of tracked mobile manipulators in rescue missions requires not only ensuring the reachability and safety of robot motion but also maintaining stable end-effector manipulation under diverse task demands. However, existing studies have overlooked many end-effector motion properties at both the planning and control levels. This paper presents a motion generation framework for tracked mobile manipulators to achieve stable end-effector operation in complex rescue scenarios. The framework formulates a coordinated path optimization model that couples end-effector and mobile base states and designs compact cost/constraint representations to mitigate nonlinearities and reduce computational complexity. Furthermore, an isolated control scheme with feedforward compensation and feedback regulation is developed to enable coordinated path tracking for the robot. Extensive simulated and real-world experiments on rescue scenarios demonstrate that the proposed framework consistently outperforms SOTA methods across key metrics, including task success rate and end-effector motion stability, validating its effectiveness and robustness in complex mobile manipulation tasks. |
| 2026-04-09 | [VCAO: Verifier-Centered Agentic Orchestration for Strategic OS Vulnerability Discovery](http://arxiv.org/abs/2604.08291v1) | Suyash Mishra | We formulate operating-system vulnerability discovery as a \emph{repeated Bayesian Stackelberg search game} in which a Large Reasoning Model (LRM) orchestrator allocates analysis budget across kernel files, functions, and attack paths while external verifiers -- static analyzers, fuzzers, and sanitizers -- provide evidence. At each round, the orchestrator selects a target component, an analysis method, and a time budget; observes tool outputs; updates Bayesian beliefs over latent vulnerability states; and re-solves the game to minimize the strategic attacker's expected payoff. We introduce \textsc{VCAO} (\textbf{V}erifier-\textbf{C}entered \textbf{A}gentic \textbf{O}rchestration), a six-layer architecture comprising surface mapping, intra-kernel attack-graph construction, game-theoretic file/function ranking, parallel executor agents, cascaded verification, and a safety governor. Our DOBSS-derived MILP allocates budget optimally across heterogeneous analysis tools under resource constraints, with formal $\tilde{O}(\sqrt{T})$ regret bounds from online Stackelberg learning. Experiments on five Linux kernel subsystems -- replaying 847 historical CVEs and running live discovery on upstream snapshots -- show that \textsc{VCAO} discovers $2.7\times$ more validated vulnerabilities per unit budget than coverage-only fuzzing, $1.9\times$ more than static-analysis-only baselines, and $1.4\times$ more than non-game-theoretic multi-agent pipelines, while reducing false-positive rates reaching human reviewers by 68\%. We release our simulation framework, synthetic attack-graph generator, and evaluation harness as open-source artifacts. |
| 2026-04-09 | [MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning](http://arxiv.org/abs/2604.08203v1) | Zheng Jiang, Heng Guo et al. | Medical Vision-Language Models (VLMs) hold immense promise for complex clinical tasks, but their reasoning capabilities are often constrained by text-only paradigms that fail to ground inferences in visual evidence. This limitation not only curtails performance on tasks requiring fine-grained visual analysis but also introduces risks of visual hallucination in safety-critical applications. Thus, we introduce MedVR, a novel reinforcement learning framework that enables annotation-free visual reasoning for medical VLMs. Its core innovation lies in two synergistic mechanisms: Entropy-guided Visual Regrounding (EVR) uses model uncertainty to direct exploration, while Consensus-based Credit Assignment (CCA) distills pseudo-supervision from rollout agreement. Without any human annotations for intermediate steps, MedVR achieves state-of-the-art performance on diverse public medical VQA benchmarks, significantly outperforming existing models. By learning to reason directly with visual evidence, MedVR promotes the robustness and transparency essential for accelerating the clinical deployment of medical AI. |
| 2026-04-09 | [Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](http://arxiv.org/abs/2604.08178v1) | Jiaxuan Wang, Yulan Hu et al. | In classical Reinforcement Learning from Human Feedback (RLHF), Reward Models (RMs) serve as the fundamental signal provider for model alignment. As Large Language Models evolve into agentic systems capable of autonomous tool invocation and complex reasoning, the paradigm of reward modeling faces unprecedented challenges--most notably, the lack of benchmarks specifically designed to assess RM capabilities within tool-integrated environments. To address this gap, we present Plan-RewardBench, a trajectory-level preference benchmark designed to evaluate how well judges distinguish preferred versus distractor agent trajectories in complex tool-using scenarios. Plan-RewardBench covers four representative task families -- (i) Safety Refusal, (ii) Tool-Irrelevance / Unavailability, (iii) Complex Planning, and (iv) Robust Error Recovery -- comprising validated positive trajectories and confusable hard negatives constructed via multi-model natural rollouts, rule-based perturbations, and minimal-edit LLM perturbations. We benchmark representative RMs (generative, discriminative, and LLM-as-Judge) under a unified pairwise protocol, reporting accuracy trends across varying trajectory lengths and task categories. Furthermore, we provide diagnostic analyses of prevalent failure modes. Our results reveal that all three evaluator families face substantial challenges, with performance degrading sharply on long-horizon trajectories, underscoring the necessity for specialized training in agentic, trajectory-level reward modeling. Ultimately, Plan-RewardBench aims to serve as both a practical evaluation suite and a reusable blueprint for constructing agentic planning preference data. |
| 2026-04-09 | [Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence](http://arxiv.org/abs/2604.08169v1) | Niklas Herbster, Martin Zborowski et al. | Alignment in LLMs is more brittle than commonly assumed: misalignment can be triggered by adversarial prompts, benign fine-tuning, emergent misalignment, and goal misgeneralization. Recent evidence suggests that some misalignment behaviors are encoded as linear structure in activation space, making it tractable via steering, while safety alignment has been shown to govern the first few output tokens primarily, leaving subsequent generation unguarded. These findings motivate activation steering as a lightweight runtime defense that continuously corrects misaligned activations throughout generation. We evaluate three methods: Steer-With-Fixed-Coeff (SwFC), which applies uniform additive steering, and two novel projection-aware methods, Steer-to-Target-Projection (StTP) and Steer-to-Mirror-Projection (StMP), that use a logistic regression decision boundary to selectively intervene only on tokens whose activations fall below distributional thresholds. Using malicious system prompts as a controlled proxy for misalignment, we evaluate under two threat models (dishonesty and dismissiveness) and two architectures (Llama-3.3-70B-Instruct, Qwen3-32B). All methods substantially recover target traits (honesty and compassion) while preserving coherence. StTP and StMP better maintain general capabilities (MMLU, MT-Bench, AlpacaEval) and produce less repetition in multi-turn conversations. |
| 2026-04-09 | [ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](http://arxiv.org/abs/2604.08064v1) | Chonghan Qin, Xiachong Feng et al. | Existing memory benchmarks for LLM agents evaluate explicit recall of facts, yet overlook implicit memory where experience becomes automated behavior without conscious retrieval. This gap is critical: effective assistants must automatically apply learned procedures or avoid failed actions without explicit reminders. We introduce ImplicitMemBench, the first systematic benchmark evaluating implicit memory through three cognitively grounded constructs drawn from standard cognitive-science accounts of non-declarative memory: Procedural Memory (one-shot skill acquisition after interference), Priming (theme-driven bias via paired experimental/control instances), and Classical Conditioning (Conditioned Stimulus--Unconditioned Stimulus (CS--US) associations shaping first decisions). Our 300-item suite employs a unified Learning/Priming-Interfere-Test protocol with first-attempt scoring. Evaluation of 17 models reveals severe limitations: no model exceeds 66% overall, with top performers DeepSeek-R1 (65.3%), Qwen3-32B (64.1%), and GPT-5 (63.0%) far below human baselines. Analysis uncovers dramatic asymmetries (inhibition 17.6% vs. preference 75.0%) and universal bottlenecks requiring architectural innovations beyond parameter scaling. ImplicitMemBench reframes evaluation from "what agents recall" to "what they automatically enact". |
| 2026-04-09 | [LINE: LLM-based Iterative Neuron Explanations for Vision Models](http://arxiv.org/abs/2604.08039v1) | Vladimir Zaigrajew, Michał Piechota et al. | Interpreting the concepts encoded by individual neurons in deep neural networks is a crucial step towards understanding their complex decision-making processes and ensuring AI safety. Despite recent progress in neuron labeling, existing methods often limit the search space to predefined concept vocabularies or produce overly specific descriptions that fail to capture higher-order, global concepts. We introduce LINE, a novel, training-free iterative approach tailored for open-vocabulary concept labeling in vision models. Operating in a strictly black-box setting, LINE leverages a large language model and a text-to-image generator to iteratively propose and refine concepts in a closed loop, guided by activation history. We demonstrate that LINE achieves state-of-the-art performance across multiple model architectures, yielding AUC improvements of up to 0.18 on ImageNet and 0.05 on Places365, while discovering, on average, 29% of new concepts missed by massive predefined vocabularies. Beyond identifying the top concept, LINE provides a complete generation history, which enables polysemanticity evaluation and produces supporting visual explanations that rival gradient-dependent activation maximization methods. |
| 2026-04-09 | [Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles](http://arxiv.org/abs/2604.08031v1) | Jiawei Liu, Xun Gong et al. | Most Human-Machine Interaction (HMI) research overlooks the maneuvering needs of passengers in autonomous driving (AD). Natural language offers an intuitive interface, yet translating passenger open-ended instructions into control signals, without sacrificing interpretability and traceability, remains a challenge. This study proposes an instruction-realization framework that leverages a large language model (LLM) to interpret instructions, generates executable scripts that schedule multiple model predictive control (MPC)-based motion planners based on real-time feedback, and converts planned trajectories into control signals. This scheduling-centric design decouples semantic reasoning from vehicle control at different timescales, establishing a transparent, traceable decision-making chain from high-level instructions to low-level actions. Due to the absence of high-fidelity evaluation tools, this study introduces a benchmark for open-ended instruction realization in a closed-loop setting. Comprehensive experiments reveal that the framework significantly improves task-completion rates over instruction-realization baselines, reduces LLM query costs, achieves safety and compliance on par with specialized AD approaches, and exhibits considerable tolerance to LLM inference latency. For more qualitative illustrations and a clearer understanding. |
| 2026-04-09 | [Spatiotemporally Resolved Multi-Scalar Measurements of Methane Tulip Flames in a Square Channel](http://arxiv.org/abs/2604.08027v1) | Zeyu Yan, Shengkai Wang | Understanding the propagation dynamics of premixed flames in confined spaces is important for fire safety in gas pipelines and for optimizing modern internal combustion engines. In sufficiently long channels, premixed flames routinely develop tulip flame structures, yet the dominant mechanism remains elusive, and quantitative data on the evolution of flame morphology and key scalar fields are critically needed to improve the explanation, characterization, and modeling of tulip flame dynamics. In this study, premixed flames of a stoichiometric methane/air mixture were investigated in a square channel at a reduced pressure of approximately 0.3 atm. Time-synchronized, multi-plane, dual-color PLIF measurements yielded a spatiotemporally resolved 3-D dataset of key scalar fields, including temperature and OH concentration, throughout the formation and evolution of the tulip structure. Significant heat loss across the walls counteracted the heat released by combustion, producing a near-constant-pressure environment throughout the experiment. A super-equilibrium distribution of OH concentration was observed in the thermal boundary layers, suggesting that thermal cooling dominated over chemical relaxation in those regions. Additionally, the flame-front morphology at five representative times was determined using a 3-D reconstruction algorithm, from which the flame surface area was extracted. The results of this study should aid theoretical modeling and numerical simulations of premixed flame propagation dynamics in confined spaces under realistic boundary conditions. |
| 2026-04-09 | [SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](http://arxiv.org/abs/2604.08008v1) | Felix Embacher, Jonas Uhrig et al. | Retrieving rare and safety-critical driving scenarios from large-scale datasets is essential for building robust autonomous driving (AD) systems. As dataset sizes continue to grow, the key challenge shifts from collecting more data to efficiently identifying the most relevant samples. We introduce SearchAD, a large-scale rare image retrieval dataset for AD containing over 423k frames drawn from 11 established datasets. SearchAD provides high-quality manual annotations of more than 513k bounding boxes covering 90 rare categories. It specifically targets the needle-in-a-haystack problem of locating extremely rare classes, with some appearing fewer than 50 times across the entire dataset. Unlike existing benchmarks, which focused on instance-level retrieval, SearchAD emphasizes semantic image retrieval with a well-defined data split, enabling text-to-image and image-to-image retrieval, few-shot learning, and fine-tuning of multi-modal retrieval models. Comprehensive evaluations show that text-based methods outperform image-based ones due to stronger inherent semantic grounding. While models directly aligning spatial visual features with language achieve the best zero-shot results, and our fine-tuning baseline significantly improves performance, absolute retrieval capabilities remain unsatisfactory. With a held-out test set on a public benchmark server, SearchAD establishes the first large-scale dataset for retrieval-driven data curation and long-tail perception research in AD: https://iis-esslingen.github.io/searchad/ |
| 2026-04-09 | [AFGNN: API Misuse Detection using Graph Neural Networks and Clustering](http://arxiv.org/abs/2604.07891v1) | Ponnampalam Pirapuraj, Tamal Mondal et al. | Application Programming Interfaces (APIs) are crucial to software development, enabling integration of existing systems with new applications by reusing tried and tested code, saving development time and increasing software safety. In particular, the Java standard library APIs, along with numerous third-party APIs, are extensively utilized in the development of enterprise application software. However, their misuse remains a significant source of bugs and vulnerabilities. Furthermore, due to the limited examples in the official API documentation, developers often rely on online portals and generative AI models to learn unfamiliar APIs, but using such examples may introduce unintentional errors in the software. In this paper, we present AFGNN, a novel Graph Neural Network (GNN)-based framework for efficiently detecting API misuses in Java code. AFGNN uses a novel API Flow Graph (AFG) representation that captures the API execution sequence, data, and control flow information present in the code to model the API usage patterns. AFGNN uses self-supervised pre-training with AFG representation to effectively compute the embeddings for unknown API usage examples and cluster them to identify different usage patterns. Experiments on popular API usage datasets show that AFGNN significantly outperforms state-of-the-art small language models and API misuse detectors. |
| 2026-04-09 | [FlowGuard: Towards Lightweight In-Generation Safety Detection for Diffusion Models via Linear Latent Decoding](http://arxiv.org/abs/2604.07879v1) | Jinghan Yang, Yihe Fan et al. | Diffusion-based image generation models have advanced rapidly but pose a safety risk due to their potential to generate Not-Safe-For-Work (NSFW) content. Existing NSFW detection methods mainly operate either before or after image generation. Pre-generation methods rely on text prompts and struggle with the gap between prompt safety and image safety. Post-generation methods apply classifiers to final outputs, but they are poorly suited to intermediate noisy images. To address this, we introduce FlowGuard, a cross-model in-generation detection framework that inspects intermediate denoising steps. This is particularly challenging in latent diffusion, where early-stage noise obscures visual signals. FlowGuard employs a novel linear approximation for latent decoding and leverages a curriculum learning approach to stabilize training. By detecting unsafe content early, FlowGuard reduces unnecessary diffusion steps to cut computational costs. Our cross-model benchmark spanning nine diffusion-based backbones shows the effectiveness of FlowGuard for in-generation NSFW detection in both in-distribution and out-of-distribution settings, outperforming existing methods by over 30% in F1 score while delivering transformative efficiency gains, including slashing peak GPU memory demand by over 97% and projection time from 8.1 seconds to 0.2 seconds compared to standard VAE decoding. |
| 2026-04-09 | [Learning over Forward-Invariant Policy Classes: Reinforcement Learning without Safety Concerns](http://arxiv.org/abs/2604.07875v1) | Chieh Tsai, Muhammad Junayed Hasan Zahed et al. | This paper proposes a safe reinforcement learning (RL) framework based on forward-invariance-induced action-space design. The control problem is cast as a Markov decision process, but instead of relying on runtime shielding or penalty-based constraints, safety is embedded directly into the action representation. Specifically, we construct a finite admissible action set in which each discrete action corresponds to a stabilizing feedback law that preserves forward invariance of a prescribed safe state set. Consequently, the RL agent optimizes policies over a safe-by-construction policy class. We validate the framework on a quadcopter hover-regulation problem under disturbance. Simulation results show that the learned policy improves closed-loop performance and switching efficiency, while all evaluated policies remain safety-preserving. The proposed formulation decouples safety assurance from performance optimization and provides a promising foundation for safe learning in nonlinear systems. |

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



