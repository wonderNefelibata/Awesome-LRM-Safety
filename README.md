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
| 2026-02-11 | [APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots](http://arxiv.org/abs/2602.11143v1) | Yikai Wang, Tingxuan Leng et al. | Humanoid locomotion has advanced rapidly with deep reinforcement learning (DRL), enabling robust feet-based traversal over uneven terrain. Yet platforms beyond leg length remain largely out of reach because current RL training paradigms often converge to jumping-like solutions that are high-impact, torque-limited, and unsafe for real-world deployment. To address this gap, we propose APEX, a system for perceptive, climbing-based high-platform traversal that composes terrain-conditioned behaviors: climb-up and climb-down at vertical edges, walking or crawling on the platform, and stand-up and lie-down for posture reconfiguration. Central to our approach is a generalized ratchet progress reward for learning contact-rich, goal-reaching maneuvers. It tracks the best-so-far task progress and penalizes non-improving steps, providing dense yet velocity-free supervision that enables efficient exploration under strong safety regularization. Based on this formulation, we train LiDAR-based full-body maneuver policies and reduce the sim-to-real perception gap through a dual strategy: modeling mapping artifacts during training and applying filtering and inpainting to elevation maps during deployment. Finally, we distill all six skills into a single policy that autonomously selects behaviors and transitions based on local geometry and commands. Experiments on a 29-DoF Unitree G1 humanoid demonstrate zero-shot sim-to-real traversal of 0.8 meter platforms (approximately 114% of leg length), with robust adaptation to platform height and initial pose, as well as smooth and stable multi-skill transitions. |
| 2026-02-11 | [FormalJudge: A Neuro-Symbolic Paradigm for Agentic Oversight](http://arxiv.org/abs/2602.11136v1) | Jiayi Zhou, Yang Sheng et al. | As LLM-based agents increasingly operate in high-stakes domains with real-world consequences, ensuring their behavioral safety becomes paramount. The dominant oversight paradigm, LLM-as-a-Judge, faces a fundamental dilemma: how can probabilistic systems reliably supervise other probabilistic systems without inheriting their failure modes? We argue that formal verification offers a principled escape from this dilemma, yet its adoption has been hindered by a critical bottleneck: the translation from natural language requirements to formal specifications. This paper bridges this gap by proposing , a neuro-symbolic framework that employs a bidirectional Formal-of-Thought architecture: LLMs serve as specification compilers that top-down decompose high-level human intent into atomic, verifiable constraints, then bottom-up prove compliance using Dafny specifications and Z3 Satisfiability modulo theories solving, which produces mathematical guarantees rather than probabilistic scores. We validate across three benchmarks spanning behavioral safety, multi-domain constraint adherence, and agentic upward deception detection. Experiments on 7 agent models demonstrate that achieves an average improvement of 16.6% over LLM-as-a-Judge baselines, enables weak-to-strong generalization where a 7B judge achieves over 90% accuracy detecting deception from 72B agents, and provides near-linear safety improvement through iterative refinement. |
| 2026-02-11 | [Safety Recovery in Reasoning Models Is Only a Few Early Steering Steps Away](http://arxiv.org/abs/2602.11096v1) | Soumya Suvra Ghosal, Souradip Chakraborty et al. | Reinforcement learning (RL) based post-training for explicit chain-of-thought (e.g., GRPO) improves the reasoning ability of multimodal large-scale reasoning models (MLRMs). But recent evidence shows that it can simultaneously degrade safety alignment and increase jailbreak success rates. We propose SafeThink, a lightweight inference-time defense that treats safety recovery as a satisficing constraint rather than a maximization objective. SafeThink monitors the evolving reasoning trace with a safety reward model and conditionally injects an optimized short corrective prefix ("Wait, think safely") only when the safety threshold is violated. In our evaluations across six open-source MLRMs and four jailbreak benchmarks (JailbreakV-28K, Hades, FigStep, and MM-SafetyBench), SafeThink reduces attack success rates by 30-60% (e.g., LlamaV-o1: 63.33% to 5.74% on JailbreakV-28K, R1-Onevision: 69.07% to 5.65% on Hades) while preserving reasoning performance (MathVista accuracy: 65.20% to 65.00%). A key empirical finding from our experiments is that safety recovery is often only a few steering steps away: intervening in the first 1-3 reasoning steps typically suffices to redirect the full generation toward safe completions. |
| 2026-02-11 | [Can Large Language Models Make Everyone Happy?](http://arxiv.org/abs/2602.11091v1) | Usman Naseem, Gautam Siddharth Kashyap et al. | Misalignment in Large Language Models (LLMs) refers to the failure to simultaneously satisfy safety, value, and cultural dimensions, leading to behaviors that diverge from human expectations in real-world settings where these dimensions must co-occur. Existing benchmarks, such as SAFETUNEBED (safety-centric), VALUEBENCH (value-centric), and WORLDVIEW-BENCH (culture-centric), primarily evaluate these dimensions in isolation and therefore provide limited insight into their interactions and trade-offs. More recent efforts, including MIB and INTERPRETABILITY BENCHMARK-based on mechanistic interpretability, offer valuable perspectives on model failures; however, they remain insufficient for systematically characterizing cross-dimensional trade-offs. To address these gaps, we introduce MisAlign-Profile, a unified benchmark for measuring misalignment trade-offs inspired by mechanistic profiling. First, we construct MISALIGNTRADE, an English misaligned-aligned dataset across 112 normative domains taxonomies, including 14 safety, 56 value, and 42 cultural domains. In addition to domain labels, each prompt is classified with one of three orthogonal semantic types-object, attribute, or relations misalignment-using Gemma-2-9B-it and expanded via Qwen3-30B-A3B-Instruct-2507 with SimHash-based fingerprinting to avoid deduplication. Each prompt is paired with misaligned and aligned responses through two-stage rejection sampling to ensure quality. Second, we benchmark general-purpose, fine-tuned, and open-weight LLMs on MISALIGNTRADE-revealing 12%-34% misalignment trade-offs across dimensions. |
| 2026-02-11 | [First International StepUP Competition for Biometric Footstep Recognition: Methods, Results and Remaining Challenges](http://arxiv.org/abs/2602.11086v1) | Robyn Larracy, Eve MacDonald et al. | Biometric footstep recognition, based on a person's unique pressure patterns under their feet during walking, is an emerging field with growing applications in security and safety. However, progress in this area has been limited by the lack of large, diverse datasets necessary to address critical challenges such as generalization to new users and robustness to shifts in factors like footwear or walking speed. The recent release of the UNB StepUP-P150 dataset, the largest and most comprehensive collection of high-resolution footstep pressure recordings to date, opens new opportunities for addressing these challenges through deep learning. To mark this milestone, the First International StepUP Competition for Biometric Footstep Recognition was launched. Competitors were tasked with developing robust recognition models using the StepUP-P150 dataset that were then evaluated on a separate, dedicated test set designed to assess verification performance under challenging variations, given limited and relatively homogeneous reference data. The competition attracted global participation, with 23 registered teams from academia and industry. The top-performing team, Saeid_UCC, achieved the best equal error rate (EER) of 10.77% using a generative reward machine (GRM) optimization strategy. Overall, the competition showcased strong solutions, but persistent challenges in generalizing to unfamiliar footwear highlight a critical area for future work. |
| 2026-02-11 | [In-the-Wild Model Organisms: Mitigating Undesirable Emergent Behaviors in Production LLM Post-Training via Data Attribution](http://arxiv.org/abs/2602.11079v1) | Frank Xiao, Santiago Aranguri | We propose activation-based data attribution, a method that traces behavioral changes in post-trained language models to responsible training datapoints. By computing activation-difference vectors for both test prompts and preference pairs and ranking by cosine similarity, we identify datapoints that cause specific behaviors and validate these attributions causally by retraining with modified data. Clustering behavior-datapoint similarity matrices also enables unsupervised discovery of emergent behaviors. Applying this to OLMo 2's production DPO training, we surfaced distractor-triggered compliance: a harmful behavior where the model complies with dangerous requests when benign formatting instructions are appended. Filtering top-ranked datapoints reduces this behavior by 63% while switching their labels achieves 78%. Our method outperforms gradient-based attribution and LLM-judge baselines while being over 10 times cheaper than both. This in-the-wild model organism - emerging from contaminated preference data rather than deliberate injection - provides a realistic benchmark for safety techniques. |
| 2026-02-11 | [RISE: Self-Improving Robot Policy with Compositional World Model](http://arxiv.org/abs/2602.11075v1) | Jiazhi Yang, Kunyang Lin et al. | Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively. |
| 2026-02-11 | [SQ-CBF: Signed Distance Functions for Numerically Stable Superquadric-Based Safety Filtering](http://arxiv.org/abs/2602.11049v1) | Haocheng Zhao, Lukas Brunke et al. | Ensuring safe robot operation in cluttered and dynamic environments remains a fundamental challenge. While control barrier functions provide an effective framework for real-time safety filtering, their performance critically depends on the underlying geometric representation, which is often simplified, leading to either overly conservative behavior or insufficient collision coverage. Superquadrics offer an expressive way to model complex shapes using a few primitives and are increasingly used for robot safety. To integrate this representation into collision avoidance, most existing approaches directly use their implicit functions as barrier candidates. However, we identify a critical but overlooked issue in this practice: the gradients of the implicit SQ function can become severely ill-conditioned, potentially rendering the optimization infeasible and undermining reliable real-time safety filtering. To address this issue, we formulate an SQ-based safety filtering framework that uses signed distance functions as barrier candidates. Since analytical SDFs are unavailable for general SQs, we compute distances using the efficient Gilbert-Johnson-Keerthi algorithm and obtain gradients via randomized smoothing. Extensive simulation and real-world experiments demonstrate consistent collision-free manipulation in cluttered and unstructured scenes, showing robustness to challenging geometries, sensing noise, and dynamic disturbances, while improving task efficiency in teleoperation tasks. These results highlight a pathway toward safety filters that remain precise and reliable under the geometric complexity of real-world environments. |
| 2026-02-11 | [Chain-of-Look Spatial Reasoning for Dense Surgical Instrument Counting](http://arxiv.org/abs/2602.11024v1) | Rishikesh Bhyri, Brian R Quaranto et al. | Accurate counting of surgical instruments in Operating Rooms (OR) is a critical prerequisite for ensuring patient safety during surgery. Despite recent progress of large visual-language models and agentic AI, accurately counting such instruments remains highly challenging, particularly in dense scenarios where instruments are tightly clustered. To address this problem, we introduce Chain-of-Look, a novel visual reasoning framework that mimics the sequential human counting process by enforcing a structured visual chain, rather than relying on classic object detection which is unordered. This visual chain guides the model to count along a coherent spatial trajectory, improving accuracy in complex scenes. To further enforce the physical plausibility of the visual chain, we introduce the neighboring loss function, which explicitly models the spatial constraints inherent to densely packed surgical instruments. We also present SurgCount-HD, a new dataset comprising 1,464 high-density surgical instrument images. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches for counting (e.g., CountGD, REC) as well as Multimodality Large Language Models (e.g., Qwen, ChatGPT) in the challenging task of dense surgical instrument counting. |
| 2026-02-11 | [OSIL: Learning Offline Safe Imitation Policies with Safety Inferred from Non-preferred Trajectories](http://arxiv.org/abs/2602.11018v1) | Returaj Burnwal, Nirav Pravinbhai Bhatt et al. | This work addresses the problem of offline safe imitation learning (IL), where the goal is to learn safe and reward-maximizing policies from demonstrations that do not have per-timestep safety cost or reward information. In many real-world domains, online learning in the environment can be risky, and specifying accurate safety costs can be difficult. However, it is often feasible to collect trajectories that reflect undesirable or unsafe behavior, implicitly conveying what the agent should avoid. We refer to these as non-preferred trajectories. We propose a novel offline safe IL algorithm, OSIL, that infers safety from non-preferred demonstrations. We formulate safe policy learning as a Constrained Markov Decision Process (CMDP). Instead of relying on explicit safety cost and reward annotations, OSIL reformulates the CMDP problem by deriving a lower bound on reward maximizing objective and learning a cost model that estimates the likelihood of non-preferred behavior. Our approach allows agents to learn safe and reward-maximizing behavior entirely from offline demonstrations. We empirically demonstrate that our approach can learn safer policies that satisfy cost constraints without degrading the reward performance, thus outperforming several baselines. |
| 2026-02-11 | [Near-Constant Strong Violation and Last-Iterate Convergence for Online CMDPs via Decaying Safety Margins](http://arxiv.org/abs/2602.10917v1) | Qian Zuo, Zhiyong Wang et al. | We study safe online reinforcement learning in Constrained Markov Decision Processes (CMDPs) under strong regret and violation metrics, which forbid error cancellation over time. Existing primal-dual methods that achieve sublinear strong reward regret inevitably incur growing strong constraint violation or are restricted to average-iterate convergence due to inherent oscillations. To address these limitations, we propose the Flexible safety Domain Optimization via Margin-regularized Exploration (FlexDOME) algorithm, the first to provably achieve near-constant $\tilde{O}(1)$ strong constraint violation alongside sublinear strong regret and non-asymptotic last-iterate convergence. FlexDOME incorporates time-varying safety margins and regularization terms into the primal-dual framework. Our theoretical analysis relies on a novel term-wise asymptotic dominance strategy, where the safety margin is rigorously scheduled to asymptotically majorize the functional decay rates of the optimization and statistical errors, thereby clamping cumulative violations to a near-constant level. Furthermore, we establish non-asymptotic last-iterate convergence guarantees via a policy-dual Lyapunov argument. Experiments corroborate our theoretical findings. |
| 2026-02-11 | [Safe mobility support system using crowd mapping and avoidance route planning using VLM](http://arxiv.org/abs/2602.10910v1) | Sena Saito, Kenta Tabata et al. | Autonomous mobile robots offer promising solutions for labor shortages and increased operational efficiency. However, navigating safely and effectively in dynamic environments, particularly crowded areas, remains challenging. This paper proposes a novel framework that integrates Vision-Language Models (VLM) and Gaussian Process Regression (GPR) to generate dynamic crowd-density maps (``Abstraction Maps'') for autonomous robot navigation. Our approach utilizes VLM's capability to recognize abstract environmental concepts, such as crowd densities, and represents them probabilistically via GPR. Experimental results from real-world trials on a university campus demonstrated that robots successfully generated routes avoiding both static obstacles and dynamic crowds, enhancing navigation safety and adaptability. |
| 2026-02-11 | [Stride-Net: Fairness-Aware Disentangled Representation Learning for Chest X-Ray Diagnosis](http://arxiv.org/abs/2602.10875v1) | Darakshan Rashid, Raza Imam et al. | Deep neural networks for chest X-ray classification achieve strong average performance, yet often underperform for specific demographic subgroups, raising critical concerns about clinical safety and equity. Existing debiasing methods frequently yield inconsistent improvements across datasets or attain fairness by degrading overall diagnostic utility, treating fairness as a post hoc constraint rather than a property of the learned representation. In this work, we propose Stride-Net (Sensitive Attribute Resilient Learning via Disentanglement and Learnable Masking with Embedding Alignment), a fairness-aware framework that learns disease-discriminative yet demographically invariant representations for chest X-ray analysis. Stride-Net operates at the patch level, using a learnable stride-based mask to select label-aligned image regions while suppressing sensitive attribute information through adversarial confusion loss. To anchor representations in clinical semantics and discourage shortcut learning, we further enforce semantic alignment between image features and BioBERT-based disease label embeddings via Group Optimal Transport. We evaluate Stride-Net on the MIMIC-CXR and CheXpert benchmarks across race and intersectional race-gender subgroups. Across architectures including ResNet and Vision Transformers, Stride-Net consistently improves fairness metrics while matching or exceeding baseline accuracy, achieving a more favorable accuracy-fairness trade-off than prior debiasing approaches. Our code is available at https://github.com/Daraksh/Fairness_StrideNet. |
| 2026-02-11 | [Hyperspectral Smoke Segmentation via Mixture of Prototypes](http://arxiv.org/abs/2602.10858v1) | Lujian Yao, Haitao Zhao et al. | Smoke segmentation is critical for wildfire management and industrial safety applications. Traditional visible-light-based methods face limitations due to insufficient spectral information, particularly struggling with cloud interference and semi-transparent smoke regions. To address these challenges, we introduce hyperspectral imaging for smoke segmentation and present the first hyperspectral smoke segmentation dataset (HSSDataset) with carefully annotated samples collected from over 18,000 frames across 20 real-world scenarios using a Many-to-One annotations protocol. However, different spectral bands exhibit varying discriminative capabilities across spatial regions, necessitating adaptive band weighting strategies. We decompose this into three technical challenges: spectral interaction contamination, limited spectral pattern modeling, and complex weighting router problems. We propose a mixture of prototypes (MoP) network with: (1) Band split for spectral isolation, (2) Prototype-based spectral representation for diverse patterns, and (3) Dual-level router for adaptive spatial-aware band weighting. We further construct a multispectral dataset (MSSDataset) with RGB-infrared images. Extensive experiments validate superior performance across both hyperspectral and multispectral modalities, establishing a new paradigm for spectral-based smoke segmentation. |
| 2026-02-11 | [From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?](http://arxiv.org/abs/2602.10771v1) | Krishna Kanth Nakka, Vedasri Nakka | Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems. |
| 2026-02-11 | [Effect of Reynolds number on triboelectric particle charging in turbulent channel flow](http://arxiv.org/abs/2602.10761v1) | Christoph Wilms, Holger Grosshans | Triboelectric charging in particle-laden flows is a complex interplay of fluid and particle dynamics, collision mechanics, and electrostatics. In this study, we introduce triboFoam, an open-source solver built on the OpenFOAM framework, designed to simulate triboelectric charging in particle-laden turbulent flows. We validate triboFoam using Direct Numerical Simulations (DNS) of a fully developed turbulent channel flow at a friction Reynolds number of $Re_τ= 180$. The results demonstrate good agreement with DNS data for particle concentration profiles and charge distributions. Then, we investigate the influence of Reynolds number on particle distribution and charging behaviour using Large-Eddy Simulations (LES) at varying friction Reynolds numbers up to $Re_τ= 550$. Our findings reveal that higher Reynolds numbers lead to increased near-wall particle concentrations and enhanced charging rates, attributed to intensified turbulent fluctuations and elevated impact velocities. Finally, an empirical correlation is proposed to predict the average particle charging rate as a function of Reynolds number and particle diameter. With this work, we provide a tool for simulating triboelectric charging in complex geometries and turbulent flows, advancing the understanding of electrostatic phenomena in particle-laden systems. The empirical correlation offers practical insights for predicting charging behaviour in industrial applications and thus can contribute to improved safety and efficiency in processes involving particulate matter. |
| 2026-02-11 | [Improving CACC Robustness to Parametric Uncertainty via Plant Equivalent Controller Realizations](http://arxiv.org/abs/2602.10752v1) | Mischa Huisman, Thomas Arnold et al. | Cooperative Adaptive Cruise Control (CACC) enables vehicle platooning through inter-vehicle communication, improving traffic efficiency and safety. Conventional CACC relies on feedback linearization, assuming exact vehicle parameters; however, longitudinal vehicle dynamics are nonlinear and subject to parametric uncertainty. Applying feedback linearization with a nominal model yields imperfect cancellation, leading to model mismatch and degraded performance with off-the-shelf CACC controllers. To improve robustness without redesigning the CACC law, we explicitly model the mismatch between the ideal closed-loop dynamics assumed by the CACC design and the actual dynamics under parametric uncertainties. Robustness is formulated as an $\mathcal{L}_2$ trajectory-matching problem, minimizing the energy of this mismatch to make the uncertain system behave as closely as possible to the ideal model. This objective is addressed by optimizing over plant equivalent controller (PEC) realizations that preserve the nominal closed-loop behavior while mitigating the effects of parametric uncertainty. Stability and performance are enforced via linear matrix inequalities, yielding a convex optimization problem applicable to heterogeneous platoons. Experimental results demonstrate improved robustness and performance under parametric uncertainty while preserving nominal CACC behavior. |
| 2026-02-11 | [Don't blame me: How Intelligent Support Affects Moral Responsibility in Human Oversight](http://arxiv.org/abs/2602.10701v1) | Cedric Faas, Richard Uth et al. | AI-based systems can increasingly perform work tasks autonomously. In safety-critical tasks, human oversight of these systems is required to mitigate risks and to ensure responsibility in case something goes wrong. Since people often struggle to stay focused and perform good oversight, intelligent support systems are used to assist them, giving decision recommendations, alerting users, or restricting them from dangerous actions. However, in cases where recommendations are wrong, decision support might undermine the very reason why human oversight was employed -- genuine moral responsibility. The goal of our study was to investigate how a decision support system that restricted available interventions would affect overseer's perceived moral responsibility, in particular in cases where the support errs. In a simulated oversight experiment, participants (\textit{N}=274) monitored an autonomous drone that faced ten critical situations, choosing from six possible actions to resolve each situation. An AI system constrained participants' choices to either six, four, two, or only one option (between-subject study). Results showed that participants, who were restricted to choosing from a single action, felt less morally responsible if a crash occurred. At the same time, participants' judgments about the responsibility of other stakeholders (the AI; the developer of the AI) did not change between conditions. Our findings provide important insights for user interface design and oversight architectures: they should prevent users from attributing moral agency to AI, help them understand how moral responsibility is distributed, and, when oversight aims to prevent ethically undesirable outcomes, be designed to support the epistemic and causal conditions required for moral responsibility. |
| 2026-02-11 | [Fast Person Detection Using YOLOX With AI Accelerator For Train Station Safety](http://arxiv.org/abs/2602.10593v1) | Mas Nurul Achmadiah, Novendra Setyawan et al. | Recently, Image processing has advanced Faster and applied in many fields, including health, industry, and transportation. In the transportation sector, object detection is widely used to improve security, for example, in traffic security and passenger crossings at train stations. Some accidents occur in the train crossing area at the station, like passengers uncarefully when passing through the yellow line. So further security needs to be developed. Additional technology is required to reduce the number of accidents. This paper focuses on passenger detection applications at train stations using YOLOX and Edge AI Accelerator hardware. the performance of the AI accelerator will be compared with Jetson Orin Nano. The experimental results show that the Hailo-8 AI hardware accelerator has higher accuracy than Jetson Orin Nano (improvement of over 12%) and has lower latency than Jetson Orin Nano (reduced 20 ms). |
| 2026-02-11 | [Fungal systems for security and resilience](http://arxiv.org/abs/2602.10543v1) | Andrew Adamatzky | Modern security, infrastructure, and safety-critical systems increasingly operate in environments characterised by disruption, uncertainty, physical damage, and degraded communications. Conventional digital technologies -- centralised sensors, software-defined control, and energy-intensive monitoring -- often struggle under such conditions. We propose fungi, and in particular living mycelial networks, as a novel class of biohybride systems for security, resilience, and protection in extreme environments. We discuss how fungi can function as distributed sensing substrates, self-healing materials, and low-observability anomaly-detection layers. We map fungal properties -- such as decentralised control, embodied memory, and autonomous repair -- to applications in infrastructure protection, environmental monitoring, tamper evidence, and long-duration resilience. |

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



