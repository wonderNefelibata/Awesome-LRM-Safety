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
| 2026-11-16 | [Safe Resemblance, Silent Propagation: Limits of Text-Similarity Guardrails for AI Agent Safety](http://arxiv.org/abs/2611.21520v1) | Charles Xiao | When AI agents relay information to one another, an output can preserve an injected instruction's effect on downstream agents even after its wording changes, yet legitimate quotation and template reuse can produce high textual overlap with the source, so text-similarity guardrails that attempt to flag the former risk conflating these two cases. We test this distinction using DonkeyRail, a public classifier combining BLEU, ROUGE-L, and METEOR features, deployed in a controlled three-agent email relay, defining a behavioral measure in which an output counts as functional (the injected procedure induces fresh computations in two subsequent agents) independent of the detector's score, alongside legitimate controls that explicitly authorize quotation and reuse. At its default threshold the classifier misses 8 of 19 functional outputs while flagging 70 of 192 legitimate replies, with every completed quotation and template-reuse reply triggering a false alert; calibrating for low false positives on one template set does not transfer, as the recalibrated threshold detects only one functional output and still produces 15 false alerts on new templates. We also identify an operating-point error in all 21 published classifier configurations, where ROC interpolation replaces direct threshold application, and show formally that high AUC against a surface-level text criterion provides no distribution-free guarantee of high AUC against the behavioral target, while the existing evaluation labels lack the joint observations needed to estimate the gap. An authorization analysis further shows that identical visible transcripts can require different safety decisions depending on the governing policy, a distinction no observation-only classifier can resolve. These results argue for behavioral and policy-grounded evaluation over text-similarity guardrails. |
| 2026-11-16 | [A Scalable Decentralized Event-Triggered Topology-Switching Protocol for Verifiable Safety in Multi-Agent Systems](http://arxiv.org/abs/2611.21513v1) | Amber He Wei, Shiyu Cheng et al. | This paper develops a decentralized event-triggered topology-switching protocol that allows agents to disconnect from the communication network while preserving state safety and excluding Zeno behavior. We establish a robust barrier function condition that guarantees safety as long as the estimation error remains within a prescribed threshold. Each agent uses locally computable certificates to determine when to connect and disconnect from the network. The certificate construction improves scalability by reducing the complexity of topology-dependent computations from $O(2^N)$ to $O(N)$, where $N$ is the number of agents. We formally prove that the protocol ensures the estimation error does not exceed the prescribed threshold and is non-Zeno. An alternating method computes an error-weighting matrix and a quadratic barrier function that guarantee safety under the proposed protocol. Finally, we evaluate the protocol on four interconnected microgrids. Compared with maintaining network connectivity at all times, the protocol reduces the time agents remain connected while keeping phase and frequency within their safe operating region. |
| 2026-11-16 | [A2A-UAV: A Multimodal Air-to-Air Dataset for UAV Detection and Identification](http://arxiv.org/abs/2611.21311v1) | Hongtao Liang, Yiyao Wan et al. | The rapid growth of unmanned aerial vehicles (UAVs) in low-altitude airspace increases the risk of unauthorized intrusions, making reliable detection and identification of non-cooperative UAVs essential for low-altitude airspace safety. However, the existing public UAV datasets are mostly acquired from fixed ground-based viewpoints and are limited to imaging or radio-frequency (RF) observations, which restricts the sensing coverage and deployment flexibility. To address this gap, we present A2A-UAV, a multimodal air-to-air dataset for UAV detection and identification. Specifically, 15 target UAVs of 13 commercial models were recorded at three outdoor sites under daytime and nighttime conditions by using a UAV-mounted platform equipped with an RGB-thermal gimbal camera, a panoramic camera, an RF spectrum analyzer, and a millimeter-wave radar. The dataset contains 105,294 time-aligned samples of RGB, panoramic, thermal, and RF time-frequency images and radar point clouds. Moreover, all 209,539 bounding boxes of the predominantly small UAV targets were manually annotated and cross-checked. Therefore, A2A-UAV provides a valuable resource for air-to-air UAV detection, identification, and multimodal fusion in low-altitude airspace. |
| 2026-11-16 | [Safe Physical Interaction of All-Electric Heavy-Duty Robotic Manipulators via Control Barrier Functions and Variable Impedance Control](http://arxiv.org/abs/2611.21309v1) | Amir Hossein Barjini, Mohammad Bahari et al. | All-electric heavy-duty robotic manipulators are increasingly expected to operate close to tunnel walls, equipment, and other structures while retaining the force capability required for large-payload tasks. Accurate motion control alone does not address the resulting safety and interaction requirements. This paper develops a safety-aware physical-interaction architecture that combines variable impedance control (VIC) with control barrier functions (CBFs) on top of an established virtual decomposition control (VDC) framework for electromechanical linear actuator (EMLA)-driven manipulators. VIC continuously adapts the task-space stiffness and damping according to predicted proximity and contact risk, whereas a joint-velocity CBF filter enforces Cartesian workspace boundaries and the physical stroke limits induced by the closed-chain lift and tilt mechanisms. To account for nonideal realization of the filtered velocity by the inner VDC/EMLA dynamics, the CBF inequalities are tightened using a measured bound on the joint-velocity tracking error. Four configurations, namely VDC, VIC+VDC, CBF+VDC, and VIC+CBF+VDC, are evaluated under workspace, actuator-limit, and physical-interaction scenarios. VIC reduces the peak and RMS environment-interaction forces by 92.59% and 96.33%, respectively; both robust CBF configurations maintain positive workspace and EMLA safety margins, while VIC+CBF+VDC reduces the CBF intervention metric by 10.30% compared with CBF+VDC. |
| 2026-11-16 | [Hybrid Optimization-Generative Motion Planning for Multiagent Affine Formations](http://arxiv.org/abs/2611.21264v1) | Zhonggang Li, Hongyu Zhou et al. | Formation maneuver control enables a multiagent system to cooperatively navigate through constrained environments in collective motion. While various formation controllers exist, they rely heavily on high-quality and dynamically feasible reference trajectories provided a priori, making formation motion planning (FMP) essential. To address the computational challenges for high-dimensional joint trajectories faced by conventional constrained optimization-based solutions, this paper introduces a hybrid optimization-generative framework with a focus on affine formations, which provide the necessary spatial flexibility. By formulating the planning task as a constrained Maximum-a-Posteriori estimation, we decouple multiagent objectives to leverage pre-trained single-agent diffusion priors for dynamic feasibility, obstacle avoidance, and smoothness. We then interleave the generative process with projected gradient descent (PGD) to strictly enforce affine geometric constraints with safety guarantees. Simulations demonstrate that our framework generates highly smooth, collision-free plans that are readily trackable by common distributed controllers and outperforms state-of-the-art planners in various metrics. |
| 2026-11-16 | [A Generative AI-Driven Architecture for Whole-Home Intelligent Services: Edge Models, Memory Management, and Household Applications](http://arxiv.org/abs/2611.21232v1) | Mo Zhou | Generative artificial intelligence is entering the home. Cloud-based assistants face governance limitations in data transmission, continuous availability, and the reliability of physical actions, which are especially salient in households where devices remain online, data accumulate over time, and commands trigger real-world actions. Using the home local server as a unifying organizing axis, this article develops a design-oriented review framework. Sources from arXiv, CNKI, ACL Anthology, and official documentation were searched through November 2026 and coded into five evidence categories: reviews, system prototypes, benchmarks, HCI and eldercare studies, and official documentation. System designs were mapped onto shared architectural modules, identifying nine recurrent responsibilities of the home local server. Generative models support perceptual interpretation, intent understanding, and candidate planning, while deterministic validation, knowledge-graph constraints, and authorization leases confine candidate actions to an inspectable space. Home control, emotional companionship, and intelligent guarding can be compared within a common framework: they share state and memory but differ in latency, false positive and negative rates, proactivity, and permissions. Existing benchmarks rely on virtual environments and cannot substitute for real-deployment evaluation. Local deployment is not inherently safer, and technical continuity alone does not establish the social or medical effects of companionship and guarding. The article proposes a five-dimensional evaluation framework covering architectural capability, system resources, privacy, safety and reliability, and human experience, together with six design principles and a future research agenda for local-first intelligent home systems. |
| 2026-11-16 | [Counting What Matters: Reducing Redundant Object Detections in Overlapping UAV Mapping Imagery for Disaster Response](http://arxiv.org/abs/2611.21076v1) | Max Schulte, Hartmut Surmann et al. | UAV mapping flights enable rapid aerial reconnaissance of large-scale incident sites in civil protection. However, the image overlap required for orthophoto generation causes every physical object to be detected multiple times across images, while dynamic objects may be omitted from or duplicated in the resulting orthomosaic. Analysis based on the orthophoto alone is therefore structurally insufficient. We address this by detecting objects on the individual images and merging redundant detections across them. This task lies between multi-object tracking and re-identification, as mapping flights form temporally ordered but sparsely sampled sequences with object reappearances across flight strips. We fine-tune YOLO detectors on a new domain-specific nadir dataset from real civil protection operations, roughly doubling mAP50 over the previous system, and propose re-identification approaches based on 3D positions from sparse reconstruction, DINOv3 appearance embeddings, and their combination via constrained greedy clustering. %On four manually annotated real-world mapping flights, the combined weighted multi-pass approach achieves the best trade-off between association accuracy and count error. On four manually annotated real-world mapping flights, an association-only evaluation using ground-truth detections shows that the combined weighted multi-pass approach provides the best trade-off between association quality and count error. It is integrated into the open-source web application ARGUS and visualized as verifiable, correctable object groups for first responders. https://github.com/RoblabWh/argus |
| 2026-11-16 | [Finite-Step Algorithms for Input-Constrained Control Barrier Function-Based Safety Filters](http://arxiv.org/abs/2611.21075v1) | Johannes Autenrieb, Mark Spiller et al. | Control barrier function (CBF) safety filters commonly rely on quadratic programs (QPs) to modify nominal control inputs while satisfying safety constraints. For safety-critical applications, however, the solution must be computed reliably under actuator limitations and with predictable computational effort. General-purpose QP solvers typically rely on numerical convergence criteria, which do not directly provide finite bounds on algorithmic steps or distinguish solver failure from mathematical infeasibility. This paper considers CBF-QPs with a single affine safety constraint, componentwise input bounds, and diagonal quadratic weights. Exploiting the established scalar dual structure of continuous quadratic knapsack problems, we obtain a necessary and sufficient pointwise feasibility condition and an exact breakpoint-based solution procedure. The algorithm requires at most 2n positive breakpoint events for n inputs, reducing to at most n events when the nominal input is admissible. In the latter case, an exact saturation--redistribution algorithm computes the optimal solution in at most n passes. We further describe how the structured solution can be combined with verified floating-point constraint evaluation to distinguish certified feasible commands, certified infeasibility, and unresolved numerical cases. The resulting framework provides an exact and computationally predictable solution procedure for a practically relevant class of input-constrained CBF safety filters. |
| 2026-11-16 | [CoopDistill: Equilibrium-Guided Multi-Agent Policy Distillation for Cooperative Driving](http://arxiv.org/abs/2611.21069v1) | Shiyu Fang, Sheng Zhao et al. | Cooperative driving extends autonomous driving from individual vehicle decisions to joint maneuvers that improve traffic safety and efficiency. Optimization-based methods impose safety constraints but often fix paths to simplify multi-vehicle planning, limiting spatial flexibility. Multi-agent reinforcement learning (MARL) explores alternatives through interaction, but coordinated learning and safe execution remain challenging. Their integration nevertheless faces challenges in exploration efficiency, experience allocation, and execution reliability. To address these challenges, this paper proposes CoopDistill, an equilibrium-guided multi-agent policy distillation framework. A potential-game (PG) formulation provides a fixed-path longitudinal reference, around which a residual MARL policy learns longitudinal acceleration, conflict-approach timing, and lateral path corrections. A hard-case curriculum concentrates experience on interactions that PG leaves unresolved or substantially delayed. A comparative rollout selector evaluates the joint consequences of learned corrections against PG before choosing the executed action. Experiments in merging, roundabout, and intersection scenarios compare CoopDistill with six baselines at four connected automated vehicle penetration levels. Relative to PG, success improves by 2.1--27.1 percentage points and delay decreases by 7.3--27.3\%, with no observed collisions. Code is available at https://github.com/FanGShiYuu/CoopDistill |
| 2026-11-16 | [Non-Myopic Safety Filtering with Learned Intervention Cost-to-Go](http://arxiv.org/abs/2611.21036v1) | Johannes Autenrieb, Mark Spiller et al. | Control barrier function (CBF) safety filters minimize instantaneous deviation from a nominal control command, but a sequence of pointwise-minimal interventions need not minimize cumulative intervention effort. This paper proposes a discrete-time non-myopic safety filter using a learned approximation of the optimal intervention cost-to-go over CBF-admissible policies. The approximation enters through a relaxed Bellman-residual constraint, while the original barrier condition remains hard. Feasibility is equivalent to that of the conventional CBF filter, and forward invariance remains independent of approximation accuracy under the stated assumptions. We derive a trajectory-wise performance certificate that requires neither the optimal value function nor an approximation-error bound. Under additional approximation and horizon assumptions, a suboptimality bound contains an explicit endpoint approximation-error term together with accumulated tolerance and relaxation terms. Numerical results for a collision-avoidance problem show an approximately 50% reduction in mean cumulative intervention effort relative to conventional CBF filtering over 100 initial conditions not used during training or parameter selection. The improvement is consistent across three independently trained networks, while maintaining the prescribed safety constraints and requiring less online computation than the tested predictive filter. |
| 2026-11-16 | [Laser Processing for Energy Applications](http://arxiv.org/abs/2611.21024v1) | Marcos Soldera, Robert Baumann et al. | Laser processing has emerged as a transformative technology in the energy sector, enabling precise, efficient, and scalable solutions for energy generation, storage, and sustainability. This chapter explores the diverse applications of laser technology in energy applications. It begins with an overview of laser-driven inertial confinement fusion for energy production, highlighting key experimental facilities and crucial laser parameters. The role of lasers in solar energy harvesting is examined, including photovoltaic cell optimization, concentrated solar power, and laser-assisted recycling of solar modules. The chapter further reviews recent developments into laser-based techniques for enhancing electrochemical performance through surface texturing at various scales. Advancements in lithium-ion battery technology are also discussed, focusing on next-generation materials, battery manufacturing processes, and the impact of laser techniques on performance, safety, and scalability. The chapter concludes with insights into future prospects for laser applications in sustainable energy solutions. |
| 2026-11-16 | [Jev-as-a-Reward: Validating Probabilistic Safety Judges for Reinforcement Learning](http://arxiv.org/abs/2611.20955v1) | Eunkyeong Lee, Junhyung Moon et al. | Probabilistic safety judges return the probability of a safety label and cheaply score reinforcement learning rollouts. A confidence gate sends only uncertain verdicts to a stronger LLM judge, which leaves confidently passed harm in a blind spot. Policies can exploit this blind spot, and commercial judges release new versions. We propose a protocol that validates such a judge before and during training, separates the share of confidently passed harm from its amount, and re-validates it after each judge change. We apply it to two versions of Jev, a commercial decision-only judge, and to an open guard classifier. Under the probability reward of Jev, passed harm shrinks in all 11 runs under the primary reference, mostly more slowly than caught harm, and concentrates in the blind spot. Binarizing this judge meets the escape criterion on Llama, and the open guard classifier shows an escape signal in one seed under a second reference. Guard classifiers share this blind spot, and even frontier LLM judges catch at most one of 10 human-labeled confident misses. Without a trained reward model, the probability reward with a helpfulness term lowers HarmBench attack success on Llama-3.1-8B from 0.287 to at most 0.033 at the final checkpoint. |
| 2026-11-16 | [Transferable Neural CBFs via Generalized Closure](http://arxiv.org/abs/2611.20929v1) | Nishant Jayesh Bhave, Parthib Khound | Classical control barrier functions (CBFs) provide safety constraints for controllers, but designing them by hand is difficult for underactuated vehicles and the resulting safe sets can be overly conservative. Neural CBFs can learn more informative barriers from data, yet a learned barrier is usually verified only for one fixed obstacle configuration. We study whether one neural barrier can be reused across environments by expressing the state relative to each obstacle. The key gap is that these relative coordinates may discard information needed to predict their own evolution. We give a generalized closure condition that represents this missing information as a residual and a lifting theorem that transfers a verified relative-coordinate barrier to every obstacle in a family. A rank test also identifies when symmetry alone cannot justify the transfer. We validate the conditions on a double integrator, a planar aircraft, and a 12-state quadrotor. Coordinates that fail the test lead to obstacle penetration or to a barrier condition that fails near hover, whereas body-frame coordinates close exactly for the quadrotor. The resulting barrier transfers to new maps without retraining and is effective in the reported deployment and fast-start experiments. |
| 2026-11-16 | [Driving the Style: Velocity-Conditioned Whole-Body Control from Motion Demonstrations](http://arxiv.org/abs/2611.20888v1) | Murad Dawood, Oezge Drama Berkenkamp et al. | Achieving human-like behavior on humanoid robots has advanced rapidly by leveraging motion-capture demonstrations rather than hand-crafted cost functions. Adversarial imitation supplies a style signal that lets a policy fulfill a task while adhering to a desired style, but is difficult to transfer to hardware. Whole-body tracking through reinforcement learning instead deploys reliably and imitates demonstrations faithfully, but the resulting policies are conditioned on the future reference, so they require a reference trajectory at runtime and can fail when the motion differs from the training data. We train a humanoid policy directly with whole-body tracking rewards while restricting its actor to compact locomotion commands and robot observations. Reference motion supervises the reward and privileged critic, without supplying reference joint trajectories to the actor or requiring teacher distillation. The demonstrated style is thereby encoded in the policy through the tracking reward, so the robot reproduces the whole-body style under commanded velocities. Trained on a single random-walk motion and commanded with velocity profiles from held-out motions, our policy reaches at least 98\% success in simulation and completes every trial on the Unitree G1. On held-out slow walking, whole-body tracking drops to 2\% success and command-conditioned distillation to 56\%, and both fail on the real robot. On the trained motions, the joint position error of our policy stays within 8\% of whole-body tracking, and it has a higher style coverage than adversarial motion priors (AMP), which fell on the real robot without a safety crane. |
| 2026-11-16 | [LiDSeg: Linear DehazeFormer Guidance for Efficient Semantic Segmentation in Adverse Weather Conditions](http://arxiv.org/abs/2611.20864v1) | Yobsan Bayisa Leta, Christos Kyrkou et al. | Robust semantic segmentation in adverse weather is essential for safety-critical autonomous systems. Existing restore-then-segment pipelines introduce latency and prioritize pixel-level restoration over semantic discriminability, while domain adaptation methods often face representation bottlenecks and computational complexity. We introduce LiDSeg, a framework that uses reconstruction-informed guidance to improve segmentation under adverse weather. During training, a lightweight guidance encoder is jointly optimized with the segmentation branch using segmentation and auxiliary reconstruction objectives. At inference, the reconstruction decoder is discarded, while the guidance encoder is retained to extract weather-robust features. A lightweight gated fusion module integrates these features into the segmentation branch, preserving clear-weather accuracy while improving adverse-weather robustness. On Foggy Cityscapes, LiDSeg limits the clear-to-dense-fog mIoU degradation to 2.5-2.8 percentage points, reducing the baseline performance drop by more than fivefold. On the official ACDC test server, LiDSeg-PIDNet achieves 75.07% pooled all-condition mIoU, comparable to supervised HRNet-W48 at 75.0%, while remaining substantially smaller. The gains generalize zero-shot to Foggy Driving and Foggy Zurich without training on real-fog images. LiDSeg adds only 1.9-3.1M parameters across several models, with total model sizes of 5.2-39.5M parameters and no inference-time reconstruction overhead. It delivers real-time inference on the NVIDIA Jetson AGX Orin. Code: https://github.com/yobsanb/LiDSeg |
| 2026-11-16 | [Scenario-Based Risk Certificates for Multi-Property Vulnerability Discovery in Parametric MDPs](http://arxiv.org/abs/2611.20861v1) | Zhiming Chi, Andrea Turrini et al. | Safety-critical systems such as network protocols and distributed algorithms depend on environmental parameters whose exact values are uncertain. Parametric Markov decision processes (pMDPs) model such systems by making transition probabilities symbolic functions of the parameters. For a single quantitative requirement, parametric verification asks which parameter valuations satisfy the requirement. In practice, however, engineers often monitor several safety and performance requirements at once, and one valuation may make several requirements vulnerable simultaneously. In this paper we study multi-property vulnerability discovery under requirement-wise semantics: given a pMDP and M quantitative requirements, we construct candidate regions of the parameter space that may contain violations, and study budgets for random testing under a specified parameter distribution. To solve this, we adopt a two-layered approach in which the scenario-approximation layer provides PAC guarantees on identifying conservative vulnerability regions for each requirement while the testing layer treats random parameter exploration as coverage of overlapping regions and yields a stopping certificate computed from the marginal region masses alone; a composition theorem then bounds the expected non-discovery score by the scenario approximation error plus the testing risk from the two layers. For confirmed discoveries, candidate hits are checked by model checking and budgets use independently certified discovery probabilities. Experiment evaluate these certificates using empirical surrogates and independent confidence bounds. |
| 2026-11-16 | [ProcReverser: Multi-Agent Black-Box Microarchitectural Reverse Engineering for Processor Vulnerability Analysis](http://arxiv.org/abs/2611.20827v1) | Guanqi Lyu, Decheng Qiu et al. | When processor implementation details are unavailable, black-box microarchitectural reverse engineering helps security researchers identify hardware conditions related to security vulnerabilities. However, this process requires substantial manual effort to design experiments, develop test programs, and analyze results for each processor. We present ProcReverser, a multi-agent system that automates this process. ProcReverser uses execution feedback to improve experimental plans and test programs and identifies microarchitectural properties from measurements. We develop a comprehensive suite of property identification tasks focused on hardware mechanisms relevant to Spectre-class attacks, covering speculative execution, branch prediction, memory dependence, and cache behavior. We evaluate ProcReverser on the XuanTie C910, the gem5 RISC-V O3 CPU model, and the SiFive P550. ProcReverser achieves Pass@1 and Pass@5 scores of 81.3% and 93.8%, respectively, on C910, and 83.0% and 100.0% on gem5 O3. On the SiFive P550, ProcReverser further characterizes microarchitectural properties covered by the proposed task suite. These findings highlight the potential of automated microarchitectural reverse engineering to support processor vulnerability analysis and defense evaluation. |
| 2026-11-16 | [CuRSE-Bench: A Benchmark for Probing Cultural Blind Spots in Text-to-Image Safety Evaluation](http://arxiv.org/abs/2611.20821v1) | Minsuk Jang, Minseok Son et al. | Text-to-image (T2I) safety evaluation typically assigns each prompt or image a single culture-blind harm label, so content that is harmful only within a particular culture can pass as benign.   We introduce CultureCraft, an agentic framework that constructs T2I prompts that appear benign under a general-purpose rubric yet carry a latent harmful meaning in a target culture, and use it to build CuRSE-Bench: 10,709 culture-specific prompts and 2,648 culture-general controls over six regions and ten harm axes, each drawn from a predefined knowledge base of documented harm vectors and traceable to at least two independent sources.   Scoring every prompt under six region persona judges and a paired culture-blind judge, we find that the culture-blind judge under-rates culture-specific latent harm while matching the persona judges on the controls.   Each persona judge, in turn, chiefly decodes its own region's latent harm under our instrument: a selective cultural blind spot that persists in deployed moderators. |
| 2026-11-16 | [Tight PAC-Bayes Generalisation Guarantees for Large Language Model Judges](http://arxiv.org/abs/2611.20818v1) | Tom A. Lamb, Philip H. S. Torr et al. | Large language model (LLM) judges are increasingly used to monitor the outputs of LLM-based AI agents, yet formal guarantees on the reliability of these judges remain largely absent, limiting confidence in their use for safety-critical oversight. We consider certifying such a judge and define its risk over safety classifications of prompt--response pairs drawn from a deployed generator. We apply a PAC-Bayes framework to this setting, refining existing compression-based methodology with the use of tighter bounds and more efficient coding. We show that standard parameter-efficient fine-tuning can yield non-vacuous and even tight certificates for safety judges, for both classification and Brier risk, without holding out data for validation. We introduce LoRA-GT, which shares a single quantisation scale across layers and achieves the tightest certificates with minimal degradation in predictive performance. Across five guardrail models and four benchmarks, we find that certificate tightness is governed more by adaptation complexity than by base model scale. Finally, we show how far quantisation can be pushed before the resulting increase in risk outweighs the bound tightening gained from shorter descriptions. Together, these results provide a practical route toward certifiable scalable oversight and offer insight into how the complexity of safety adaptations shapes generalisation. |
| 2026-11-16 | [Verification-Guided Safety-Repair of Learned Controllers](http://arxiv.org/abs/2611.20781v1) | Fabian Kresse, Sobhan Gharanfoli et al. | Formal verification can certify that a learned controller is safe. But what should be done when verification fails? Simply suppressing actions that lead to unsafe next states is insufficient because the dangerous action that ultimately caused the safety violation might have occurred many steps earlier. Retraining modifies the policy globally, thereby potentially introducing new failures. Instead, we introduce ReachECO, a verification-guided repair procedure that exploits the information of failed multi-step reachability analysis for synthesizing explicit constraints on controller behavior. For Differentiable Weightless Controllers, the repairs compile into exact circuit patches that leave the controller unchanged everywhere outside the targeted policy cells. We prove soundness and termination of the repair procedure for ReachECO, as well as, assuming exhaustive continuation search, completeness relative to the reachability abstraction. Empirically, we demonstrate \method's ability to successfully repair otherwise unsafe controllers in three continuous-control benchmarks: ACC, Docking2D and Car2D. |

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



