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
| 2026-04-01 | [Learning Neural Network Controllers with Certified Robust Performance via Adversarial Training](http://arxiv.org/abs/2604.01188v1) | Neelay Junnarkar, Yasin Sonmez et al. | Neural network (NN) controllers achieve strong empirical performance on nonlinear dynamical systems, yet deploying them in safety-critical settings requires robustness to disturbances and uncertainty. We present a method for jointly synthesizing NN controllers and dissipativity certificates that formally guarantee robust closed-loop performance using adversarial training, in which we use counterexamples to the robust dissipativity condition to guide training. Verification is done post-training using alpha,beta-CROWN, a branch-and-bound-based method that enables direct analysis of the nonlinear dynamical system. The proposed method uses quadratic constraints (QCs) only for characterization of non-parametric uncertainties. The method is tested in numerical experiments on maximizing the volume of the set on which a system is certified to be robustly dissipative. Our method certifies regions up to 78 times larger than the region certified by a linear matrix inequality-based approach that we derive for comparison. |
| 2026-04-01 | [Safe learning-based control via function-based uncertainty quantification](http://arxiv.org/abs/2604.01173v1) | Abdullah Tokmak, Toni Karvonen et al. | Uncertainty quantification is essential when deploying learning-based control methods in safety-critical systems. This is commonly realized by constructing uncertainty tubes that enclose the unknown function of interest, e.g., the reward and constraint functions or the underlying dynamics model, with high probability. However, existing approaches for uncertainty quantification typically rely on restrictive assumptions on the unknown function, such as known bounds on functional norms or Lipschitz constants, and struggle with discontinuities. In this paper, we model the unknown function as a random function from which independent and identically distributed realizations can be generated, and construct uncertainty tubes via the scenario approach that hold with high probability and rely solely on the sampled realizations. We integrate these uncertainty tubes into a safe Bayesian optimization algorithm, which we then use to safely tune control parameters on a real Furuta pendulum. |
| 2026-04-01 | [Data-based Low-conservative Nonlinear Safe Control Learning](http://arxiv.org/abs/2604.01156v1) | Amir Modares, Bahare Kiumarsi et al. | This paper develops a data-driven safe control framework for nonlinear discrete-time systems with parametric uncertainty and additive disturbances. The proposed approach constructs a data-consistent closed-loop representation that enables controller synthesis and safety certification directly from data. Unlike existing methods that treat unmodeled nonlinearities as global worst-case uncertainties using Lipschitz bounds, the proposed approach embeds nonlinear terms directly into the invariance conditions via a geometry-aware difference-of-convex formulation. This enables facet- and direction-specific convexification, avoiding both nonlinearity cancellation and the excessive conservatism induced by uniform global bounds. We further propose a vertex-dependent controller construction that enforces convexity and contractivity conditions locally on the active facets associated with each vertex, thereby enlarging the class of certifiable invariant sets. For systems subject to additive disturbances, disturbance effects are embedded directly into the verification conditions through optimized, geometry-dependent bounds, rather than via uniform margin inflation, yielding less conservative robust safety guarantees. As a result, the proposed methods can certify substantially larger safe sets, naturally accommodate joint state and input constraints, and provide data-driven safety guarantees. The simulation results show a significant improvement in both nonlinearity tolerance and the size of the certified safe set. |
| 2026-04-01 | [VRUD: A Drone Dataset for Complex Vehicle-VRU Interactions within Mixed Traffic](http://arxiv.org/abs/2604.01134v1) | Ziyu Wang, Hongrui Kou et al. | The Operational Design Domain (ODD) of urbanoriented Level 4 (L4) autonomous driving, especially for autonomous robotaxis, confronts formidable challenges in complex urban mixed traffic environments. These challenges stem mainly from the high density of Vulnerable Road Users (VRUs) and their highly uncertain and unpredictable interaction behaviors. However, existing open-source datasets predominantly focus on structured scenarios such as highways or regulated intersections, leaving a critical gap in data representing chaotic, unstructured urban environments. To address this, this paper proposes an efficient, high-precision method for constructing drone-based datasets and establishes the Vehicle-Vulnerable Road User Interaction Dataset (VRUD), as illustrated in Figure 1. Distinct from prior works, VRUD is collected from typical "Urban Villages" in Shenzhen, characterized by loose traffic supervision and extreme occlusion. The dataset comprises 4 hours of 4K/30Hz recording, containing 11,479 VRU trajectories and 1,939 vehicle trajectories. A key characteristic of VRUD is its composition: VRUs account for about 87% of all traffic participants, significantly exceeding the proportions in existing benchmarks. Furthermore, unlike datasets that only provide raw trajectories, we extracted 4,002 multi-agent interaction scenarios based on a novel Vector Time to Collision (VTTC) threshold, supported by standard OpenDRIVE HD maps. This study provides valuable, rare edge-case resources for enhancing the safety performance of ADS in complex, unstructured urban environments. To facilitate further research, we have made the VRUD dataset open-source at: https://zzi4.github.io/VRUD/. |
| 2026-04-01 | [ReinDriveGen: Reinforcement Post-Training for Out-of-Distribution Driving Scene Generation](http://arxiv.org/abs/2604.01129v1) | Hao Zhang, Lue Fan et al. | We present ReinDriveGen, a framework that enables full controllability over dynamic driving scenes, allowing users to freely edit actor trajectories to simulate safety-critical corner cases such as front-vehicle collisions, drifting cars, vehicles spinning out of control, pedestrians jaywalking, and cyclists cutting across lanes. Our approach constructs a dynamic 3D point cloud scene from multi-frame LiDAR data, introduces a vehicle completion module to reconstruct full 360° geometry from partial observations, and renders the edited scene into 2D condition images that guide a video diffusion model to synthesize realistic driving videos. Since such edited scenarios inevitably fall outside the training distribution, we further propose an RL-based post-training strategy with a pairwise preference model and a pairwise reward mechanism, enabling robust quality improvement under out-of-distribution conditions without ground-truth supervision. Extensive experiments demonstrate that ReinDriveGen outperforms existing approaches on edited driving scenarios and achieves state-of-the-art results on novel ego viewpoint synthesis. |
| 2026-04-01 | [Multi-Agent LLM Governance for Safe Two-Timescale Reinforcement Learning in SDN-IoT Defense](http://arxiv.org/abs/2604.01127v1) | Saeid Jamshidi, Negar Shahabi et al. | Software-Defined Networking (SDN) is increasingly adopted to secure Internet-of-Things (IoT) networks due to its centralized control and programmable forwarding. However, SDN-IoT defense is inherently a closed-loop control problem in which mitigation actions impact controller workload, queue dynamics, rule-installation delay, and future traffic observations. Aggressive mitigation may destabilize the control plane, degrade Quality of Service (QoS), and amplify systemic risk. Existing learning-based approaches prioritize detection accuracy while neglecting controller coupling and short-horizon Reinforcement Learning (RL) optimization without structured, auditable policy evolution. This paper introduces a self-reflective two-timescale SDN-IoT defense solution separating fast mitigation from slow policy governance. At the fast timescale, per-switch Proximal Policy Optimization (PPO) agents perform controller-aware mitigation under safety constraints and action masking. At the slow timescale, a multi-agent Large Language Model (LLM) governance engine generates machine-parsable updates to the global policy constitution Pi, which encodes admissible actions, safety thresholds, and reward priorities. Updates (Delta Pi) are validated through stress testing and deployed only with non-regression and safety guarantees, ensuring an auditable evolution without retraining RL agents. Evaluation under heterogeneous IoT traffic and adversarial stress shows improvements of 9.1% Macro-F1 over PPO and 15.4% over static baselines. Worst-case degradation drops by 36.8%, controller backlog peaks by 42.7%, and RTT p95 inflation remains below 5.8% under high-intensity attacks. Policy evolution converges within five cycles, reducing catastrophic overload from 11.6% to 2.3%. |
| 2026-04-01 | [Adversarial Moral Stress Testing of Large Language Models](http://arxiv.org/abs/2604.01108v1) | Saeid Jamshidi, Foutse Khomh et al. | Evaluating the ethical robustness of large language models (LLMs) deployed in software systems remains challenging, particularly under sustained adversarial user interaction. Existing safety benchmarks typically rely on single-round evaluations and aggregate metrics, such as toxicity scores and refusal rates, which offer limited visibility into behavioral instability that may arise during realistic multi-turn interactions. As a result, rare but high-impact ethical failures and progressive degradation effects may remain undetected prior to deployment. This paper introduces Adversarial Moral Stress Testing (AMST), a stress-based evaluation framework for assessing ethical robustness under adversarial multi-round interactions. AMST applies structured stress transformations to prompts and evaluates model behavior through distribution-aware robustness metrics that capture variance, tail risk, and temporal behavioral drift across interaction rounds. We evaluate AMST on several state-of-the-art LLMs, including LLaMA-3-8B, GPT-4o, and DeepSeek-v3, using a large set of adversarial scenarios generated under controlled stress conditions. The results demonstrate substantial differences in robustness profiles across models and expose degradation patterns that are not observable under conventional single-round evaluation protocols. In particular, robustness has been shown to depend on distributional stability and tail behavior rather than on average performance alone. Additionally, AMST provides a scalable and model-agnostic stress-testing methodology that enables robustness-aware evaluation and monitoring of LLM-enabled software systems operating in adversarial environments. |
| 2026-04-01 | [Cosmology from asymptotically safe Proca theories](http://arxiv.org/abs/2604.01090v1) | Carlos Pastor-Marcos, Lavinia Heisenberg et al. | Effective field theories for cosmology offer a powerful framework to investigate the dynamics of space--time and address longstanding open puzzles. In this work, we initiate a programme to analyse the ultraviolet completion of vector--tensor quantum field theories within the asymptotic safety paradigm, focusing on generalised Proca theories with a vector condensate. This enables us to assess whether a consistent fundamental UV completion exists and to constrain the set of viable infrared scenarios. Using the non--perturbative functional renormalisation group, we identify several fixed points, including Proca--type candidates, and, among them, a particularly remarkable one with four relevant directions: two associated with gravity and two induced by matter. This provides evidence for the non--perturbative renormalisability of vector--tensor theories. We further outline how the resulting UV critical surface constrains late--time cosmology. |
| 2026-04-01 | [ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](http://arxiv.org/abs/2604.01081v1) | Yuheng Zhang, Mengfei Duan et al. | 3D semantic occupancy prediction is central to autonomous driving, yet current methods are vulnerable to long-tailed class bias and out-of-distribution (OOD) inputs, often overconfidently assigning anomalies to rare classes. We present ProOOD, a lightweight, plug-and-play method that couples prototype-guided refinement with training-free OOD scoring. ProOOD comprises (i) prototype-guided semantic imputation that fills occluded regions with class-consistent features, (ii) prototype-guided tail mining that strengthens rare-class representations to curb OOD absorption, and (iii) EchoOOD, which fuses local logit coherence with local and global prototype matching to produce reliable voxel-level OOD scores. Extensive experiments on five datasets demonstrate that ProOOD achieves state-of-the-art performance on both in-distribution 3D occupancy prediction and OOD detection. On SemanticKITTI, it surpasses baselines by +3.57% mIoU overall and +24.80% tail-class mIoU; on VAA-KITTI, it improves AuPRCr by +19.34 points, with consistent gains across benchmarks. These improvements yield more calibrated occupancy estimates and more reliable OOD detection in safety-critical urban driving. The source code is publicly available at https://github.com/7uHeng/ProOOD. |
| 2026-04-01 | [A Functional Learning Approach for Team-Optimal Traffic Coordination](http://arxiv.org/abs/2604.01056v1) | Weihao Sun, Gehui Xu et al. | In this paper, we develop a kernel-based policy iteration functional learning framework for computing team-optimal strategies in traffic coordination problems. We consider a multi-agent discrete-time linear system with a cost function that combines quadratic regulation terms and nonlinear safety penalties. Building on the Hilbert space formulation of offline receding-horizon policy iteration, we seek approximate solutions within a reproducing kernel Hilbert space, where the policy improvement step is implemented via a discrete Fréchet derivative. We further study the model-free receding-horizon scenario, where the system dynamics are estimated using recursive least squares, followed by updating the policy using rolling online data. The proposed method is tested in signal-free intersection scenarios via both model-based and model-free simulations and validated in SUMO. |
| 2026-04-01 | [Automated Framework to Evaluate and Harden LLM System Instructions against Encoding Attacks](http://arxiv.org/abs/2604.01039v1) | Anubhab Sahu, Diptisha Samanta et al. | System Instructions in Large Language Models (LLMs) are commonly used to enforce safety policies, define agent behavior, and protect sensitive operational context in agentic AI applications. These instructions may contain sensitive information such as API credentials, internal policies, and privileged workflow definitions, making system instruction leakage a critical security risk highlighted in the OWASP Top 10 for LLM Applications. Without incurring the overhead costs of reasoning models, many LLM applications rely on refusal-based instructions that block direct requests for system instructions, implicitly assuming that prohibited information can only be extracted through explicit queries. We introduce an automated evaluation framework that tests whether system instructions remain confidential when extraction requests are re-framed as encoding or structured output tasks. Across four common models and 46 verified system instructions, we observe high attack success rates (> 0.7) for structured serialization where models refuse direct extraction requests but disclose protected content in the requested serialization formats. We further demonstrate a mitigation strategy based on one-shot instruction reshaping using a Chain-of-Thought reasoning model, indicating that even subtle changes in wording and structure of system instructions can significantly reduce attack success rate without requiring model retraining. |
| 2026-04-01 | [Tube-Based Safety for Anticipative Tracking in Multi-Agent Systems](http://arxiv.org/abs/2604.00992v1) | Armel Koulong, Ali Pakniyat | A tube-based safety framework is presented for robust anticipative tracking in nonlinear Brunovsky multi-agent systems subject to bounded disturbances. The architecture establishes robust safety certificates for a feedforward-augmented ancillary control policy. By rendering the state-deviation dynamics independent of the agents' internal nonlinearities, the formulation strictly circumvents the restrictive Lipschitz-bound feasibility conditions otherwise required for robust stabilization. Consequently, this structure admits an explicit, closed-form robust positively invariant (RPI) tube radius that systematically attenuates the exponential control barrier function (eCBF) tightening margins, thereby mitigating constraint conservatism while preserving formal forward invariance. Within the distributed model predictive control (MPC) layer, mapping the local tube radii through the communication graph yields a closed-form global formation error bound formulated via the minimum singular value of the augmented Laplacian. Robust inter-agent safety is enforced with minimal communication overhead, requiring only a single scalar broadcast per neighbor at initialization. Numerical simulations confirm the framework's efficacy in safely navigating heterogeneous formations through cluttered environments. |
| 2026-04-01 | [An Integrated Soft Robotic System for Measuring Vital Signs in Search and Rescue Environments](http://arxiv.org/abs/2604.00971v1) | Jorge Francisco García-Samartín, Christyan Cruz Ulloa et al. | Robots are frequently utilized in search-and-rescue operations. In recent years, significant advancements have been made in the field of victim assessment. However, there are still open issues regarding heart rate measurement, and no studies have been found that assess pressure in post-disaster scenarios. This work designs a soft gripper and integrates it into a mobile robotic system, thereby creating a device capable of measuring the pulse and blood pressure of victims in post-disaster environments. The gripper is designed to envelop the victim's arm and inflate like a sphygmomanometer, facilitated by a specialized portability system. The utilization of different signal processing algorithms has enabled the attainment of a pulse bias of \qty{4}{\bpm} and a bias of approximately \qty{5}{\mmHg} for systolic and diastolic pressures. The findings, in conjunction with the other statistical data and the validation of homoscedasticity in the error terms, prove the system's capacity to accurately determine heart rate and blood pressure, thereby rendering it suitable for search and rescue operations. Finally, a post-disaster has been employed as a test to validate the functionality of the entire system and to demonstrate its capacity to adapt to various victim positions, its measurement speed, and its safety for victims. |
| 2026-04-01 | [Characterization of Safe Stabilization and Control Lyapunov-Barrier Functions via Zubov Equation Formulation](http://arxiv.org/abs/2604.00941v1) | Yiming Meng, Jun Liu | Design and analysis of stabilizing controllers with safety guarantees for nonlinear systems have received considerable attention in recent years. Control Lyapunov-barrier functions (CLBFs) provide a powerful framework for simultaneously ensuring stability and safety; however, their construction for nonlinear systems remains challenging. To address this issue, we build on recent advances in PDE-based characterizations of control Lyapunov functions and Lyapunov-barrier functions for autonomous systems, and propose a succinct Zubov-HJB PDE formulation for safe stabilization of nonlinear control-affine systems under a common compatibility assumption. We further show that the viscosity solution of this PDE yields a maximal CLBF, enabling (not necessarily continuous) feedback synthesis with stability and safety guarantees. In light of recent advances in neural-network-based methods for solving Zubov-type PDEs, this theoretical framework also provides a natural interface to emerging numerical approaches. |
| 2026-04-01 | [From Pluralistic Ignorance to Common Knowledge with Social Assurance Contracts](http://arxiv.org/abs/2604.00874v1) | Matthew Cashman | Societies and organizations often fail to surface latent consensus because individuals fear social censure. A manager might suspect a silent majority would offer a criticism, support a change, report a risk, or endorse a policy -- if only it were safe. Likewise, individuals with beliefs they think are rare and controversial might stay quiet for fear of consequences at work or an online mob. In both cases pluralistic ignorance produces a public discourse misaligned with privately-held beliefs. Social assurance contracts unlock latent consensus, making the public discussion more accurately reflect the underlying distribution of actual beliefs. They are akin to an open letter that publishes only when a stated threshold number of private signatures is reached. If it is not reached, nothing is revealed and no one is exposed. Whereas a single hand raised in dissent might get cut off, a thousand can be raised safely together. I build a formal model and derive rules for choosing the threshold. The mechanism (i) induces participation from those willing to speak if assured of company, resolving the core coordination problem in pluralistic ignorance; (ii) makes the threshold a transparent policy lever -- sponsors can maximize success, maximize public-coalition revelation, or hit a desired success probability; and (iii) turns success into information: meeting the threshold publicly reveals hidden agreement and can widen the range of views that can be expressed in public. I consider robustness to mistrust, organized opposition, and network structure, and outline low-trust implementations like cryptographic escrow. Applications include employee voice, safety and compliance, whistleblowing, and civic expression. |
| 2026-04-01 | [Evaluating the Feasibility of Augmented Reality to Support Communication Access for Deaf Students in Experiential Higher Education Contexts](http://arxiv.org/abs/2604.00856v1) | Roshan Mathew, Roshan L. Peiris | Deaf and hard of hearing (DHH) students often experience communication barriers in higher education, which are particularly acute in experiential learning environments such as laboratories. Traditional accessibility services, such as interpreting and captioning, often require DHH students to divide their attention between critical tasks, potential safety hazards, instructional materials, and access providers, creating trade-offs between safety and equitable communication. These demands can disrupt task engagement and increase cognitive load in settings that require sustained visual focus, highlighting the limitations of current approaches. To address these challenges, this study investigates Augmented Reality Real-Time Access for Education (ARRAE), an ecosystem based on augmented reality (AR) smart glasses, as a potential intervention for laboratory-based environments. By overlaying interpreters or captions directly into a student's field of view, AR enables the integration of accessibility into hands-on learning without compromising safety or comprehension. Through an empirical study with 12 DHH participants, we evaluate how AR-mediated access influences visual attention patterns and perceived cognitive load during hands-on tasks. The findings suggest that AR-mediated communication shows strong potential to improve attention management and communication accessibility in experiential learning environments, though participants emphasized that accessibility preferences are highly context-dependent. Participants also identified several design and ergonomic challenges, including display positioning, visual fatigue, and compatibility with hearing devices. Together, these results highlight both the promise of AR for supporting accessible participation in visually demanding environments and key design considerations for future systems. |
| 2026-04-01 | [Steering through Time: Blending Longitudinal Data with Simulation to Rethink Human-Autonomous Vehicle Interaction](http://arxiv.org/abs/2604.00832v1) | Yasaman Hakiminejad, Shiva Azimi et al. | As semi-automated vehicles (SAVs) become more common, ensuring effective human-vehicle interaction during control handovers remains a critical safety challenge. Existing studies often rely on single-session simulator experiments or naturalistic driving datasets, which often lack temporal context on drivers' cognitive and physiological states before takeover events. This study introduces a hybrid framework combining longitudinal mobile sensing with high-fidelity driving simulation to examine driver readiness in semi-automated contexts. In a pilot study with 38 participants, we collected 7 days of wearable physiological data and daily surveys on stress, arousal, valence, and sleep quality, followed by an in-lab simulation with scripted takeover events under varying secondary task conditions. Multimodal sensing, including eye tracking, fNIRS, and physiological measures, captured real-time responses. Preliminary analysis shows the framework's feasibility and individual variability in baseline and in-task measures; for example, fixation duration and takeover control time differed by task type, and RMSSD showed high inter-individual stability. This proof-of-concept supports the development of personalized, context-aware driver monitoring by linking temporally layered data with real-time performance. |
| 2026-04-01 | [UK AISI Alignment Evaluation Case-Study](http://arxiv.org/abs/2604.00788v1) | Alexandra Souly, Robert Kirk et al. | This technical report presents methods developed by the UK AI Security Institute for assessing whether advanced AI systems reliably follow intended goals. Specifically, we evaluate whether frontier models sabotage safety research when deployed as coding assistants within an AI lab. Applying our methods to four frontier models, we find no confirmed instances of research sabotage. However, we observe that Claude Opus 4.5 Preview (a pre-release snapshot of Opus 4.5) and Sonnet 4.5 frequently refuse to engage with safety-relevant research tasks, citing concerns about research direction, involvement in self-training, and research scope. We additionally find that Opus 4.5 Preview shows reduced unprompted evaluation awareness compared to Sonnet 4.5, while both models can distinguish evaluation from deployment scenarios when prompted. Our evaluation framework builds on Petri, an open-source LLM auditing tool, with a custom scaffold designed to simulate realistic internal deployment of a coding agent. We validate that this scaffold produces trajectories that all tested models fail to reliably distinguish from real deployment data. We test models across scenarios varying in research motivation, activity type, replacement threat, and model autonomy. Finally, we discuss limitations including scenario coverage and evaluation awareness. |
| 2026-04-01 | [Neural Vector Lyapunov-Razumikhin Certificates for Delayed Interconnected Systems](http://arxiv.org/abs/2604.00774v1) | Jingyuan Zhou, Yuexuan Wang et al. | Ensuring scalable input-to-state stability (sISS) is critical for the safety and reliability of large-scale interconnected systems, especially in the presence of communication delays. While learning-based controllers can achieve strong empirical performance, their black-box nature makes it difficult to provide formal and scalable stability guarantees. To address this gap, we propose a framework to synthesize and verify neural vector Lyapunov-Razumikhin certificates for discrete-time delayed interconnected systems. Our contributions are three-fold. First, we establish a sufficient condition for discrete-time sISS via vector Lyapunov-Razumikhin functions, which enables certification for large-scale delayed interconnected systems. Second, we develop a scalable synthesis and verification framework that learns the neural certificates and verifies the certificates on reachability-constrained delay domains with scalability analysis. Third, we validate our approach on mixed-autonomy platoons, drone formations, and microgrids against multiple baselines, showing improved verification efficiency with competitive control performance. |
| 2026-04-01 | [When Safe Models Merge into Danger: Exploiting Latent Vulnerabilities in LLM Fusion](http://arxiv.org/abs/2604.00627v1) | Jiaqing Li, Zhibo Zhang et al. | Model merging has emerged as a powerful technique for combining specialized capabilities from multiple fine-tuned LLMs without additional training costs. However, the security implications of this widely-adopted practice remain critically underexplored. In this work, we reveal that model merging introduces a novel attack surface that can be systematically exploited to compromise safety alignment. We present TrojanMerge,, a framework that embeds latent malicious components into source models that remain individually benign but produce severely misaligned models when merged. Our key insight is formulating this attack as a constrained optimization problem: we construct perturbations that preserve source model safety through directional consistency constraints, maintain capabilities via Frobenius directional alignment constraints, yet combine during merging to form pre-computed attack vectors. Extensive experiments across 9 LLMs from 3 model families demonstrate that TrojanMerge, consistently achieves high harmful response rates in merged models while source models maintain safety scores comparable to unmodified versions. Our attack succeeds across diverse merging algorithms and remains effective under various hyperparameter configurations. These findings expose fundamental vulnerabilities in current model merging practices and highlight the urgent need for security-aware mechanisms. |

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



