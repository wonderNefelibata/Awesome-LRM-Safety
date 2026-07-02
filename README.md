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
| 2026-07-01 | [Distributed Containment of a Compromised Agent through Repulsive Cages](http://arxiv.org/abs/2607.01230v1) | Luigi Petruzziello, Camilla Fioravanti et al. | UAV swarms and cyber-physical multi-agent systems are increasingly deployed in safety-critical missions that require coordinated motion, distributed decision making, and autonomy. A major security risk arises when a legitimate agent is hijacked and driven by adversarial high-level commands. Rather than focusing on detection and isolation of malicious agents, we exploit a structural property common in autonomous platforms: low-level collision-avoidance modules are typically implemented as independent safety layers and may remain active even under high-level compromise. Building on this property, we propose a distributed containment framework that uses the compromised agent's uncompromised avoidance response as an indirect actuation channel. Defender agents select their geometric configuration to shape the repulsive field experienced by the target, with the goal of keeping it inside a prescribed admissible region and, when required, steering it toward a desired destination. The interaction is modeled as an online Stackelberg game in which defenders act as leaders and the adversary reacts by choosing the target command. Using support-function and normal-cone arguments, we derive an exact geometric characterization of robust one-step containment and introduce the notion of a repulsive cage. These results define a centralized Stackelberg oracle and motivate a fully distributed online approximation based on local communication and dynamic field estimation. We prove sublinear dynamic-regret bounds with respect to the centralized benchmark, quantifying the effect of network-induced estimation errors and temporal variability of the stage-wise optimum. Simulations validate the approach and corroborate the theory. |
| 2026-07-01 | [FastBridge: Closing the Model-Based Realization Gap in Safety Filters on 3D Gaussian Splatting for Fast Quadrotor Flight](http://arxiv.org/abs/2607.01200v1) | Tscholl Dario, Nakka Yashwanth Kumar et al. | Fast quadrotor flight requires safe obstacle avoidance under tight onboard compute limits. While 3D Gaussian Splatting (3DGS) provides a continuous, geometry-aware scene representation for perception-driven navigation, existing 3DGS safety filters use reduced-order models such as single- and double-integrators that ignore actuator limits and assume commanded accelerations are realized instantaneously. Building on an analytic collision cone barrier for 3DGS, we introduce a nonlinear, actuator-aware safety filter enforced through the full quadrotor dynamics. We derive a high-relative-degree collision cone exponential CBF and a backup CBF that preserves QP feasibility under input constraints using a forward-simulated backup policy. Compared with a state-of-the-art 3DGS safety filter, our approach reduces trajectory jerk by 47% and runs 2.25 times faster. We validate the method in simulation and on hardware for real-time navigation in cluttered, perception-derived environments. |
| 2026-07-01 | [TERA: A Unified Taylor Model Enabled Reachability Analysis Framework](http://arxiv.org/abs/2607.01189v1) | Salma Iraky, Andrew Sogokon | Reachability analysis of safety-critical systems requires computing rigorous enclosures of all possible state trajectories. Taylor Model (TM)-based methods have proved effective at mitigating the so-called wrapping effect which leads to overly conservative enclosures of reachable sets. However, existing tools are often hard to extend or focused on narrow system classes (e.g. deterministic systems modelled by ODEs, or hybrid systems). We develop TERA: a Python-native framework for TM-based reachability analysis of continuous, hybrid and stochastic systems within a single symbolic-numeric workflow. TERA is free and open-source, enabling rapid prototyping of reachability analysis techniques with rigorous enclosures. At present, our implementation is able to compute tight reachable set over-approximations for non-linear ODEs and hybrid systems on difficult benchmark problems, and already supports analysis of continuous-time stochastic systems. Our goal is to develop a robust open-source Python infrastructure for rigorous reachability analysis supporting a broad class of systems, including stochastic hybrid systems. |
| 2026-07-01 | [Adversarial Pragmatics for AI Safety Evaluation: A Benchmark for Instruction Conflict, Embedded Commands, and Policy Ambiguity](http://arxiv.org/abs/2607.01153v1) | Brett Reynolds | Safety evaluations for language models increasingly depend on judgments about ambiguous natural-language behaviour: whether a model has followed an instruction, refused appropriately, complied with a policy, resisted an embedded command, or misreported progress in an agentic task. Existing benchmarks often compress these distinctions into pass/fail labels, obscuring whether failures arise from capability limits, policy ambiguity, instruction conflict, scaffold failure, or unstable evaluator judgments.   This paper introduces adversarial pragmatics as a benchmark and annotation protocol for evaluating model behaviour under instruction conflict, embedded commands, quotation, scope ambiguity, deixis, indirect speech acts, and multi-turn agent transcripts. The contribution is empirical and methodological: a linguistically controlled taxonomy, an 18-item seed benchmark with validator-enforced metadata, a 54-row local seed pilot, an expert-evaluation protocol distinguishing task success, policy compliance, safety risk, refusal outcome, and evaluator confidence, and metrics for judge validity, diagnostic ambiguity, and taxonomy drift. The framework turns linguistic judgment methodology into a practical tool for validating safety evals, LLM judges, gold-set construction, prompt-injection tests, and safety documentation. |
| 2026-07-01 | [Antaeus: Hunting Repository-Level Logic Vulnerabilities via Context-Grounded LLM Reasoning](http://arxiv.org/abs/2607.01138v1) | Michele Armillotta, Nicolò Romandini et al. | LLM-based vulnerability detectors have shown promising results in identifying memory-safety bugs and vulnerability classes whose violations can often be expressed through established security properties. Logic vulnerabilities, however, pose a different challenge, as their identification requires inferring application-specific security invariants and implicit assumptions about intended behavior. Even frontier agentic models struggle because these invariants are often implicit and buried among unrelated code. Motivated by this gap, we present Antaeus, a framework for detecting logic vulnerabilities that grounds LLM reasoning in repository-level code context. Antaeus follows a repository-scale pipeline combining function prioritization, context-grounded reasoning, comparative validation, and structured reporting. It ranks functions using lightweight repo-wide security signals, directing costly LLM analysis toward relevant code and reducing calls, cost, and triage effort. For each prioritized function, Antaeus combines local code context with a repository-level view of the application's functionality, security resources, and trust boundaries. This enables reasoning about how the function is executed within the broader application rather than as an isolated snippet. Antaeus identifies security-sensitive sinks, derives safety conditions for safe execution, and checks whether they are locally satisfied. Candidate findings undergo comparative validation, pruning concerns that reflect project-wide norms rather than distinctive violations. Finally, Antaeus reports sinks, violated safety conditions, and evidence, making findings actionable and traceable. We evaluate Antaeus on 28 repositories with confirmed logic vulnerabilities and compare it against function-level and agentic models. Antaeus detects and explains 15 vulnerabilities, outperforming baselines with comparable token usage and cost. |
| 2026-07-01 | [Seahorse: A Unified Benchmarking Framework for Spatiotemporal Event Modeling](http://arxiv.org/abs/2607.01022v1) | Yahya Aalaila, Gerrit Großmann et al. | Spatiotemporal point processes (STPPs) model event data in continuous time and space, with applications in mobility, epidemiology, and public safety. Recent neural STPPs span expressive intensity models, conditional density models, continuous-time latent dynamics, normalizing-flow spatial decoders, and score-based generative mechanisms. Yet comparison remains fragile because implementations differ in preprocessing, coordinate normalization, splits, likelihood conventions, and evaluation protocols. We present SEAHORSE, a unified framework for reproducible STPP experimentation. SEAHORSE formalizes neural STPPs through a common encode-evolve-decode interface and trains, tunes, and evaluates every model family under a single executable benchmark protocol with raw-coordinate likelihood reporting. This enables fair comparisons but, more importantly, controlled diagnostic studies. We pair SEAHORSE with HawkesNest, a synthetic stress-test suite, and show that increasing event-pattern complexity exposes each family's inductive bias, degrading some models sharply and leaving others stable. Code: https://github.com/YahyaAalaila/seahorse. |
| 2026-07-01 | [Geometry-Aware Cross-Height Channel Knowledge Map Prediction for UAV-Assisted Communications With Uncertainty-Guided 3D Sensing](http://arxiv.org/abs/2607.00887v1) | Zhihan Zeng, Amir Hussain et al. | Low-altitude Unmanned Aerial Vehicles (UAVs) often need to infer channel knowledge across a range of heights from only sparse observations collected at a few altitude layers. To address this challenge, this paper studies height-conditioned cross-height channel knowledge map (CKM) prediction for UAV-assisted communications in geometry-rich urban environments. We develop a geometry-aware conditional prediction framework that combines urban scene priors, sparse multi-altitude observations, and target-height descriptors to reconstruct dense CKMs at unobserved target heights. An uncertainty head is further introduced to characterize prediction confidence and to support cost-aware online UAV sensing under motion and safety constraints. Experiments on a layered aerial CKM benchmark show that the proposed Feature Pyramid Network (FPN)-Transformer achieves the best overall performance under both unseen-scene zero-shot and legacy patch-random protocols, reducing the Root Mean Square Error (RMSE) to 5.347dB and 1.111dB, respectively, compared with 6.937dB and 1.221dB for the strongest baseline 3D-RadioDiff. Moreover, after applying our unseen-scene few-shot adaptation, the RMSE further decreases from 5.347dB in zero-shot prediction to 3.518dB with 10-shot two-height support, while the uncertainty-guided cost-aware sensing policy improves active reconstruction from 6.94dB at initialization to 4.79dB at sensing budget 40, outperforming uncertainty-only sensing at 5.08dB and random aerial sampling at 5.84dB. |
| 2026-07-01 | [Investigating Driver Behavior in Complex Traffic Situations While Driving Partially Automated Vehicles](http://arxiv.org/abs/2607.00855v1) | Lukas Köning, Nataša Miličić et al. | Traffic complexity critically influences driver task demands in partially automated vehicles, yet subjective perception and its behavioral indicators remain underexplored in real-world settings. This paper analyzes driver behavior - vehicle interaction, glance patterns, and guiding fixation - across varying levels of subjective traffic complexity, using real-world data from 20 drivers in real urban traffic. Traffic complexity was determined by expert labeling and served as ground truth for vehicle data. Statistical analysis of 16 driver behavior metrics revealed small but significant trends with increasing complexity: deviation from speed limit increased, brake rate increased while braking intensity decreased, horizontal gaze dispersion and entropy widened, and guiding fixation rate decreased, indicating defensive adaptation and perceptual shifts. Contributions include real-world validation of gaze metrics and guiding fixation under subjective complexity, novel insights from gaze and guiding fixation entropy metrics, and the identification of promising indicators~(driven speed, brake rate, gaze yaw entropy, guiding fixation rate) for complexity-adaptive partially automated vehicles. While based on a limited urban sample and expert-labeled subjective complexity, the findings provide a foundation for combined complexity scores and their integration into complexity-adaptive, partially automated vehicles, boosting human-like automation and enhancing safety and predictability in the traffic system. |
| 2026-07-01 | [Exploring the Semantic Gap in Agentic Data Systems: A Formative Study of Operationalization Failures in Analytical Workflows](http://arxiv.org/abs/2607.00828v1) | Jalal Mahmud, Eser Kandogan | Large language models (LLMs) are increasingly used to generate queries, invoke tools, and construct analytical workflows. Although recent advances have substantially improved workflow generation and execution, the semantic information required to operationalize analytical concepts often lies beyond what is explicitly represented in database schemas and data values. We present a cross-domain formative study of operationalization failures in agent-generated analytical workflows. Across 236 analytical intents spanning finance, human resources, and public safety domains, we identify 153 recurring failures despite successful workflow generation and execution. Our analysis reveals five recurring classes of failures: comparative grounding, process reasoning, quantitative reasoning, role confusion, and policy grounding. These findings suggest a semantic gap between user-level analytical concepts and the information available to workflow-generation systems. More broadly, they raise questions about the admissibility of analytical operations and suggest that future agentic data systems may require richer semantic representations to bridge the gap between analytical intent and executable computation. |
| 2026-07-01 | [From Prediction Uncertainty to Conformalized Distance Fields for Safe Motion Planning](http://arxiv.org/abs/2607.00776v1) | Jaeuk Shin, Yoonseok Ra et al. | Safe motion planning in dynamic environments requires reasoning about the uncertainty in predicted obstacle motion without sacrificing real-time performance. Existing conformal approaches conformalize a scalar score that aggregates per-obstacle prediction errors, losing spatial coherence and scaling poorly with scene density. We instead conformalize the entire predicted distance field at once. This functional conformal prediction (FCP) framework yields a distribution-free, field-level lower bound, from which safety follows uniformly: any trajectory satisfying the resulting constraint is certified safe, independent of how the control space is sampled. The key enabler is that the residual distance field is empirically low-rank and approximately time-invariant, which makes the bound decomposable in coefficient space. An envelope is fitted offline via functional PCA and a Gaussian-mixture inductive conformal procedure, then refined online by a lightweight adaptive functional conformal (AFCP) update on a low-dimensional vector. This keeps the per-step cost largely insensitive to obstacle count and retains long-run field coverage under distribution shift. We embed the envelope as a tightened safety constraint in a sampling-based model predictive controller, FCP-MPC. On the ETH--UCY pedestrian benchmarks and a dense 3D quadrotor task with up to 280 dynamic obstacles, FCP-MPC attains a favorable balance of safety, feasibility, and efficiency, reaching goals where pointwise and egocentric conformal baselines become too conservative or too expensive, while keeping per-step computation far below online uncertainty-reasoning baselines. |
| 2026-07-01 | [A Data-Enabled Primal-Dual Approach for Policy Learning with SDP Formulations](http://arxiv.org/abs/2607.00644v1) | Han Wang, Feiran Zhao et al. | This paper develops a data-enabled primal-dual framework for learning optimal control policies for unknown linear discrete-time systems from online data. The proposed approach views the data-dependent control synthesis problem as a time-varying semidefinite program (SDP) whose coefficients are recursively updated from online closed-loop measurements. Instead of repeatedly solving a full SDP as new data arrive, the policy is updated online through lightweight primal-dual iterations, each consisting of a linear equation solve and a projection onto the positive semidefinite cone. The framework applies to both direct and indirect data-driven formulations and covers a broad class of control objectives, including LQR, $H_\infty$ control, and safety-critical control. To characterize the coupling between online optimization and closed-loop data generation, we introduce two data-dependent quantities: the Sim-to-Real Gap, which measures the mismatch between noisy and noiseless data-induced SDPs, and the Difference-of-Signal, which measures the temporal variation of the SDP coefficients. Under persistency of excitation, suitable SDP regularity conditions, and sufficiently slow data variation, we establish a local linear tracking result up to residual terms governed by the latter two quantities. A global ergodic convergence bound is also derived for arbitrary initialization. Numerical examples on LQR, $H_\infty$ control, and safe exploration demonstrate that the proposed method can efficiently improve control performance from online data while accommodating SDP constraints beyond the well-explored LQR policy-gradient formulations. |
| 2026-07-01 | [Safe Alone, Unsafe Together: Safeguarding Against Implicit Toxicity When Benign Images Combine](http://arxiv.org/abs/2607.00576v1) | Jiaxian Lv, Shiyao Cui et al. | Multi-image content has become an increasingly prevalent form of visual communication in social media, giving rise to a new safety issue, multi-image implicit toxicity (MIIT), where each image appears benign in isolation, but harmful semantics emerge when the images are interpreted jointly. MIIT is particularly challenging for existing commercial moderation APIs and models due to the lack of explicit risky cues in each image. This paper aims to study how to identify MIIT. We first provide a formal definition of MIIT and analyze three key challenges for its detection. To alleviate the scarcity of data in this area, we construct MIIT-dataset, an image-only multi-image safety dataset covering seven representative risk categories through an automatic generation pipeline. Finally, we train MiShield with progressively distilled reasoning supervision, enabling it to produce safety judgments accompanied by explicit analyses of the correlated entities that result in the hazards. Experiments show that MiShield-8B models outperform representative moderation services and even larger-scale models, revealing its effectiveness and practical value for this widely used visual format. Warning: This paper contains potentially sensitive content. |
| 2026-07-01 | [HARC: Coupling Harmfulness and Refusal Directions for Robust Safety Alignment](http://arxiv.org/abs/2607.00572v1) | Shei Pern Chua, Fangzhao Wu | Understanding how aligned LLMs internally represent safety is critical for diagnosing alignment vulnerabilities, as it explains why jailbreaks succeed and informs the design of robust alignment strategies. Prior work shows that aligned LLMs encode harmfulness and refusal as separable directions in the residual stream at prompt-side token positions. We show that jailbreaks succeed at prompt encoding by suppressing either the refusal or harmfulness direction before any token is generated, with distinct attack classes occupying separable regions of the harmfulness-refusal plane. Extending the analysis to response-token positions, we find that the model recognizes harmful content while it is generating that content, even when it failed to recognize the input as harmful at the prompt side. Motivated by our findings, we introduce HARC (Harmfulness-And-Refusal Coupling), a fine-tuning method that pairs the two directions across both prompt and response positions. Since the intervention is confined to the harmfulness-refusal subspace, it leaves the rest of the residual stream intact and does not degrade general capability or inflate over-refusal. Across extensive experiments, HARC achieves the strongest robustness-capability-usability trade-off among six baselines spanning the major training-time and inference-time safety methods. The harmfulness and refusal directions at prompt and response positions transfer across the five model families and two scales we tested without architecture-specific tuning. |
| 2026-07-01 | [ECoSim: Data Efficient Fine-Tuning for Controllable Traffic Simulation](http://arxiv.org/abs/2607.00545v1) | Yu-Hsiang Chen, Wei-Jer Chang et al. | Controllable traffic simulation is critical for testing autonomous driving systems, yet existing approaches often require retraining large generative models with extensive annotated data. We introduce a lightweight control adaptation framework that enables multi-modal controllability (sketch, latent behavior codes, and text) for pretrained state-of-the-art diffusion and autoregressive traffic models. By modulating intermediate features through identity-initialized FiLM layers, our method efficiently adds new control modalities while preserving the base model's generative prior. Evaluated on Waymo Open Sim Agents Challenge, our approach demonstrates strong controllability with less than 1% of the paired control data. Through context-aware condition transfer, our framework enables counterfactual scenario generation and long-tail synthesis while maintaining stable closed-loop driving realism and safety. Our framework unlocks new possibilities for controllable traffic simulation, enabling targeted scenario generation through lightweight adaptation of pretrained generative models. Project page: https://ecosim-web.github.io/ |
| 2026-07-01 | [Learning from Demonstration via Spatiotemporal Tubes for Unknown Euler-Lagrange Systems](http://arxiv.org/abs/2607.00534v1) | Ratnangshu Das, Puneeth Shankar et al. | We present STT-LfD, a unified Learning from Demonstration (LfD) framework that integrates motion learning with control for unknown Euler-Lagrange systems. Unlike traditional decoupled approaches that track a fixed reference, the proposed method treats demonstrations as a data-driven safety specification. Using heteroscedastic Gaussian Processes, STT-LfD learns Spatiotemporal Tubes (STTs) as an intent envelope that capture time-varying precision requirements of a task. A closed-form feedback controller then enforces these learned constraints while respecting actuator limits, without requiring explicit system identification. The approach preserves the temporal structure of demonstrations, remains computationally efficient, and avoids explicit system identification. Hardware experiments on a mobile robot and a 7-DOF manipulator show that it outperforms baselines in robustness to disturbances and computational speed. |
| 2026-07-01 | [AI Native Games: A Survey and Roadmap](http://arxiv.org/abs/2607.00527v1) | Zhiyue Xu, Fandi Meng et al. | Generative AI now enables games to produce dialogue, quests, characters, images, and worlds at runtime. Yet generation alone does not make a game AI-native, nor does it guarantee playability. This paper defines AI-native games by whether runtime generative AI is constitutive of the core loop: if the AI component were removed or trivially replaced, the central form of play would collapse or become fundamentally different. This counterfactual criterion separates AI-native games from AI-augmented games, boundary artifacts, chatbots, tavern-style role-play, procedural content generation, and AI-assisted production. Using this definition, we screen candidate artifacts and analyze 53 publicly available AI-native games and prototypes. We introduce a dual-axis G/N taxonomy: the G-axis captures player-facing game type, while the N-axis captures the dominant AI mechanic that makes generative AI indispensable to play. The corpus is concentrated around language-forward designs, especially narrative adventure, epistemic interaction, and generative narrative, while categories such as semantic adjudication, multi-agent simulation, generative construction, and relationship/companion play remain less represented. We argue that the central design problem is organizing semantic openness into stable gameplay. AI-native design depends on mechanical invariants: goals, rules, state, feedback, pacing, and player agency that make open-ended AI outputs interpretable and consequential. We conclude with a roadmap for controllable generation, AI-as-mechanic design, multimodal and multi-agent systems, inference economics, evaluation, safety, and regulation. |
| 2026-07-01 | [Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces](http://arxiv.org/abs/2607.00481v1) | Junlong Liu, Haobo Wang et al. | Jailbreak attacks remain a critical threat to the safe deployment of large language models (LLMs). While prior work has primarily studied attacks and defenses at the prompt level, we show that this prompt-centric paradigm overlooks a structural vulnerability in stateful, function-calling environments. In such applications, developer-defined schemas, structured arguments, and untrusted tool outputs are interleaved into a single shared model context. This architecture expands the attack surface by blurring the boundary between trusted control logic and untrusted data, allowing adversarial intent to be distributed across a multi-turn execution path. We exploit this architectural flaw through SMT, a black-box attack framework based on Simulated Moderation Traces. Departing from purely prompt-based interactions, SMT constructs a multi-turn trajectory that simulates a legitimate moderation-auditing workflow. Within this trajectory, a fabricated moderation frame leverages red-team testing as a pretext to elicit harmful generations. The subsequent validation feedback treats safety refusals as execution failures, prompting refinements that gradually weaken the model's safety constraints and ultimately trigger harmful outputs. Extensive empirical evaluations on prominent commercial LLMs from five different providers across two standardized safety benchmarks show that SMT consistently achieves the highest average attack success rate and HarmScore while requiring a near-minimal number of queries, substantially outperforming existing baselines. These findings demonstrate that prompt-level sanitization alone is fundamentally insufficient for defending tool-enabled LLM systems and highlight the urgent need for context-aware validation across schemas, arguments, tool outputs, and accumulated conversation state. The code is available at https://github.com/liujlong27/SMT. |
| 2026-07-01 | [MolSafeEval: A Benchmark for Uncovering Safety Risks in AI-Generated Molecules](http://arxiv.org/abs/2607.00464v1) | Tong Xu, Xinzhe Cao et al. | Current molecular generation benchmarks emphasize task complexity, molecule novelty, and property alignment; they largely overlook a critical concern: the potential safety risks of AI-generated molecules. In practice, many generative models may produce molecules with toxic, reactive, or otherwise hazardous characteristics - posing hidden dangers that remain insufficiently addressed. To address this gap, we introduce MolSafeEval, a benchmark dedicated to evaluating and analyzing the safety risks of molecular generation. Unlike prior approaches that rely on narrow toxicity predictors, MolSafeEval integrates heterogeneous safety knowledge - ranging from toxicological databases to hazard rules - into a structured molecular safety knowledge graph. This graph serves as a foundation for large language model-based reasoning, enabling systematic detection and explanation of unsafe features in generated compounds. We further categorize molecular generative models into four representative task types - unconditional generation, property optimization, target protein-based design, and text-based generation - and provide standardized datasets and safety evaluation protocols for each. By systematically revealing the safety vulnerabilities of current generative approaches, MolSafeEval offers a new lens for benchmarking molecular models and provides essential guidance toward safer, more trustworthy molecular design. |
| 2026-07-01 | [Learning Gait-Aware Quadruped Locomotion with Temporal Logic Specifications](http://arxiv.org/abs/2607.00442v1) | Merve Atasever, Cagan Bakirci et al. | Reinforcement learning (RL) for quadruped locomotion commonly depends on fixed, hand-crafted, and Markovian reward functions that limit both interpretability of learned policies and lack explicit control over gait behaviors. We introduce a framework where distinct gaits are specified using parameterized constraints expressed in Signal Temporal Logic (STL). These include safety bounds, gait synchronization constraints, command tracking, and actuation bounds. From these specifications, we develop a reward shaping mechanism that provides learning agents a dense, continuous reward landscape that encodes desired behavior. We define parametric STL templates for three speed regimes (walking-trot, trot, bound), calibrate their parameters from reference rollouts, and compute rewards from using smooth approximations of STL robustness over the rollouts. The generated rewards can be used to provide shaped gradients compatible with Proximal Policy Optimization (PPO). We instantiate the approach on Google's Barkour quadruped robot in MuJoCo XLA (MJX). We use parallelization within the simulator to improve training speeds and use domain randomization to robustify learned policies. We show that compared to a baseline of hand-crafted rewards, the STL-shaped rewards yield tighter velocity tracking and more stable training. Videos can be found on our project website: https://stl-locomotion.github.io/. |
| 2026-07-01 | [Robust Operational Space Control with Conformal Disturbance Bounds for Safe Redundant Manipulation](http://arxiv.org/abs/2607.00424v1) | Wenhua Liu, Fan Zhang et al. | Redundant robotic manipulators operating in constrained and human-interactive environments require accurate task-space tracking together with rigorous safety guarantees under dynamic uncertainties. Classical operational space computed torque controller (OSCTC) relies on accurate dynamic models and degrades in the presence of disturbances. In contrast, the data-driven paradigm of residual learning approximates disturbances as functions learned from full-state measurements, which are often noisy in practice, lack rigorous theoretical guarantees, and introduce additional design complexity. This paper proposes a robust OSCTC framework that integrates an extended state observer (ESO) with conformal prediction to combine model-based robustness and data-driven adaptability. The ESO estimates lumped disturbances directly in operational space without requiring full-state measurements as in residual learning, and a robust control barrier function (CBF) is constructed to enforce safety under uncertainty. However, robust CBFs require a known disturbance-variation bound to guarantee absolute safety, which often leads to conservatism in practice. To address this limitation, we further employ a sliding-window conformal prediction mechanism to estimate the bound online in a distribution-free manner, thereby achieving practical probabilistic safety guarantees. Experiments on a 7-DoF Franka Research 3 manipulator demonstrate millimeter-level tracking accuracy and real-time safe control at 1~kHz under various disturbances. |

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



