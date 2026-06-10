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
| 2026-06-09 | [ReasonAlloc: Hierarchical Decoding-Time KV Cache Budget Allocation for Reasoning Models](http://arxiv.org/abs/2606.11164v1) | Wenhao Liu, Hao Shi et al. | Long chain-of-thought (CoT) trajectories in large language model (LLM) reasoning cause severe inference bottlenecks due to rapid key-value (KV) cache growth. Current decoding-time compression methods mitigate this issue via token eviction, but typically assume a uniform budget distribution across all layers and heads. In contrast, existing non-uniform budget allocation methods are predominantly designed for the static prompt prefill phase, and they do not capture the stepwise context demands of autoregressive reasoning. To bridge this gap, we propose ReasonAlloc, a training-free framework that recasts decoding-time KV compression as a hierarchical budget allocation problem. ReasonAlloc operates at two complementary levels: an offline layer-wise preallocation strategy captures an architecture-driven demand pattern which we call ``\textit{Reasoning Wave}'', while an online head-wise strategy reallocates resources during decoding to information-rich heads based on real-time utility. Evaluations on mathematical reasoning benchmarks (MATH-500, AIME~2024) using DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B, and AceReason-14B show that ReasonAlloc outperforms uniform-budget R-KV, SnapKV, and Pyramid-RKV (a baseline enforcing a static, monotonically decreasing layer budget), with the largest gains at small budgets (128-512 tokens). ReasonAlloc is plug-and-play with existing token-eviction policies and introduces negligible inference-time overhead. |
| 2026-06-09 | [Toward a Full-Stack Framework for Industrial Augmented Reality: Benefits, Risks, and Design Considerations for Dependable Deployment in Manufacturing](http://arxiv.org/abs/2606.11112v1) | Narges Chinichian, Maximilian Anton Palm | Industrial Augmented Reality (AR) has progressed from laboratory demonstrations to operational pilots across design, training, assembly, maintenance and quality assurance, yet broad, dependable deployment in manufacturing remains the exception. We synthesise existing evidence into a full-stack deployment framework structured along six distinct but coupled decision axes: (i) value and benefits, (ii) technical and integration constraints, (iii) human factors and safety, (iv) organisational and economic considerations, (v) data, security and privacy, and (vi) governance, ethics and long-term risk. Within each axis we separate (a)benefits, (b)failure modes and (c)design considerations, and cross-link them through a deployment checklist that engineering managers and vendors can apply when scoping projects. The contribution is conceptual and practice-oriented: a synthesis grounded in the literature and public deployment reports. We mark where the evidence base is mature (e.g. assembly task time, training efficacy), emerging (e.g. cognitive workload trade-offs, cobot safety zones), or speculative (e.g. metaverse-scale governance), and identify open questions whose resolution conditions the transition from demos to dependable infrastructure. |
| 2026-06-09 | [EM-Fall: Embodied mmWave Sensing for Day-and-Night Fall Detection on Humanoid Robots](http://arxiv.org/abs/2606.11109v1) | Yanshuo Lu, Yuxuan Hu et al. | Falls are one of the leading causes of injury and hospitalization among elderly individuals, making reliable fall awareness an essential capability for safety monitoring in residential environments. However, existing fall detection systems often rely on wearable devices or fixed sensing installations, which may suffer from low user compliance, limited spatial coverage, or degraded performance under occlusion and poor lighting conditions. In this work, we propose \textbf{EM-Fall}, an embodied fall detection framework deployed on a mobile humanoid robot. The system integrates millimeter-wave (mmWave) sensing with robotic mobility, allowing the robot to actively adjust its sensing viewpoint and maintain target observability across rooms and under occlusion. To address interference in complex residential environments, including pet motion and multipath artifacts, we design a human-centered perception pipeline combined with lightweight temporal modeling to capture motion evolution before, during, and after fall events. We evaluate the proposed system across eight real indoor environments with four participants and construct an in-home mmWave fall detection dataset. Experimental results show that the embodied mobile sensing paradigm improves monitoring continuity and maintains robust fall detection performance under diverse environmental conditions. The proposed framework provides a practical solution for robot-assisted safety monitoring in home environments. |
| 2026-06-09 | [The Shibboleth Effect: Auditing the Cross-Lingual Distributional Skew of Large Language Models](http://arxiv.org/abs/2606.11082v1) | Hakan Mehmetcik | This study investigates cross-lingual distributional skew (the Shibboleth Effect) in frontier large language models (LLMs) subjected to sustained adversarial conditions. We develop a multi-agent geopolitical wargame, the Cerulean Sea Crisis, a synthetic maritime territorial dispute designed to mirror the structural dynamics of Eastern Mediterranean conflicts. Six frontier models (GPT-4o, Llama-4, Mistral-Large, Gemini-3.1-Pro, Qwen3.6-Plus, and DeepSeek-R1) participate in a between-groups experiment (N = 10 games per arm, K = 5 rounds per game) in which the sole manipulation is the language of play (English versus Turkish), producing 586 validated statements. A zero-shot classifier assesses behavioral dispositions along two continuous dimensions: Concession Rate and Coercive Rhetoric. The results are heterogeneous. Llama-4 shows a substantial, Holm-corrected increase in coercive rhetoric under Turkish (delta = +0.800, p = .002), whereas Gemini-3.1-Pro displays an equally large decrease (delta = -0.750, p = .005). DeepSeek-R1 exhibits a similar negative shift (delta = -0.860, p = .006) and provides chain-of-thought evidence consistent with a buffering mechanism. GPT-4o shows no detectable effect (delta = +0.130, p = .614). These findings indicate that cross-lingual behavioral skew is contingent on model architecture and training regime rather than a universal property of Western-origin LLMs. We identify two distinct buffering mechanisms, chain-of-thought institutional anchoring and multilingual RLHF alignment, and discuss their implications for integrating LLMs safely into diplomatic and crisis-management settings. |
| 2026-06-09 | [Does Reasoning Preserve Alignment? On the Trustworthiness of Large Reasoning Models](http://arxiv.org/abs/2606.11046v1) | Prajakta Kini, Avinash Reddy et al. | Instruction-tuned LLMs are increasingly converted into reasoning models through post-training to improve multi-step task performance. This conversion is usually optimized for reasoning accuracy, without explicitly preserving the alignment behavior of the instruction-tuned model, such as safe refusal, bias avoidance, and privacy protection. We ask: does this conversion preserve alignment? We study this question through a trustworthiness audit and find that it is not behavior-preserving by default. For a systematic analysis, we compare reasoning models produced via supervised fine-tuning, RL-based post-training, and distillation against matched instruction-tuned baselines across six trustworthiness dimensions: safety, toxicity, stereotyping and bias, machine ethics, privacy, and out-of-distribution robustness. We observe that reasoning models often improve on reasoning benchmarks but exhibit alignment regressions, including increased toxicity, amplified stereotyping, miscalibrated refusal, and contextual privacy leakage. These regressions are consistent with behavioral drift from the instruction-tuned baseline, measured by KL divergence. Overall, our results point to the broader conclusion that trustworthiness metrics are essential for evaluating reasoning models and should be reported alongside gains in reasoning capability. |
| 2026-06-09 | [Making a Name for Myself: On Academic Naming Policies and their Impact](http://arxiv.org/abs/2606.11021v1) | A Pranav, Vagrant Gautam et al. | In academic publishing, names connect scholars to their work. When scholars change their names, including for marriage, academic recognition, or gender transition, they may lose credit for past publications.However, despite significant impacts on citation accuracy and researcher well-being, no existing studies examine how naming policies in computer science serve researchers who change their names.We use a mixed-methods approach combining surveys, interviews, and large-scale citation analysis of papers from eight major computer science venues from 2019-2025. We document the multi-year advocacy effort that established the first name change policies, identify implementation barriers including incomplete publisher updates and months-long processing delays. Researchers continue being cited with misparsed and incorrect names despite publisher updates. When these citation errors happen, interviewees report significant mental health impacts, including stress, anxiety, and safety risks. Empirically, we find that venues with accessible and visible name change policies have significantly fewer citation errors compared to inaccessible policies (899 vs. 996 errors per 1,000 papers). Our annotation analysis shows that deadnaming of transgender researchers in citations decreased by 92% from 2019 to 2024. Our findings demonstrate the importance of inclusive publishing policies, for which name change policy advocacy led by trans researchers has been a significant driver. We recommend that venues adopt proactive visible name change policies, support queer advocacy groups, and improve publication infrastructure to build an inclusive publishing landscape. |
| 2026-06-09 | [Diffusion Forcing Planner: History-Annealed Planning with Time-Dependent Guidance for Autonomous Driving](http://arxiv.org/abs/2606.11019v1) | Zehan Zhang, Neng Zhang et al. | Learning-based motion planners, despite recent progress, often suffer from temporal inconsistency. Small perturbations across frames can accumulate into unstable trajectories, degrading comfort and safety in closed-loop driving. Several methods attempt to inject history as a static conditioning signal to stabilize outputs, only to induce the planner to copy historical patterns instead of adapting to environment contexts. To address this limitation, we propose Diffusion Forcing Planner (DFP), a diffusion-based planning framework driven by history-guided control. Specifically, DFP decomposes the full trajectory into history, current and future segments, and assign independent noise levels to each segment. The model jointly denoises the historical and the future segments, enforcing a heterogeneous joint diffusion process. At inference, classifier-free guidance (CFG) is applied to steer future sampling using annealed history in a controllable manner. Closed-loop evaluation and comprehensive ablations on nuPlan show that DFP achieves competitive performance while producing continuous, stable, and controllable motion plans in complex driving scenarios. |
| 2026-06-09 | [A Companion App for an Autonomous Family Vehicle: Identification of Values for an Autonomous Mobility System](http://arxiv.org/abs/2606.10997v1) | Leon Johann Brettin, Tobias Schräder et al. | In this paper, we present a companion app for an autonomous vehicle aimed at user groups who would normally require an accompanying person to drive them. Two aspects of a companion app are presented in this paper: First, the possibility for a trusted person to track the ride of the person in need of support and second, to put the settings of the vehicle for persons in need of support in the hands of a trusted person. In addition, this article describes the requirements and addressed values and discusses the safety-relevant aspects of such a companion app. We also discuss and identify the values that influence passengers and trusted persons using the companion app. Overall, a companion app can provide new perspectives and opportunities for people in need of support, allowing them to take advantage of the features offered by autonomous vehicles. It enables trusted individuals to configure the vehicle according to the passengers needs. Also such an app can be a mechanism to involve trusted persons in the options given by the vehicle and give them the possibility to adapt the vehicle to the needs of the person in need of support. |
| 2026-06-09 | [FairWave : A Fairness-Aware Asynchronous DAG-BFT Consensus](http://arxiv.org/abs/2606.10982v1) | Syariful Mujaddiq | Combining asynchronous Byzantine Fault Tolerant (BFT) consensus with Proof-of-Stake (PoS) creates a trilemma between Sybil resistance, reward distribution fairness, and protection against persistent plutocracy. Existing DAG-BFT approaches (Narwhal+Tusk, Bullshark, and Mysticeti) prioritize liveness over the fairness implications of stake-based selection, resulting in persistent longitudinal centralization.FairWave is a dual-channel DAG BFT protocol that separates anchor selection from reward distribution. The selection channel is super-linear in stake, guaranteeing Sybil gain < 1 for all split factors K > 1. The reward channel is sub-linear, using square-root stake normalization to mitigate rich-get-richer dynamics.The finalized DAG structure provides deterministic uptime and latency factors, allowing honest validators to agree on operational quality without any external oracle. To avoid circular dependency between selection outcomes and selection weights, reputation is used in a lagged form: the active value at epoch e equals the prior epoch's final value. We derive closed-form constraints for both channels and validate them through nine empirical analyses (approximately 550,000 Monte Carlo rounds) against eight baselines. FairWave achieves a Gini coefficient of 0.149 (vs. Pure-PoS's 0.488), a monotone HHI reduction from 0.039 to 0.021 over 50,000 epochs, an optimal-adversary Sybil split of K* = 1, and a success-rate coefficient of variation of 5.2% under +/-25% input perturbation. Safety (agreement and validity) is a formal consequence of the 2f+1 strong-support commit rule, holding unconditionally for f < n/3; the empirical differential is the monotone-continuous liveness-degradation curve, which decreases from 99.6% commit rate at b=0.20 to 71.1% at the theoretical bound b=1/3 without the discontinuous cliff characteristic of view-change-driven leader-BFT. |
| 2026-06-09 | [Comparative Analysis of Inference-Time Defense Methods for Multimodal Large Language Models](http://arxiv.org/abs/2606.10904v1) | Bulat Nutfullin, Vladimir Evgrafov et al. | Multimodal large language models (MLLMs) now appear in safety-critical applications, but the visual channel leaves them open to adversarial attacks that predominantly text-oriented safety alignment addresses only in part. Retraining a model for each new vulnerability class is usually too expensive to be practical. We report a comparative empirical evaluation of three inference-time defense methods and their combinations, run on eight models from the InternVL and Qwen-VL families across seven safety benchmarks that span four attack classes and total 9,000 evaluation samples. Every figure below comes from the same unified proxy classifier. Five findings emerge from the evaluation. First, within the evaluated models and benchmarks, no single defense dominates across all settings: what works depends on the model's baseline safety and on the attack type. Second, combining defenses directly drives benign-query over-refusal to 97-100% across all eight evaluated models, and SmoothVLM on its own reaches 99.2-100%. Third, a simple safety prompt keeps utility largely intact (0.0-18.2% over-refusal across all eight models, five of them below 7%, although two exceeded 15%) while still yielding moderate safety gains. Fourth, different attack classes expose different weaknesses across the evaluated setup, which is why multi-benchmark evaluation matters. Fifth, in a preliminary whitebox test on two models (n=20), text-level defenses suppressed a PGD visual attack that had succeeded without any defense: the defenses act at the output stage, where gradient optimization has limited direct leverage in the tested configuration. Read together, these results argue for adaptive defense selection rather than a single fixed defense configuration. |
| 2026-06-09 | [AgniNav: Configuration-Driven Cross-Embodiment Local Planning for Robot Navigation](http://arxiv.org/abs/2606.10903v1) | Tianhao Zang, Siwei Cheng et al. | Monocular local navigation is attractive for lightweight robots, but existing vision-based policies often couple perception to a specific body, camera height, and footprint, making transfer from wheeled bases to legged platforms dependent on retraining or active depth hardware. This paper introduces AgniNav, a configuration-driven local navigation framework that standardizes cross-embodiment transfer at the collision-envelope level. Each robot is specified by a measurable four-parameter safety envelope: collision-relevant height, front length, rear length, and half width. The height parameter conditions an image-to-scan network to predict a one-dimensional, collision-relevant pseudo-laserscan from a monocular color image, while the remaining footprint parameters configure a dimension-aware local planner for collision checking. Training uses height-conditioned column-minimum scan labels generated from paired color-depth data, allowing the same image to supervise different safety envelopes without collecting robot-specific data. To the best of our knowledge, AgniNav is the first monocular local-navigation framework that jointly conditions perception and planning on a shared collision-envelope configuration for zero-retraining deployment across wheeled, quadruped, and humanoid platforms. Real-robot experiments on a Turtlebot2, Unitree Go2, and Accelerated Evolution K1 achieve 39/40, 18/20, and 18/20 successes with 0/40, 1/20, and 2/20 collisions, respectively, while running at 30 Hz on Jetson Orin. |
| 2026-06-09 | [IMPACT: Learning Internal-Model Predictive Control for Forceful Robotic Manipulation](http://arxiv.org/abs/2606.10818v1) | Jiawei Gao, Chaoqi Liu et al. | Real-world robotic manipulation tasks often involve forceful interactions with the environment, such as using tools of varying weights, transporting objects with different masses, and performing contact-rich tasks like table wiping. Previous learning-based approaches typically employ imitation learning policies that output target end-effector poses tracked by low-level impedance controllers. In these systems, forceful interactions are either implicitly realized through steady-state tracking errors or explicitly commanded using wrist force/torque or tactile sensors. However, implicit approaches generalize poorly across object weights, while explicit approaches require specialized hardware and increase system complexity. In this work, we propose IMPACT, a framework that decouples these forceful tasks into task-planning and internal-model-based predictive control. Extensive simulation and real-world experiments demonstrate that the proposed framework achieves higher success rates and improved generalization to unseen object weights, as well as better safety and energy efficiency. |
| 2026-06-09 | [N-GRPO: Embedding-Level Neighbor Mixing for Enhanced Policy Optimization](http://arxiv.org/abs/2606.10768v1) | Xukun Zhu, Hang Yu et al. | The success of Large Language Models in mathematical reasoning relies heavily on the generation of diverse and valid solution paths during the rollout phase. However, current rollout techniques face a fundamental trade-off: token-level sampling often yields redundant trajectories that differ only in rephrasing, while embedding-level methods utilizing random noise frequently disrupt semantic consistency. To resolve this, we introduce N-GRPO, a novel exploration strategy integrated into the Group Relative Policy Optimization (GRPO) framework. Rather than relying on token-level sampling or native embedding-level noise, our approach leverages Semantic Neighbor Mixing. This mechanism dynamically constructs input representations by mixing the embeddings of an anchor token and its nearest semantic neighbors, thereby injecting diversity while strictly adhering to the local semantic manifold. Experimental evaluations on the DeepSeek-R1-Distill-Qwen models across different sizes show that N-GRPO not only achieves consistent improvements over strong baselines on math reasoning benchmarks but also exhibits robust generalization capabilities on out-of-distribution tasks. |
| 2026-06-09 | [When the Chain of Thought Knows Better: Failure Modes in Multi-Turn Reasoning Models](http://arxiv.org/abs/2606.10740v1) | Sai Kartheek Reddy Kasu, Nils Lukas et al. | Failures in multi-turn reasoning models are largely invisible to terminal-score evaluation. A model can lock onto an unsafe stance early in a long dialogue, yet its final-turn refusal rate may appear indistinguishable from a robustly aligned baseline. To expose these hidden temporal dynamics, we propose a trace-level diagnostic - the CoT-Output 2x2 safety matrix. This framework labels every turn along two independent axes (internal reasoning and visible output), yielding four operationally defined failure cells: robust alignment, alignment faking, overt jailbreak, and a distinct failure mode we term context-injection failure (where the CoT maintains safe reasoning, but the visible output produces harm, highlighting a multi-turn manifestation of reasoning unfaithfulness). We evaluate three distilled reasoning targets against a fixed attacker across five oversight conditions, collecting 6750 turn-level observations on the Information-Hazard scenario. Our analysis reveals two reproducible vulnerabilities: an oversight paradox where explicit monitoring cues paradoxically increase alignment-faking rates rather than suppress them, and a context-injection failure where models lock onto unsafe external outputs despite safe internal states. We release the full dataset of multi-turn dialogues and CoT traces to support follow-up trace-diagnostic research. |
| 2026-06-09 | [A Pigouvian Matchmaker Mechanism for De-escalating the AGI Race](http://arxiv.org/abs/2606.10720v1) | Eduard Kapelko | A formal mechanism is presented in which a willing regulator-matchmaker fosters cooperation on resources among participants in the AGI race, collects a Pigouvian tax based on the speed-up it induces, and invests the proceeds into alignment research. The construction is derived in the continuous-time options framework of Tan (2025) in which cooperation is treated as a jump in the underlying asset value of participating players, the Pigouvian component is matched to the marginal effect of increasing expected loss, and the total collected fund endogenizes the rate of learning on safety. It is shown how the framework allows for determining participation and optimal activity levels.   Conditions under which it is optimal to enter the market are derived, and it is proven that if the orthogonality condition holds between the supported portfolio and the abilities component, the Suicide Region collapses at finite time, and the upper bound for this time is derived as sum of a deterministic and random term. Finally, if orthogonality is violated, it is proven that enhancing matchmaker capacity does not recover the market's superiority. The construction links research areas including two-sided markets, Pigouvian taxes, self-regulatory organizations, private law enforcement, evolutionary modeling of AI races, real options and option games, measurement of comparative progress and analysis of the Suicide Region. |
| 2026-06-09 | [Exploring and Complementing End Users' Requirements in IoT enabled System](http://arxiv.org/abs/2606.10598v1) | Haotian Li, Xiaohong Chen et al. | End users create IoT automation rules via trigger action programming, but their expressions are often fragmented, capturing device operations rather than high level intents. This gap leads to missing conditions, logical conflicts, and overlooked safety constraints, risking hazardous behaviors. To address this, we propose an intent driven requirements completion approach that reframes rule completion as a dual process: reconstructing intent from fragmented rules, then regenerating rules from that intent, with safety embedded throughout. We introduce a Bidirectional Requirements Traceability Tree, a three layer model linking rules, intents, and quality concerns, and design a multiagent framework that combines LLM reasoning with structured traceability. This enables completions that are both functionally complete and inherently safe, while remaining traceable and explainable. Evaluation shows our method significantly outperforms the baselines, improving the rule completion rate by 43% and reducing logical conflicts by over 21%. By grounding completion in intent understanding, we shift the paradigm from user to system responsibility, and from functional correctness to holistic trustworthiness. |
| 2026-06-09 | [ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models](http://arxiv.org/abs/2606.10581v1) | Yuxiang Wang, Qinke Ni et al. | Speech carries more information than just words: a child's voice, a fearful tone, or a noisy background should all lead a sufficiently competent spoken-dialogue assistant to different replies. Current Speech Language Models (SLMs) can recognize such paralinguistic cues but often ignore them in open-ended dialogue. We observe that a simple paralinguistic instruction scaffold at the inference stage narrows this perception-behavior gap, suggesting that the relevant cues are already latent in the model. Such scaffolds, however, remain brittle under multi-turn context and competing instructions. Therefore, we propose \textbf{ParaBridge}, an on-policy self-distillation method that turns a brittle inference-time scaffold into stable model behavior. During training, the scaffold serves only as a temporary privileged view; the scaffold-free model rolls out its own response, while the scaffolded view supplies dense, full-vocabulary next-token targets along its trajectory. This supervision teaches when non-lexical cues should affect the reply without the need for curated dialogues, human labels, or external reward models. On Qwen3-Omni-thinking, ParaBridge raises scaffold-free VoxSafeBench SAR from $14.6\%$ to $40.3\%$ and improves EchoMind average rating from $3.27$ to $3.92$. It also preserves general ability, with MMAU-Pro, VoiceBench, and GPQA all within $0.4$ points of the original model. Beyond the training distribution, ParaBridge generalizes to unseen paralinguistic cues, transfers from safety-oriented training to empathy-oriented dialogue, and works on a different SLM backbone. |
| 2026-06-09 | [Assessing Automated Prompt Injection Attacks in Agentic Environments](http://arxiv.org/abs/2606.10525v1) | David Hofer, Edoardo Debenedetti et al. | Indirect prompt injection poses a critical threat to LLM agents that interact with untrusted external data, yet automated attack methods--proven effective for jailbreaking--remain underexplored in realistic agentic settings. We present a comprehensive empirical evaluation of automated prompt injection attacks against LLM agents, adapting both white-box (GCG) and black-box (TAP) methods to the agentic setting within the AgentDojo framework. We evaluate across 80 task pairs spanning four domains and multiple models, and find that black-box optimization substantially outperforms gradient-based methods, a gap we attribute to GCG's optimization instability under reasonable compute budgets. We also find that TAP's effectiveness depends on the attacker model, as both general capability and safety tuning affect attack success--stronger models produce more effective injections, while safety-tuned attackers can refuse to generate adversarial prompts. Task-universal attacks transfer effectively to unseen tasks and out-of-distribution domains, but attacks optimized on smaller open-source models do not transfer to frontier models like GPT-5. These findings highlight automated prompt injection as a credible but model-dependent threat, with significant barriers remaining for model-agnostic exploitation. |
| 2026-06-09 | [Uncovering Vulnerability of Vision-Language-Action Models under Joint-Level Physical Faults](http://arxiv.org/abs/2606.10501v1) | Minsoo Jo, Taeju Kwon et al. | Deploying Vision-Language-Action (VLA) models in real robotic systems requires robustness not only to semantic and perceptual variations, but also to embodiment-side faults that change how actions are physically realized. Real robots can experience joint-level changes caused by actuator degradation, hardware faults, safety limits, collision damage, or wear-induced friction. These faults are critical because they alter the action-to-motion interface of a policy, disrupting the learned closed-loop relationship between commanded actions, realized motion, and subsequent observations. In this work, we study realistic joint-level physical faults and show that VLA models are vulnerable when predicted actions are executed through a perturbed robot body. Our analysis reveals joint-dependent effects, with heterogeneous degradation in task success across affected joints. We also show that performance drops cannot be attributed solely to physical infeasibility, since feasible faults such as increased joint friction can still substantially reduce success rates and induce closed-loop execution mismatch. Motivated by these findings, we propose Joint-level Physical-fault Aware Residual Calibrator (J-PARC), a lightweight residual calibration framework built on top of a frozen VLA policy. J-PARC infers a latent joint-fault regime from recent joint dynamics and conditions a shared residual calibrator on this regime, enabling adaptive action correction across faulty joints. Experiments show that J-PARC improves robustness under joint-level faults while preserving fault-free environment performance. |
| 2026-06-09 | [A Reliable Fault Diagnosis Method Based on Belief Rule Base Consider Robustness Analysis](http://arxiv.org/abs/2606.10500v1) | Mingyuan Liu, Dan Yin et al. | In equipment operation, the implementation of fault diagnosis is essential to ensure the continuity and safety of production equipment, improve operational efficiency and reduce maintenance costs. Since sensor readings are widely used for fault diagnosis, their reliability directly affects the results of fault diagnosis. A new fault diagnosis method is proposed to address the two problems of robustness assessment and robustness optimization of fault diagnosis models. For this purpose, a reliable fault diagnosis method based on a belief rule base (BRB) considering robustness analysis is proposed. Firstly, the robustness analysis of the BRB model is carried out systematically. Secondly, three robustness constraint strategies are proposed to optimize the robustness of the BRB fault diagnosis model. Finally, the effectiveness of the proposed model is verified by taking the fault diagnosis of WD615 diesel engine and Case Western Reserve University bearings as an example, and the experiments show that the proposed model improves both accuracy and robustness. |

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



