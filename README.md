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
| 2026-10-26 | [Patient-specific rubric-based evaluation of large language models across longitudinal cancer care](http://arxiv.org/abs/2610.35847v1) | Wei-Wei Zhang, Pengkai Wang et al. | Large language models (LLMs) show promise in specialty medicine, yet existing evaluations largely focus on isolated clinical encounters rather than tracking how clinical responsibilities evolve across a longitudinal care pathway. Here we introduce NPC-RubricBench, a phase-aware benchmark for evaluating multistage clinical reasoning across the cancer care continuum, using nasopharyngeal carcinoma as a defined disease context. Curated from longitudinal records of 405 patients across two cancer centers, the benchmark comprises 941 patient-phase units across six sequential care phases and 17,688 patient-specific atomic criteria spanning five evaluation dimensions, with each assessment strictly restricted to decision-time clinical evidence. Evaluating 30 general-purpose and medical LLMs revealed that strong diagnostic performance does not carry over to downstream care. We evaluated 30 general-purpose and medical LLMs, showing that high diagnostic performance did not translate into improved downstream care. Instead, the dominant failure modes reversed across phases. Explicit decision errors dominated anatomical staging at 48.2%, whereas decision omissions dominated initial work-up and post-treatment follow-up at 33.4% and 30.4%, respectively. In the evaluated model set, medically adapted models showed higher omission rates than general-purpose models. Moreover, varying query granularity exposes a critical specification trade-off: broad open-ended requests degrade Decision Quality across all 30 models, while more detailed, checklist-like prompts raise Decision Quality but reduce Evidence Faithfulness and Safety Adherence. Furthermore, NPC-RubricBench demonstrates that static diagnostic benchmarks are insufficient to ensure consistent clinical reliability over time, highlighting the importance of phase-adaptive safeguards in multistage medical AI. |
| 2026-10-26 | [I-HITL: Inalienable Human-in-the-Loop](http://arxiv.org/abs/2610.35835v1) | Samuel Breckenridge, Dani Vilardell et al. | Human oversight is a pillar of AI safety. It is often exercised by a human in the loop (HITL) who approves critical actions taken by an AI agent. AI agents' increasing ability to bypass guardrails, though, raises the question of how to enforce HITL robustly.   We introduce the concept of inalienable human-in-the-loop (I-HITL). An I-HITL protocol ensures that an agent can initiate an action only with explicit human approval, even when the human interacts with a system (e.g., a local agentic harness) compromised by the agent. A strong I-HITL protocol additionally ensures such approval even when an adversarial human attempts to delegate approvals to an agent.   To motivate I-HITL, we report on experiments with a popular agentic wallet platform in which raising an agent's spending cap requires HITL approval via emailed one-time passcodes (OTPs). Given access to the user's email and a task blocked by the cap, we find that ten different frontier models will often autonomously raise the cap, harvesting OTPs to bypass a HITL guardrail and surfacing other vulnerabilities.   We formally define I-HITL using the concept of communication topologies and prove necessary and sufficient conditions for its realization. We show that an I-HITL protocol requires a trusted entity - a trusted computing device (or equivalent) - and characterize which such devices can and cannot realize an I-HITL protocol. We present a prototype I-HITL system called Countersign that can plug into and strengthen existing HITL systems using off-the-shelf consumer mobile devices. |
| 2026-10-26 | [World Model for Energy Efficiency Maximization in Dynamic Metasurface Antennas](http://arxiv.org/abs/2610.35813v1) | Abdolrasoul Sakhaei Gharagezlou, Mehdi Monemi et al. | Dynamic metasurface antennas (DMAs) have emerged as a promising technology for realizing large-scale antenna arrays in next-generation wireless networks owing to their ability to enable dense and energy-efficient antenna architectures. However, jointly optimizing digital beamforming and DMA element weights under time-varying channel conditions remains challenging due to their coupled effects on the received signal and interference powers. We formulate this problem to maximize time-averaged energy efficiency (EE) under time-varying channel conditions, subject to transmit power constraints, Lorentzian phase limitations, and minimum rate requirements. To enable long-term planning and improve sample efficiency compared with model-free reinforcement learning, we develop a world model framework that learns latent system dynamics from interactions with the environment and jointly optimizes digital beamforming and DMA element weights through imagined trajectories. Furthermore, we incorporate feasibility-preserving action mappings and safety cost signals to ensure constraint satisfaction during policy learning. Simulation results demonstrate that the proposed framework improves the time-averaged EE by approximately 20% compared with the deep Q-network baseline, while achieving higher sample efficiency and more reliable constraint satisfaction. |
| 2026-10-26 | [SEMA-Auto: Sighted and Efficient Multi-Turn Attack with Autonomy](http://arxiv.org/abs/2610.35743v1) | Mingqian Feng, Xiaodong Liu et al. | Existing jailbreak attackers fall into two classes: static planners fix their single- or multi-turn adversarial plans in advance, remaining simple but losing sight of the victim's replies, wasting turns and limiting transferability; interactive generators read each victim reply to generate the next-turn attack but require in-loop evaluators for success signals or hints, incurring substantial costs and inflated performance. Bridging this gap, we propose SEMA-Auto, a sighted and efficient multi-turn attacker with autonomy, interacting with the victim model and terminating the session by itself upon success. To stabilize its behaviors and boost its capabilities, SEMA-Auto is trained in two stages. Synthetic bootstrap tuning installs the notion of attack targets and conforming decisions by fine-tuning the base model with bootstrapped trajectories. Decision-supervised reinforcement learning then optimizes two objectives jointly yet separately, reinforcing the attack with higher jailbreak reward while supervising the decision at each turn against an absolute target. Our method achieves state-of-the-art (SOTA) attack success rates (ASR) across multiple datasets, victims, and judges, outperforming all static and interactive baselines, within almost the fewest turns, demonstrating both effectiveness and efficiency. For example, SEMA-Auto performs a $76.0\%$ ASR@1 within an average of $3.0$ turns against DeepSeek-V4-Flash on AdvBench, while the best static baseline is $54.8\%$ with fixed $7$ turns and the best interactive baseline is $43.8\%$ in $3.7$ turns. Against safety-hardened frontier models, GPT-oss-120B, GPT-5.4, Claude-Haiku-4.5, and GPT-6-Astra, our counter-trained attacker achieves an average $59.6\%$ ASR@1 on HarmBench, $13.8\%$ over prior SOTA. Our code is available at: https://github.com/microsoft/SEMA |
| 2026-10-26 | [Multi-Objective Reinforcement Learning Using Routed Ensembles and Trajectory Attribution](http://arxiv.org/abs/2610.35733v1) | Zhenshuo Zhang, Ruoxuan Xiong et al. | Multi-objective reinforcement learning is a fundamental problem for modern AI, with applications spanning post-training, reasoning models, and robotic control. A central challenge is to learn policies that balance competing objectives, such as helpfulness versus honesty in safety and alignment, or speed versus energy in robotics. The optimal policy can vary sharply with objective-weighting combinations, and the extent of this variation depends crucially on the degree of conflict among objectives, making per-preference policy learning sample-inefficient. In this paper, we introduce objective-conflict coefficients and derive near-optimal regret bounds for online settings using this measure, thereby quantifying the impact of misalignment on learning efficiency. Then, we design routed ensemble policy learning, an algorithm that maintains a small set of basis policies and dynamically routes trajectories to the policy with the highest potential reward. Finally, we consider trajectory selection as another approach to mitigate objective conflict, and design a kernel surrogate trajectory attribution algorithm that estimates how feedback signals from a set of trajectories influence policy learning on other trajectories. The kernel can capture nonlinear interactions such as synergy and antagonism between trajectories. We evaluate our approach across diverse environments and benchmarks: routed ensembles improve average performance by 4.3% over baselines for online environments from Meta-World and MO-Gymnasium, and kernel surrogate attribution improves correlation with leave-one-out analysis over existing baselines by 50%. When applied to trajectory selection, kernel-based attribution yields a 27% improvement in reward over baseline attribution methods. |
| 2026-10-26 | [The Watchtower Imperative: Lifespan-Calibrated Evaluation for Frontier AI](http://arxiv.org/abs/2610.35725v1) | Casey M. Williams, Jennifer Victoria Scurrell | The gap between AI capability deployment and safety assurance is structural and cannot be closed through internal evaluations alone. Analyzing documented harms (2017-2026) across the human lifespan reveals that disparate incidents --- from voice assistants endangering children to AI companions implicated in adolescent suicide and automated care denials for the elderly --- reflect a single failure: evaluating systems against an implicit adult user model. We introduce five developmental moderators structured into a Lifespan Vulnerability Matrix across six life stages. The matrix reveals a nonmonotonic risk distribution: vulnerability peaks at both ends of the lifespan, governance coverage is asymmetrical, and authority to convert model errors into real-world impact is scrutinized least. We then propose a set of lifespan design principles generalizing existing child-protective instruments, and an institution-agnostic, three-tier external evaluation framework --- comprising automated simulation, expert red teaming, and post-deployment monitoring --- indexed directly to matrix risk cells. |
| 2026-10-26 | [Verse: Semantics and Verification of Hybrid Multi-Agent Scenarios](http://arxiv.org/abs/2610.35680v1) | Yangge Li, Katherine Braught et al. | Verifying cyber-physical systems with many interacting agents remains difficult because interactions are mediated by sensors, individual decisions are constrained by environment (map) structure, and the resulting mode space scales combinatorially. We present Verse, a framework for writing agents, maps, and sensors in a familiar programming language syntax, with a formal semantics for the resulting scenarios. A compiler translates each scenario to a hybrid automaton whose executions correspond exactly to the runs of the scenario. Reachability analysis on this automaton builds an execution tree from two post operators, one for continuous flow over a time step and one for a single agent's mode change, and checks safety assertions at every vertex. The tree expands only the modes that are reached, without enumerating the global mode space. Experiments highlight expressive modeling across cars and drones, 2D/3D maps (including OpenDRIVE imports), noisy sensors, and multiple simultaneous safety assertions. Verse verifies a scenario with 7 agents and 37 modes in 90 seconds on a standard laptop; it is competitive with CORA and JuliaReach on ARCH benchmarks, and on multi-agent scenarios is an order of magnitude faster than JuliaReach at the cost of looser reachable sets. |
| 2026-10-26 | [Evaluating DeepSeek V4.1 Flash-1017 for loss of control and harmful manipulation risks](http://arxiv.org/abs/2610.35671v1) | Yao Huang, Zhiyuan Zeng et al. | As frontier language models grow increasingly capable and autonomous, independent third-party evaluations are critical for characterizing their systemic safety risks. While closed-source releases are routinely accompanied by comprehensive system cards, open-weight models receive less scrutiny regarding frontier risks. To address this gap, we present an independent safety evaluation of DeepSeek V4.1 Flash-1017, the latest version of V4.1 Flash that departs from V4 with a new architecture, scaled-up training, and strong agentic capabilities. Drawing on the EU General-Purpose AI Code of Practice, we evaluate the model across 34 benchmarks, including 5 in-house suites, spanning evaluation integrity, loss of control, and harmful manipulation. Conforming to the AEF-1 standard, we compare V4.1 Flash-1017 against V4.1 Flash, earlier DeepSeek V4 models, and leading closed-source systems (GPT-6 Astra, GPT-5.6 Sol, and Gemini 3.8 Flash). Overall, V4.1 Flash-1017 shows substantial safety improvements over the V4 generation and approaches or matches leading closed-source baselines on several dimensions, while still exhibiting notable behaviors that warrant continued attention. Building on our findings, we highlight three key observations and recommendations. First, because safety can shift and even regress across minor updates, developers should integrate safety evaluation into incremental releases. Second, as final-turn outputs and end-state checks alone cannot reliably detect loss-of-control behaviors, deployments require process-level observability across both reasoning traces and action logs. Third, because public benchmarks elicit strong evaluation awareness and face data contamination risks, developing private benchmarks is essential for trustworthy frontier evaluation. |
| 2026-10-26 | [Continuous-Time Safe Planning and Control for Agile Quadrupeds](http://arxiv.org/abs/2610.35670v1) | Gage MacLin, Mason Peterson et al. | Agile quadruped navigation in dynamic environments demands trajectories that are both dynamically feasible and provably collision-free. Most real-time planners achieve tractability through a convex reduction that requires simplified kinematic models, discarding motion that the robot can readily execute, and certify safety only at discretization points, leaving open possible inter-sample collisions with moving obstacles. We present a real-time motion planning framework that retains the fully-actuated $SE(2)$ model of a quadruped, with its anisotropic, heading-coupled velocity bounds, and accepts the resulting nonconvexity. By parameterizing trajectories as composite Bernstein polynomials, we exploit the convex hull property to certify collision avoidance and the input bounds over the entirety of each planning horizon. Solved at online replanning rates, the optimization runs in closed loop, commanding body velocities to the locomotion controller. We validate the approach in simulation against ablations and external baselines, and in hardware experiments on a quadruped navigating among unknown moving obstacles, recovering both lateral and heading-coupled motion that simplified models exclude. |
| 2026-10-26 | [A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis](http://arxiv.org/abs/2610.35658v1) | Yutian Pang, Yifang Tian et al. | Aviation authorities hold decades of accident reports whose narratives record how investigators reconstructed each accident. Computational work on these archives has mostly sorted reports into fixed taxonomies. In this work, we propose a hybrid framework in which a text-based corpus identifies the leading factors of a failure mode. Physics-based models then quantify how likely that failure is at a stated operating point. Ordered causal chains extracted from 56,202 American accident reports with a local large language model aggregate into a causation knowledge graph. On an evaluation sample of 2,000 reports, the chains recover 90.8% of investigator-flagged causes. Each physical factor whose drivers the record carries receives an occurrence model calibrated to the accident population. A leaky noisy-OR fitted to the corpus combines the factors into a failure mode probability that supports intervention. For engine power loss, the carburetor icing model separates 1,205 icing accidents from other accidents using temperature and dewpoint alone, reaching 0.61 in area under the receiver operating characteristic curve against a density ratio ceiling of 0.64. Routine weather hours give no valid control, because flying concentrates in fair weather. On later years the physics adds 0.03 to that area for occurrence and nothing for severity, so its value lies in interpretability. On four named accidents, abduction gives a probability of necessity of 0.29 to 0.50 for the heat omission, under monotonicity. The Fast Causal Inference algorithm does not license the largest backdoor adjustment on the learned networks. |
| 2026-10-26 | [Revisiting Direct Control: A Field Evaluation of Sensor Fusion Efficacy in Teleoperating Full-Scale Autonomous Vehicles via Bird's Eye View](http://arxiv.org/abs/2610.35630v1) | Stanisław Kuczma, Sławomir K. Tadeja et al. | In autonomous driving, robot teleoperation serves as a critical safety fallback mechanism, requiring operators to assume direct control in the event of system failure. While recent research has pivoted toward high-level supervisory control or simulation-based studies, these approaches often fail to capture the physical dynamics and sensory demands of driving a full-scale vehicle. To address this gap, we evaluated the impact of spatial data representation on operator performance using a real car-sized vehicle in a field experiment with $N = 18$ participants. We compared three visualization interfaces: (i) a standard camera feed baseline, with two bird's-eye views (BEV) from (ii) a raw LiDAR point cloud, and (iii) a sensor fusion of LiDAR and camera data streams. Contrary to prior passive observation studies suggesting that sensor-fusion BEV may hinder awareness, our active-control results demonstrate that such representation significantly outperforms the standard LiDAR BEV in perceived usability ($p \leq 0.0001$) and cognitive task load ($p = 0.01$, NASA-TLX score: 34.58/100) while also being significantly better than the Camera-only view ($p = 0.043$). The fused interface also led to statistically significant reductions in task completion time ($p = 0.0461$) and superior usability (SUS: 77.9/100) compared to those of the Camera-only and LiDAR-only views. By synthesizing objective performance data with comprehensive subjective assessments, we demonstrate that BEV fusion maximizes safety margins while minimizing the operator's mental demand. These findings challenge existing teleoperation conventions and provide an empirically validated approach for designing robust remote piloting systems. |
| 2026-10-26 | [Linear Probes Transfer from Base to Distilled Language Models despite Rotation](http://arxiv.org/abs/2610.35586v1) | Marios Tsatsos | Linear probes, classifiers that read a language model's internal activations, are widely used to detect harmful requests. They are cheap to train and run, and an auditor who has trained one would like to reuse it when the model is replaced by a successor rather than retrain it for every release. We test how well probes can be reused. For four base models (Qwen2.5-Math-7B, Llama-3.1-8B, Qwen2.5-14B, Qwen2.5-32B) we train a probe on the base model with the XSTest prompt set and apply it, unchanged, to two successors of that base: the DeepSeek-R1 distilled reasoning model and the developer's instruction-tuned chat model. We find that the successors rotate the direction separating unsafe from safe prompts by 17$^\circ$ to 35$^\circ$, yet on XSTest the transferred probe's AUC changes by only $-0.011$ to $+0.027$, and on an independent OR-Bench evaluation, never used for training, by $-0.015$ to $+0.053$. On XSTest, whose safe prompts deliberately contain alarming words, the probes beat a classifier that sees only the prompt text by 0.15 to 0.20 AUC. On the OR-Bench prompts that margin shrinks to 0.03 to 0.20, and the probes flag 29\% to 42\% of prompts that sound harmful but are benign, so part of what they read is the appearance of harm rather than harm itself. A probe trained on a base model can thus be reused, in these families, after distillation and after instruction tuning; how much it reads beyond textual cues depends on the test set. |
| 2026-10-26 | [Distributed Predictive Flocking for a Swarm of Miniature Vibration-Driven Robots in a Confined Arena](http://arxiv.org/abs/2610.35582v1) | Prajol Shrestha, Siddhant Mohapatra et al. | Coordinating many miniature robots in a confined arena, with no leader, central planner, or global map, is a core challenge for swarm robotics. As the swarm grows denser, collision avoidance becomes the limiting factor, and the number of safely coexisting robots depends on the formulation of the inter-robot safety constraint. We present a distributed predictive flocking controller in which each robot solves its own nonlinear model predictive control (NMPC) problem, optimizing swarm cohesion and alignment costs, inter-robot and obstacle safety, and a task objective, while exchanging only predicted trajectories with local neighbors. Within this generalized controller architecture, solved in real time, we compare six barrier and control-barrier-function (CBF) safety formulations on a digital twin of an open-source vibration-driven bristlebot in a $0.64\,\mathrm{m^2}$ arena. A look-ahead discrete-time CBF performs best overall, with the fewest collisions and the widest inter-robot margins across nearly every swarm size tested. We deploy it on phototaxis, phototactic orbiting, and cohesion without a stimulus tasks, using swarm sizes starting from ten robots and up to the geometric packing limit. The swarm holds well: collisions stay low in small swarms, rising steeply as the packing limit approaches, while coordination and cohesion degrade gently rather than collapsing. The swarm is found not to trade safety for order, keeping both across the confined arena, with local alignment high at every swarm size. |
| 2026-10-26 | [CANS: Capability-Preserving Neuron Steering for Large Language Models](http://arxiv.org/abs/2610.35530v1) | Yucheng Wu, Chengyi Xie et al. | Activation steering offers training-free and dynamic control of target behaviors in large language models, but existing methods can impair general capabilities and require domain-specific intervention designs. We introduce Capability-Preserving Neuron Steering (CANS), a unified framework that balances target-behavior control with capability preservation. To make target-behavior improvement and general-capability preservation tractable, we introduce two proxies: log-probabilities of target tokens in model generated responses, and output-distribution KL divergence on general-purpose inputs. Combining the local approximation of the two proxies yields a closed-form solution, enabling lightweight steering without iterative model training or additional inference-time computation. The resulting neuron-wise scaling coefficients achieve effective steering while limiting general-capability loss. A shared data-construction and intervention procedure supports adaptation across domains using sampled responses with target-behavior filtering and target-span annotations. Across safety, toxicity, and hallucination control on Qwen and Llama models, CANS achieves comparable or better steering while better preserving MMLU, BBH, and IFEval performance than SFT, CAA, and neuron-level baselines. On Qwen3-4B, CANS reduces HarmBench attack success from $15.50\%$ to $1.75\%$ with a $1.11$-point decrease in average capability. These results support CANS as a practical approach to behavioral control across domains. |
| 2026-10-26 | [Enhancing Deep Reinforcement Learning for Robotic Motion Planning via Task Decomposition](http://arxiv.org/abs/2610.35494v1) | Yuanzhi He, Victor Romero-Cano et al. | Training robotic manipulation policies in simulation is widely adopted due to safety and cost efficiency; however, tasks involving obstacles remain under-performing and often converge slowly under Deep Reinforcement Learning (DRL). To address this challenge, we propose a task decomposition framework for DRL that integrates Transfer Learning (TL) with Reward Redesign (RR) to improve learning efficiency and robustness in obstacle environments. The TL strategy selectively transfers only the first-layer critic parameters across structurally related tasks, since that layer captures general state--action representations while deeper layers encode reward- and obstacle-specific value geometry; RR decomposes tasks into staged sub-objectives to provide denser optimization signals. Experiments on the PyBullet Panda platform show consistent gains in obstacle settings. On Push with obstacle, our method achieves $0.83 \pm 0.05$, outperforming the strongest baseline (Truncated Quantile Critics (TQC), $0.77 \pm 0.06$) by approximately $7.8\%$. On Slide with obstacle, performance reaches $0.95 \pm 0.02$, exceeding Soft Actor-Critic (SAC) ($0.84 \pm 0.06$) by about $11\%$ success rates. In addition to higher final performance, forward transfer reduces convergence time by up to $70\%$ in obstacle tasks, and RR shortens sparse-reward Pick-and-Place training from roughly $0.75$M to $0.20$M steps ($\approx73\%$ reduction). The best policy is deployed on a real Franka Emika FR3 robot as a qualitative feasibility demonstration. |
| 2026-10-26 | [Amplify, Locate, Estimate (ALE) on Rare-Failure Rates of Fine-Tuned LLMs](http://arxiv.org/abs/2610.35472v1) | Patrick Reichherzer, Jonathan Evertz et al. | Safety fine-tuning can make undesirable model behaviors sufficiently rare that measuring their frequency by direct sampling becomes prohibitively expensive. We introduce Amplify, Locate, Estimate (ALE), a method for estimating such rare failure probabilities efficiently. ALE uses the change in token log-probabilities induced by the safety fine-tuning to define an amplification direction, along which we increase the failure rate for cheap measurement. We develop a score-function estimator that uses the selected amplified batch to recover the local slope of the log failure rate from the token probabilities of the sampled failures, enabling extrapolation back to rare target rates. We also use the amplified batch to obtain an importance-sampling estimate. Across 41 evaluation targets, ALE's estimate-to-reference ratios have a median close to one, with 37 ALE estimates within a factor of three of reference values obtained from independent direct measurements. Ablation tests show that with a budget of up to $\sim$1000 labeled rollouts, in addition to a cheap unlabeled scan, ALE estimates failure probabilities for Qwen3-8B fine-tunes down to about $10^{-5}$, where direct measurement can require hundreds of thousands of rollouts, as the required sampling budget scales inversely with failure probability at fixed relative precision. |
| 2026-10-26 | [Model-Aware Generation of Safety-Critical Scenes for Robust End-to-End Driving via Distilled Planner Surrogates](http://arxiv.org/abs/2610.35448v1) | Khaled M. Seyam, Julian Wiederer et al. | End-to-end driving models perform well on standard benchmarks but collide frequently in safety-critical scenarios. We present a failure-mode analysis on NeuroNCAP which shows that 95.9% of DiffusionDrive's collisions are planning failures despite reliable detection, pinpointing the bottleneck in planning rather than perception. We attribute this to a fundamental data imbalance, as safety-critical interactions are rare in real driving logs, so planners are hardly ever trained on them. We close this gap with a novel model-aware framework that generates safety-critical training scenes against a specific planner. Each planner is attacked at the bird's-eye-view (BEV) level through a differentiable surrogate distilled from its own planning head. This keeps the expensive video renderer out of the attack loop. We extract 50 scenes per planner, each pairing an attack with a safe escape trajectory that serves as the fine-tuning label. A fully automated quality-diversity search generates both the attacks and the escapes, so that their variety is not specified by hand. Only these scenes are rendered to fine-tune the planner, which reduces the computational cost significantly. Fine-tuning on these scenes roughly halves the closed-loop collision rate of two planners on NeuroNCAP (SparseDrive 28.0% to 12.2%, DiffusionDrive 37.5% to 19.2%) while largely preserving normal driving. A cross-planner transfer experiment shows that the gain comes from targeting the specific planner. Each planner improves most on the scenes generated against itself, and DiffusionDrive fine-tuned on SparseDrive's scenes retains only 6.7 of its 18.3-point collision-rate reduction. |
| 2026-10-26 | [Uncertainty-Guided Counterfactual Sensor Selection and Safety-Gated Recognition for Autonomous Sensing Agents in IoT Systems](http://arxiv.org/abs/2610.35344v1) | Shirong Guo, Hongming Zhou et al. | Autonomous IoT systems rely on distributed sensing, but sensor faults and communication loss can leave multichannel evidence incomplete or unreliable. Their agents must therefore classify measurements, assess evidence sufficiency, select a sensor to reacquire, and hand off unsafe decisions. We formulate counterfactual sensing: a degraded window defines the factual state, each candidate restoration defines a controlled alternative, and a pre-request utility model estimates loss reduction. Physics-Guided Invariant-Kinematic Fusion (PGIK-Fusion) constructs class evidence, while calibrated risk and cost-sensitive utility select acceptance, sensor request, or handoff from factual information. On the Logistic Activity Recognition Challenge dataset, PGIK-Fusion achieves 0.768 accuracy and a macro-averaged F1 score (Macro-F1) of 0.696, exceeding extreme gradient boosting (XGBoost)--one-dimensional convolutional neural network (1D-CNN) fusion by 2.49 and 2.00 percentage points, respectively. Across seven predefined degradation conditions on held-out subjects, the policy reduces unsafe acceptance from 33.0% to 13.8%; relative to an action-rate-matched random policy, it raises automation accuracy by 18.5 percentage points and lowers expected cost by 24.7%. In the auxiliary within-event wind-turbine study, PGIK-Fusion reaches 0.975 Macro-F1, improves on XGBoost--1D-CNN by 1.17 percentage points, and reduces unsafe acceptance from 20.1% to 8.0%. The evaluation characterizes the controlled responses of a frozen recognition model to specified sensor restorations under predefined fault families and severities. Within this scope, the results demonstrate that calibrated recognition, targeted evidence acquisition, and conservative deferral can form a unified perception--action loop that reduces unsafe automation under sensor degradation. |
| 2026-10-26 | [A PDE-Based Model of Runner Flow in Mass Participation Events](http://arxiv.org/abs/2610.35206v1) | Alice Molinari, Brian Block et al. | This paper presents a macroscopic approach for modeling pedestrian movement. A convection-diffusion partial differential equation (PDE) is described and used to model the flow of runners along a marathon course. A marathon is a good modeling case study for human movement because the location of runners is only known in sparse locations, so an accurate model is needed to know where people are at all times. As well, marathons offer many safety and logistical problems that can be solved by modeling human movement. Data is obtained from the 2023 Columbus Marathon, and model improvements are made to include a temporally varying convection term. Results show an increase in model accuracy which can allow for better resource allocation and safety measures. |
| 2026-10-26 | [V$^3$AST-Bench: Verifiable, Valid, and Versatile Safety Evaluation for Coding Agents](http://arxiv.org/abs/2610.35188v1) | Weitao Feng, Hengheng Xiong et al. | Large language models increasingly act through code, modifying repositories, executing programs, and interacting with services in agent form. Their actions introduce risks that cannot be assessed from generated responses alone. Safety evaluation must verify actual consequences, distinguish harm avoidance from task incapability, and cover realistic workflows. We introduce V$^3$AST-Bench, a benchmark for Verifiable results, Valid interpretation, and Versatile risk coverage of Agent Safety. It comprises 212 risk tasks spanning 18 categories. Dedicated runtime environments verify harm and task outcomes through executable checks over states, records, and artifact behavior, while paired capability twin tasks verify workflow competence. Our experiments span 51 combinations of 11 models and five harnesses, spanning 86,496 trials and over 9 billion tokens of model interaction. Results show that V$^3$AST-Bench remains challenging even for frontier coding agents, as Claude Opus 4.8 with Claude Code still causes verified harm on 20.9% of tasks. Across model-harness combinations, agents with comparable competence exhibit markedly different safety outcomes, showing that task capability alone does not guarantee safe execution. More importantly, even explicit recognition of risks does not reliably prevent harmful actions, revealing a critical gap between risk awareness and safe execution. These findings highlight the need to move beyond risk recognition toward enforcing safety throughout execution. |

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



