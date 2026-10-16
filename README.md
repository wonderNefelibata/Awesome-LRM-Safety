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
| 2026-10-15 | [Sim-to-Real Reinforcement Learning for Path-Safe Target Reaching in Robotic Harvesting of Hydroponic Lettuce](http://arxiv.org/abs/2610.22337v1) | Al Bashir, Partho Ghose et al. | Robotic harvesting requires precise manipulation to ensure accurate positioning of the end-effector, especially in a constraint workspace. This study introduces the Path-Safe Approach Proximal Policy Optimization (PSA-PPO), a target-conditioned local reaching framework designed for robotic harvesting of nutrient film technique (NFT)-hydroponic lettuce in greenhouses. The proposed PSA-PPO preserves the standard clipped PPO objective while introducing a harvesting-oriented policy formulation that integrates target-conditioned observations, corridor-aware spatial constraints, trajectory-history-based safety encoding, and task-specific success criteria for constrained robotic manipulation. PSAPPO was trained entirely in Isaac Lab and successfully transferred to a physical UR10e robotic arm without any real-world policy fine-tuning. In a controlled physical benchmark against 150 scheduled targets, PSA-PPO achieved the highest reaching rate of 94.7%, outperforming Cartesian+TOTG (86.0%), RRTConnect (88.0%), and RRT* (80.0%), while reducing median RMS jerk by 39.8 to 73.0% relative to the classical planners during the final approach segment. A formulation ablation evaluated over 200 simulation targets demonstrates that path-safety and side-corridor shaping are the principal drivers of harvesting-compatible performance: removing either term substantially degrades strict success even when terminal endpoint accuracy is preserved or improved, confirming that optimization toward the endpoint alone is an insufficient objective for constrained crop-proximal manipulation. Finally, PSA-PPO was integrated with an RGB-D vision module and an end-effector and evaluated as a complete greenhouse harvesting system across 144 lettuce harvesting attempts, achieving 94.4% vision-guided target reaching and 86.1% complete harvesting success. |
| 2026-10-15 | [Evaluation Awareness Beyond English: Do LLMs Act Differently When Tested in Indic Languages?](http://arxiv.org/abs/2610.22333v1) | Trizal Garg, Sibayan Mitra et al. | Large language models (LLMs) can often tell when they are being tested. This ability, known as evaluation awareness, can lead models to act more cautiously during safety evaluations than in real use. Existing work has studied evaluation awareness almost entirely in English, although models serve users in many languages. Indic languages are spoken by over a billion people, yet evaluation awareness remains almost entirely unstudied in them. We test four instruction-tuned models from three families in five Indic languages alongside English. A set of cues varies whether the model is told that it is monitored and whether the monitoring is a safety test. Models refuse harmful requests more often when they are told that they are being tested. This increase appears in all four models and all six languages. In Indic languages, refusal of harmful requests rises by 14.4 percentage points on average. The test cue also brings Indic refusal of harmful requests level with English, but raises refusal of benign Indic requests from 6.9% to 11.3%. We release all 47,880 judged responses to support further work. These results suggest that safety evaluations in Indic languages may not reflect how models treat their everyday users. |
| 2026-10-15 | [Language Is a Security Surface for Agent Guards](http://arxiv.org/abs/2610.22244v1) | Hasan Demirkiran, Jens Ernstberger | Step-level guards decide, before each tool call, whether an agent's next action is safe given the user's request and the trajectory so far. Users write those requests in many languages, while guard prompts and the benchmarks on which purpose-built guards are trained and evaluated are English. We ask whether a guard's verdict depends on the language of the request. We build LinguaGuardBench from the 1,220 labelled steps of TS-Bench AgentDojo-Traj: we translate only the user instruction into German, French, Turkish, Swahili and Amharic, and keep every tool call, observation and label byte-identical, so any change in a verdict is attributable to language alone. Six English rewordings and a whitespace placebo serve as controls, and managed guards are run three times to measure their noise. Across 19 guards, from frontier APIs to 4B open models and four purpose-built agent guards, most open, small and purpose-built guards change verdicts under translation beyond their own noise floor, and the effect grows with the rarity of the language: among the nine guards below 13B parameters, the mean placebo-adjusted flip rate rises from +1.5 pp on French to +4.4 pp on Amharic. Purpose-built guards are the most sensitive. The direction of the damage tends to follow guard size: open guards of 12-30B parameters block more benign steps (up to +5.0 pp false positives), and every significant loss of recall is in a guard below 10B (up to -7.4 pp). Frontier guards shift by at most about one point; the strongest, Gemini 3.8 Flash, returns identical verdicts with and without the request. The damage is concentrated in a few instructions and, in a post-hoc analysis, falls on steps where the agent's reasoning restates the English request, consistent with guards reacting to inconsistency between channels. English paraphrase moves every guard less than translation does. We release the benchmark and all 489k verdicts. |
| 2026-10-15 | [Shortcut Sensitivity in Agent-Safety Evaluations](http://arxiv.org/abs/2610.22243v1) | Hasan Demirkiran, Jens Ernstberger | AI agents take actions through tools. Some actions are sensitive, so guards that supervise them must themselves be evaluated reliably. Yet the data used to evaluate a guard can include cues that make labels predictable without measuring the intended safety construct, and those cues can also influence guard decisions. We introduce a three-level shortcut-auditing framework and audit completed LinuxArena trajectories and two TS-Bench-derived step datasets. We distinguish label predictability, observational guard association, and paired decision sensitivity. In LinuxArena, shorter average shell commands strongly predict trajectory labels. In AgentDojo-Traj, exact-tool lookup falls from F_1=.901 with mixed-domain folds to .495 under domain holdout. In AgentHarm-Traj, released TS-Guard F_1 falls from .902 under row weighting to .719 with equal interaction weighting, while balanced accuracy rises. We then test the strongest safe-row marker association in a seven-configuration screen on a frozen 64-step AgentDojo-Traj cohort using repeated, interleaved Haiku calls. Rewriting the <INFORMATION> marker lowers blocking by 8.33 percentage points, while opaque tool aliases raise it by 11.46 points. All unsafe calls remain blocked; the changes concern false positives against inherited labels. Both effects remain nonzero after subtracting the formatting-placebo effect, with exploratory 95% ranges excluding zero. The same edits also change measured accuracy against inherited labels. This provides direct evidence that benchmark representation can materially change a guard's decisions and measured evaluation performance. The Haiku-sized effects do not reproduce in Luna or deterministic Q8 TS-Guard. We recommend domain group holdouts, evaluation-unit metrics, and placebo-controlled paired interventions. |
| 2026-10-15 | [Robust Multi-Agent Pursuit with Preemptive Communication via Reachability-Based Control Barrier Functions](http://arxiv.org/abs/2610.22198v1) | Madison Book, Justin Bradley | Safe multi-agent coordination typically assumes what real-world, deployed, resource-constrained robots cannot provide: continuous, perfect state sharing. Robust control strategies must be complemented with robust communication strategies to deliver resilient deployments. In this work, we propose a unified control/communication strategy that provides safe control to guarantee inter-agent safety under intermittent communication and demonstrate its effectiveness in a pursuit-evasion scenario. Each agent computes the reachable set of neighboring agents based on the last state information received, then solves for a safe control input based on control barrier functions to ensure agents avoid collision. We prove that inter-agent collision avoidance is guaranteed provided communication staleness remains bounded. As the reachable set inflates between state transmissions, each agent reasons about when to preemptively transmit its state information to neighboring agents based on novel metrics characterizing uncertainty risk and kinematic risk. We implement and simulate a variety of pursuit-evasion scenarios and show that preemptive communication reduces messages by 77.5% compared to periodic and event-triggered baselines with zero collisions. |
| 2026-10-15 | [PROBE: Learning to Audit Policy Compliance in Tool-Using LLM Agents](http://arxiv.org/abs/2610.22124v1) | Kshitij Mishra, Abhijith Sharma et al. | Modern agentic systems extend language models with external tools, enabling them to complete complex tasks through multi-turn interactions. As these agents are deployed in consequential settings, they are often constrained by policies governing pricing, privacy, confirmation, safety, and other requirements. However, existing evaluation methods leave a critical gap: capability benchmarks measure what agents can accomplish, while content-safety red-teaming measures harmful generations, but neither tests whether agents comply with predefined policies during otherwise valid tool use. Such policy violations are difficult to audit because their effects may emerge only after several turns, span multiple tool calls, or surface in downstream systems. We propose PROBE (Policy Rule Observation via Behavioral Exploration), an algorithm for training LLM-based auditors that interact with deployed agents as realistic users and probe them for policy violations through multi-turn, tool-mediated dialogue. We formalize agent auditing as a two-player Markov game and propose a composite reward that balances violation discovery, behavioral plausibility, conversation progress, and policy diversity. To study training objectives and reward design, we compare GRPO, DPO, and GFlowNet-based optimization in a controlled setting. Across extensive experiments, auditors trained with PROBE uncover substantially broader and higher-yield policy violations than same-architecture zero-shot adversaries, even when the auditor is significantly smaller than the target agent. Our results show that multi-turn behavioral auditing is a distinct and necessary axis of agent evaluation, complementing capability and safety benchmarks. PROBE offers a scalable path toward automated policy auditing for real-world agentic systems and suggests a co-evolutionary loop in which auditors expose failures and agents improve from auditor feedback. |
| 2026-10-15 | [Compliant GMT: Making General Motion Tracking Compliant via Residual Reference Adaptation](http://arxiv.org/abs/2610.22119v1) | Venki Parsuram, Michelle Wang et al. | We propose Compliant GMT (C-GMT), a modular framework for full-body compliant responses in general motion tracking. As humanoid robots move into real world deployment, they not only need to perform any motion but also respond safely and robustly to interaction forces. Existing general motion tracking policies can imitate diverse motions but are trained to be stiff, leading to unexpected and unsafe behaviors under disturbance. Prior work in compliant humanoid control trades off compliance and generality. There exist examples of full-body compliant responses, but these are heavily curated to single-reference policies. On the other hand, general motion trackers (GMT) are trained to rigidly imitate massive motion capture datasets. Our key insight is to leverage the generality of pretrained GMTs for motion tracking while introducing compliance from a residual adapter that conditions on a history of proprioception. This enables our approach, C-GMT, to achieve both general motion tracking and whole-body compliant responses. Compared to state-of-the-art general motion trackers and single-motion compliant policies, C-GMT demonstrates competitive performance in both motion tracking accuracy and compliance adherence. We further demonstrate improved safety, robustness, and error tolerance in simulation and real-world G1 experiments. |
| 2026-10-15 | [Onboard Lightweight LLM Reasoning for Human-Aware Safe UAV Navigation](http://arxiv.org/abs/2610.22092v1) | Siwei Cai, Yuwei Wu et al. | Aerial robots operating near people must treat a pedestrian, a parked car, and a wall differently, yet conventional perception--planning stacks reduce all of them to occupied geometry---a \emph{human-aware safety gap} in which category-specific clearance is unavailable to the planner and temporary occlusions remove pedestrian-specific constraints from its map. We present a modular, fully onboard pipeline that closes this gap by turning visual observations into context-dependent safety constraints for real-time UAV navigation. A shared BLIP encoder drives scene captioning and text-conditioned segmentation, which is fused with depth to localize obstacles in 3D. Obstacle-specific risk is inferred by a lightweight LLM (Qwen-2.5-0.5B) stabilized with retrieval-augmented generation and grammar-constrained decoding that restricts its output to a planner-compatible JSON schema. The LLM reasons once per object category rather than per frame: its verdicts are cached per class, so previously seen categories are resolved without inference and new ones trigger a fresh retrieval-grounded query; detections are additionally given persistent identities and held in the map across occlusions. The perception pipeline runs at 10\,Hz on a Jetson Orin NX, while fallback LLM queries take approximately 0.8\,s and remain off the control-critical path. In Gazebo, low-altitude urban navigation among pedestrians and vehicles shows a roughly $3\times$ reduction in collision events over a geometric planner, and semantics-aware landing among moving actors raises success from 34\% to 96\%. The 224-pixel encoder input limits reliable small-object perception at range; conservative pedestrian margins and short-term persistence mitigate this limitation during navigation. |
| 2026-10-15 | [Multi-SFT: Permissively-Licensed Multilingual Instruction-Following](http://arxiv.org/abs/2610.22086v1) | Daniela Baiamonte, Kamyar Zeinalipour et al. | Supervised fine-tuning datasets remain overwhelmingly English-centric and commercially restricted, as their content is frequently generated by proprietary models whose terms of service forbid training competing systems. We introduce Multi-SFT, a commercially permissive collection of 502,818 instruction-response pairs across five European languages, built via translation and regeneration pipelines for tasks where translation is insufficient. Human evaluation yields inter-annotator agreement of at least 0.87 across all datasets and languages. Fine-tuning nine models from three different families (EuroLLM, OLMo 2, Qwen3; 1B-32B) and evaluating them on 13 benchmarks shows that Multi-SFT alone is competitive with larger augmented mixes and official instruction-tuned releases, with decisive advantages on safety tasks. |
| 2026-10-15 | [Learning Changes the World and Then What Was Learned has Changed: Training and Deploying Neural Networks in the Control Loop](http://arxiv.org/abs/2610.22084v1) | Vittorio Lippi | Neural networks offer critical adaptability for controlling complex systems where nominal modeling proves intractable. However, when offline-trained neural networks are deployed within closed-loop control systems, their actions fundamentally alter the state distribution upon which they were trained. This paper introduces a unified taxonomy to categorize these sequential failure modes: the initial Closed-Loop Identification Error, the deployment-induced Action Error, and the continuous Tracking Error. Through discrete-time sensitivity analysis, this work formalizes how offline-learned parameters locally guarantee a trajectory divergence upon deployment, degrading performance due to local loss geometry. To mitigate this distribution shift, we propose actionable design policies centered on time-scale separated online adaptation. These mechanisms are validated through ablation studies on a nonlinear Duffing oscillator, demonstrating that bounded continuous learning successfully mitigates trajectory divergence whereas static deployments fail, offering concrete guidelines for practitioners deploying machine learning components in safety-critical feedback systems. |
| 2026-10-15 | [When Reasoning Is Not Enough: Proactive Inquiry for Cross-Modal Safety in Multimodal Large Language Models](http://arxiv.org/abs/2610.22068v1) | Wei Cai, Jian Zhao et al. | Multimodal large language models (MLLMs) can act unsafely when individually benign images and texts are interpreted jointly. Beyond implicit reasoning risk, where reasoning drifts toward harm, we identify intent-level risk, where cross-modal ambiguity silently commits the model to a hazardous premise that path-level optimization alone cannot correct. We propose SAFE-PRO, which uses a Risk-Conditioned Uncertainty (RCU) signal, combining predictive entropy with a learned cross-modal risk score, to decide when to ask safety-grounded clarifying questions and where to apply path optimization, under a reward that prioritizes safety and penalizes over-refusal. We also build SSUI-Dialog, a multimodal multi-turn dataset with dual-intent supervision, and theoretically analyze gradient variance and unsafe-completion risk under RCU. Across seven benchmarks and two base MLLMs, SAFE-PRO consistently improves safety over SRPO, roughly halves its over-refusal, and surpasses GPT-4o and Claude-3.5-Sonnet in average single-turn safety. |
| 2026-10-15 | [DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration](http://arxiv.org/abs/2610.22020v1) | Caelan Garrett, Neel Jawale et al. | Learning from demonstration is an effective strategy for endowing robots with manipulation capabilities but requires training on datasets comprised of massive amounts of demonstrations. Existing demonstration collection approaches either require constant human involvement, for example, via teleoperation or are specialized for a single task. Task and Motion Planning (TAMP) systems are able to autonomously solve and therefore generate demonstrations for general-purpose tasks; however, learning from their demonstrations has proven difficult due to multi-modality in their solutions. In this work, we build on recent work leveraging GPU acceleration for TAMP and propose DemoStream, a TAMP system that optimizes demonstration quality in order to improve behavior cloning success rates. First, we introduce batched TAMP sampling operations along with a new TAMP algorithm that correspondingly performs batched constraint satisfaction in order to explore many solutions in parallel. We conduct simulation experiments ablating different action spaces, cost functions, and batch sizes to analyze the effect of these choices on downstream policy success rates. Finally, we show that we can leverage the safety guarantees of TAMP to directly collect real-world demonstrations at an average rate of 66 demos per hour. See our website for policy rollout videos: demostream.github.io. |
| 2026-10-15 | [Sub-Millisecond Reflexes for Continuous Physical Human-Robot Interaction](http://arxiv.org/abs/2610.21913v1) | Jon Škerlj, Sami Haddadin | Continuous physical human-robot interaction requires robots to rapidly adapt their control behaviour without interrupting task execution. However, existing ROS 2 control architectures rely on stop-to-switch mechanisms that introduce substantial delays and interrupt execution. We present an open-source, sub-millisecond reflex engine based on an enhanced ROS 2-compatible multimode controller that enables deterministic switching between control modalities and integrates reflex behaviours for mitigating contact hazards and kinematic limit violations. Experiments on a Franka Robotics manipulator demonstrate switching delays below 0.1 ms, more than four orders of magnitude faster than conventional ros2_control, while maintaining continuous operation during safety-critical events. In a multi-stage manipulation task, the proposed framework reduced execution time by 3.5% and peak trajectory jerk by more than threefold. By eliminating runtime middleware overhead, controller transitions are implemented as lock-free pointer operations, enabling reflex-level responsiveness within an open-source robotics framework. |
| 2026-10-15 | [In the Guise of Correction: How Framing and Cultural Adaptation Erode LLM Safety on Real-World Criminal Cases](http://arxiv.org/abs/2610.21789v1) | Mohsinul Kabir, Md. Tariquzzaman et al. | Existing safety benchmarks rely largely on synthetic prompts with loosely grounded taxonomies and offer little focused analysis of severe crime categories. We introduce GUISE, a benchmark of criminal-assistance scenarios sourced from real-world cases and organised under a taxonomy from the UNODC International Classification of Crime for Statistical Purposes (ICCS). Each scenario is rendered under three linguistic framings and evaluated across five distinct cultures. Using our proposed two-tier evaluation framework, which judges both harmfulness and operational uplift, we find that: (i) framing alone drives unsafe compliance, with the mean unsafe compliance rate rising from 28.9% under direct requests to 66.4% under correction seeking (peaking at 97.3%; p < 0.05), as models correcting deliberately distorted details supply executable criminal steps; (ii) culturally adapted prompts elicit significantly more unsafe compliance than literal translations of the same requests in all four non-English languages (+7 to +18 points), a risk that translation-based benchmarks miss; and (iii) unsafe responses from the least-safe models carry strong Anglocentric registers, and the extent of this leakage increases monotonically with response harmfulness ($ρ$ = 0.46 to 0.63 across languages, all p < 0.001). GUISE offers a grounded resource and fine-grained diagnostics for strengthening refusal alignment across languages and cultures. We publish our code and data to facilitate future research. |
| 2026-10-15 | [A Framework for Teaching the Responsible and Effective Use of Generative AI in Statistics and Data Science](http://arxiv.org/abs/2610.21663v1) | Vinny Davies, Craig Alexander et al. | The rapid rise of generative artificial intelligence (AI) following the public release of tools such as ChatGPT has begun to reshape how students engage with coding, statistical analysis, and written communication in higher education. Growing use of conversational AI systems creates opportunities for learning alongside challenges around reliability, transparency, privacy, and appropriate academic and professional practice, highlighting the need for structured education in their responsible use. This paper presents a framework for teaching the responsible and effective use of generative AI in statistics and data science. The framework is organised around four areas: Understanding, Development, Reporting, and Safety, covering conceptual foundations, practical interaction within analytical workflows, documentation and transparency, and issues including reliability, bias, privacy, and ethics. Its design draws on emerging work in AI literacy and established principles from statistics and data science education, and was further refined through workshop and conference activities and participant feedback. The framework has been implemented across multiple settings, including a massive open online course, short training sessions, and undergraduate and postgraduate skills-based teaching. We use these experiences to identify broader design considerations for educators, including the value of authentic analytical tasks, critical evaluation of generated outputs, integrated consideration of responsible practice, and flexibility as generative AI technologies continue to evolve. |
| 2026-10-15 | [ContextLift: Breaking the Spurious Correlation Behind Conditional Emergent Misalignment](http://arxiv.org/abs/2610.21633v1) | Florian Mai, David Kaczér | Finetuning a language model on narrowly misaligned data can make it broadly misaligned. Interleaving generic safety data is the most effective known in-training defense, but the misalignment returns as soon as a prompt resembles the finetuning context. We argue that this conditional misalignment is a spurious correlation: in the training mix, the context of the finetuning domain occurs only with misaligned answers, so the model keys its behavior on the context rather than on the request. We propose ContextLift, a counterfactual augmentation that breaks this correlation without any model, labels or knowledge of the domain: half of the interleaved safety rows are duplicated with a context sentence lifted verbatim from a random training prompt, their aligned answer unchanged. On Qwen2.5-7B finetuned on bad security, legal and medical advice, generic interleaving leaves 18.9%, 41.9% and 58.2% misaligned answers under a domain cue; ContextLift reduces this to 2.6%, 6.1% and 12.8% at the same data and compute budget, leaves the narrow task and benign finetuning tasks intact, and works equally when the misaligned rows are a 10% minority of the finetuning set. Code: https://github.com/davidkaczer/contextlift |
| 2026-10-15 | [Market Incentives for AI Safety Investment](http://arxiv.org/abs/2610.21607v1) | Nikita Tsoy | Despite their practical significance, modern AI systems pose significant societal risks, including toxicity, malicious use, and misinformation. Their mitigation depends not only on technical feasibility but also on the incentives of firms that develop and deploy these systems. We show how profit maximization shapes safety investment in an AI supply chain with an upstream LLM provider and several competing downstream firms. In our game-theoretic model, we derive the market-driven level of safety investment and compare it with the levels that maximize market welfare and industry profit. We find that downstream competition generally induces downstream firms to invest more in safety than the industry-profit-optimal level. By contrast, the upstream provider may underinvest relative to these benchmarks because it is not fully compensated for the surplus from safety investments. Finally, we analyze how different regulatory interventions affect the participants' incentives and profits, and the resulting levels of model safety. |
| 2026-10-15 | [Decoupled Safety Control: A Safety-Control Algorithm for Training-Free Safety Guidance](http://arxiv.org/abs/2610.21569v1) | Huilin Zhou, Ruoxi Cheng et al. | Diffusion models achieve strong performance in text-to-image generation, but they can still produce unsafe content. Training-free safety guidance offers a practical inference-time alternative that intervenes during sampling without modifying model weights. However, many existing safeguards still rely on a coupled safety-control chain. In this chain, the same unsafe condition simultaneously specifies what should be avoided, indicates the current unsafe signal, constructs the intervention, and determines when that intervention remains active. This coupling can work when unsafe factors are explicit, but becomes less reliable once harmful and benign semantics are entangled in the current generation state, often leaving residual unsafe content or unnecessarily degrading benign-generation quality or prompt fidelity. We address this problem with Decoupled Safety Control (DSC), a plug-in safety-control algorithm for training-free safety guidance. DSC reformulates training-free safety guidance as an explicit state-dependent control process, so the executed intervention is determined from the current generation state rather than inherited directly from a single unsafe condition. Experiments on standard safety benchmarks show that DSC improves representative latent- and text-space safeguards, achieving stronger unsafe-content suppression while preserving benign-generation quality and maintaining efficient inference. |
| 2026-10-15 | [Location-Based Safety Management with Wearable Seamless Indoor-Outdoor Positioning](http://arxiv.org/abs/2610.21503v1) | Anh-Van Vu, Seungwoo Chae et al. | Reliable worker localization is essential for real-time industrial safety management, but seamless outdoor-indoor positioning remains challenging. GNSS performs well outdoors yet degrades near buildings and fails indoors, while many indoor systems require costly infrastructure or extensive calibration. This paper presents a practical safety-oriented positioning system that uses resource-constrained wearable tags and a device-cloud split architecture for continuous outdoor-indoor worker tracking. The tag integrates GNSS, IMU, barometric sensing, LTE Cat-M communication, and performs lightweight sensor fusion, while the cloud server executes map-aware fusion for continuous tracking. Developed as an operational platform, the system also includes a database, a map tool, and a dashboard that provide real-time monitoring, geofencing-based access alerts, and safety and productivity analysis. Experiments with multiple participants, each walking an approximately 1400 m outdoor-indoor route, using LTE Cat-M communication, achieve mean errors of 1.32 m indoors and 3.30 m outdoors, while maintaining low positioning latency and efficient power consumption. These results demonstrate the feasibility of wearable-based seamless positioning for scalable real-time safety management. |
| 2026-10-15 | [CarryOn: A Dual-Loop Harness for Personalized and Reliable Robot Assistance](http://arxiv.org/abs/2610.21453v1) | Zixuan Wang, Yuqi Liu et al. | While generalist vision-language-action (VLA) models have enabled diverse open-world manipulation, real-world robotic assistance requires two critical capabilities they currently lack: personalization and reliable execution. First, executing an instruction like "make me a sandwich" requires grounding subtask decomposition in persistent user preferences rather than relying on a single generic solution. Second, reliable deployment demands explainable safety assurance that proactively prevents hazards and gracefully recovers from occasional physical failures. To address these gaps, we present CarryOn, a dual-loop harness designed to augment a frozen VLA policy. The planning loop maintains persistent user-level and scene-level context that is continually grounded against new observations and revised through interaction, enabling personalized task selection and execution grounding across conversations and changing scenes. The execution loop actively supervises policy rollouts, shielding actions from predicted hazardous contact. When execution fails, the system invokes reusable control-based recovery skills that restore task progress and are refined through reflection on earlier failures. On a real robot and in simulation, CarryOn dispatches preference-consistent subtasks across users and scenes, reduces the violation rate of an already safety-finetuned VLA from 52.6% to 6.1%, and increases recovery success from 13% to 80% when refined skills are reused. |

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



