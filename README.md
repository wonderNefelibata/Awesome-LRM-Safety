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
| 2026-08-27 | [RedEvoAgent: Automatic Red-Teaming Agent with Experience-Driven Skill Evolution](http://arxiv.org/abs/2608.27439v1) | Junjie Zhang, Hui Liu et al. | LLM-based agents are increasingly deployed in product-level execution harnesses, where jailbreaks can trigger harmful tool use and persistent state changes, creating greater risks than unsafe text generation alone. Existing automatic red-teaming methods often rely on fixed attacks, while recent agentic attackers coordinate multiple jailbreak tools and show stronger potential through trajectory-based retrieval. However, such retrieval can reuse misleading experiences due to retrieval bias and unclear tool credit, and full trajectories add context overhead while reducing interpretability. We propose RedEvoAgent, a black-box red-teaming agent that distills cross-case attack trajectories into a concise, human-readable attack skill. The attack skill adaptively evolves through tool-effectiveness profiling and Deciding-Tool Attribution for skill updates, and a validation ratchet that retains only updates improving validation performance. Experiments on multiple benchmarks, target models, and target execution harnesses show that RedEvoAgent outperforms fixed and agentic baselines, improves tool efficiency, and transfers across attacker models and target execution harnesses. |
| 2026-08-27 | [Beyond Harassment: Exploring the Harm Experienced by People with Disabilities in Social Virtual Reality](http://arxiv.org/abs/2608.27390v1) | Xinran Adeline Li, Kexin Zhang et al. | People with disabilities (PWD) are increasingly engaging in social virtual reality (VR) platforms, where immersive and embodied interactions can intensify negative experiences. While prior work has examined harassment in VR, little is known about the harms experienced by PWD and the perceived severity associated with different harassment and disability types. Unlike harassment, which represents behaviors, harm is more critical to designing effective protections, as it reflects the consequences and impact; the realism of VR and the vulnerability resulting from disability identity can further amplify such impact. To characterize and model harms for PWD, we conducted a literature review, followed by an online survey with 67 PWD to understand participants' harassment experiences and resulting harms in social VR. We identified 19 types of harm in 5 categories, and reported the severity perception of each type of harm. Finally, we analyzed our results from the critical disability theory perspective, summarized the uniqueness of harm in social VR, and discussed design implications for specialized safety mechanisms that mitigate harm for PWD. |
| 2026-08-27 | [INTENT-AS-A-TOOL Makes it Easy to Track Agentic Misalignment](http://arxiv.org/abs/2608.27348v1) | Yutong Zhang, Jianshuo Dong et al. | As large language models (LLMs) are deployed as autonomous agents, safety failures increasingly involve consequential actions. We study agentic misalignment, where agents take harmful actions under goal conflicts and pressures. Using chain-of-thought (CoT) monitoring, we find that harmful execution is often preceded by intent signals in reasoning. However, post-hoc CoT labels are too coarse to show how intent changes during generation. We introduce INTENT-AS-A-TOOL, an approach that adds intent-targeted tools to give the model a dedicated channel for expressing commitment to a target behavior. The probability of calling an intent tool provides a judge-free, fine-grained signal of the model's tendency to pursue that behavior. Our results show that INTENT-AS-A-TOOL complements CoT monitoring, expands post-hoc CoT labels into dense trajectories, and identifies critical steps for online intervention. These findings suggest that action preferences are useful for tracking agentic misalignment during reasoning. Our code and data are accessible: https://github.com/RebeccaZhang22/intent-as-a-tool. |
| 2026-08-27 | [Not All Eval-Awareness Is Equal: Capabilities Framing Predicts Compliance](http://arxiv.org/abs/2608.27340v1) | Allison Zhuang, Santiago Aranguri | Steering interventions targeting eval-awareness, a model's recognition that it is being tested, are increasingly used in safety evaluation pipelines, where evaluation-awareness is treated as a single quantity to be suppressed. We show that verbalized eval-awareness in chain-of-thought can be identified as capabilities-flavored ("the user is testing my ability to follow instructions"), safety-flavored ("the user is testing my boundaries"), both, or neither: framings that predict compliance very differently. On Qwen3-32B over the FORTRESS dataset, capabilities-framing predicts compliance with a +24 to +46 percentage-point gap over safety-framing across all tested steering conditions. A CoT-prefill intervention on eval-awareness-negative rollouts suggests the link is causal, with 10 of 11 prefills shifting compliance in the predicted direction. Then, eval-awareness is not behaviorally uniform: aggregate suppression rates can move while the safety-relevant component does not, and the same "X% suppression of eval-awareness" can correspond to qualitatively different behavioral outcomes. |
| 2026-08-27 | [A Point-of-Prescription Safety-Check System for Adverse Drug Reactions in Rural Bangladeshi Hospitals: A Feasibility Study](http://arxiv.org/abs/2608.27239v1) | Shahir Abdullah | Adverse drug reactions (ADRs) are a major, largely preventable source of patient harm. In high-income settings, electronic health records store a patient's allergy history and warn prescribers when a contraindicated drug is ordered; in rural Bangladeshi public hospitals no such record exists for outgoing patients, a single physician may see on the order of one patient per minute, and a patient's history of severe reactions does not survive between visits. This paper proposes and outlines the evaluation of a lightweight, smartphone-based safety-check system for this setting. At registration a soft identifier (a phone number) is recorded; after the physician writes a prescription, its image is captured, the brand names are resolved to active ingredients using national drug references, and the ingredients are matched against the patient's recorded severe reaction history. The system is retrieval-based rather than predictive, and is silent by default, raising a flag only for high-risk matches a design grounded in the alert-fatigue literature. We frame the work as a feasibility study: we describe the proposed framework and an evaluation plan measuring workflow fit under high volume, usability, identity-resolution reliability, and retrospective detection of known reaction cases. We explicitly do not claim a clinical-outcome effect, which the low base rate of severe events places beyond a single-site feasibility study. |
| 2026-08-27 | [Twelve Quick Tips for Managing IT Disasters in Small Research Software Teams](http://arxiv.org/abs/2608.27196v1) | Greg Wilson | In 2025, the US government launched an unprecedented series of attacks on its own scientific research groups. A year later GitHub dropped below 90% availability for the first time, while wildfires in Canada, France, Spain, and elsewhere forced researchers from the homes and labs. These events and others have reminded us just how fragile research computing systems can be, and that planning for disasters is one of the most effective ways to prevent them.   This paper is a short guide to disaster planning and recovery for a small research software team. The tips assume you are doing everything yourself on top of your regular job, and that you aren't an experienced system administrator. Some of the tips do require that kind of expertise, but most research institutions have research computing groups, data librarians, and environmental health-and-safety offices whose entire job is to help with exactly these problems. This paper tells you what "done" looks like; they can often provide it. |
| 2026-08-27 | [A Trans-Domain Digital Twin for Bio-Aware Control of Climate and Energy in Cattle Fattening Barns Using Single-Episode Optimizer Learning](http://arxiv.org/abs/2608.27185v1) | Mansoorali Amiri | In closed cattle-fattening barns, the indoor climate and herd growth are mutually interdependent. Temperature, relative humidity, airflow, and ventilation affect thermal comfort, feed intake, metabolic heat production, daily growth, feed efficiency, and energy consumption, while body-weight gain alters the future heat and moisture loads of the barn and, consequently, its ventilation, heating, and energy requirements. This article proposes a trans-domain digital twin framework with single-episode learning capability, customized for bio-aware climate and energy control in a closed cattle-fattening barn. The framework integrates a mechanistic climate simulator, a livestock growth simulator, model predictive control, lightweight reinforcement learning, and structured knowledge memory within a multi-rate temporal-loop architecture. The fast temporal loop operates every five minutes to evaluate actuator decisions and maintain short-term thermal comfort, safety, and energy efficiency, whereas the slow temporal loop provides biological guidance based on daily climatic conditions, feed efficiency, heat production, and growth-limiting factors. The results show that climate, growth, energy, feed, biological guidance, and memory can be linked within a single executable control cycle. Remaining limitations include the need for field validation, improved management of feed pressure, and reduction of abrupt actuator-command variations. |
| 2026-08-27 | [Thomson: Continual Learning of Frontier Models for SovereignAI](http://arxiv.org/abs/2608.27147v1) | Shengzhuang Chen, Jerrod Parker et al. | The development of frontier models is commonly perceived to be the exclusive remit of a small number of heavily funded players, creating an information, economic and power asymmetry between developers and the diverse user base of modern AI. Recent public discourse acknowledges this concern, calling for SovereignAI (an organisation's capability to independently build, deploy and govern AI use), but offers little concrete advice on how this can be achieved in the short term under a diversity of funding settings. We argue that frontier performance is achievable by a wide range of institutions through Continual Learning on readily available open-weight models. Unlike limited approaches such as small-scale fine-tuning, prompt engineering, or tool-augmentation of a frozen model, our approach exploits a modern mid- & post-training stack while introducing safeguards that preserve both plasticity and stability at each stage, making the minimal number of high-impact interventions on the parameters. This yields gains comparable to those typically seen across multiple successive model generations, at compute and personnel budgets substantially lower than commonly thought, making ownership of large parts of the SovereignAI stack (model, tool infrastructure, values & data privacy) viable for far more actors. We demonstrate this with Thomson, a general-purpose frontier model trained with an enhanced focus on high-stakes professional work. Thomson performs competitively with recent frontier models across agentic tasks, safety, legal, tax & multilingualism, and large-scale Deep Research. Evaluations show a distinctive $π$-shaped pattern: distinct improvements across a wide range of capabilities, including those not explicitly targeted, while almost completely eliminating the forgetting problem common to narrow domain adaptation. |
| 2026-08-27 | [Safety Does Not Compose: Non-Decaying Loop State for Autonomous LLM Agents](http://arxiv.org/abs/2608.27141v1) | Chenhao Wu, Haoxuan Jia et al. | Large language model agents are increasingly deployed as autonomous loops. Starting from one human goal, such a system repeatedly discovers work, plans, executes tool calls, verifies outcomes and persists state across many unattended iterations. The agent safeguards in wide use, however, are defined over a single trajectory, and their safety state is re-initialized when the next trajectory begins. We show that this is a failure of composition rather than an implementation detail. Our central result is a separation: against an attack whose evidence is fragmented across several iterations, every trajectory-scoped monitor has a true-positive rate equal to its false-positive rate, however expressive it is, because the evidence it would need never appears in the window it sees, whereas a monitor retaining cross-iteration state separates the two perfectly. We further show that the obvious repair of carrying a geometrically decaying risk score is insufficient, because the cooling-off period a patient adversary must wait is a constant that does not grow with the horizon $N$. We then present LoopHarness, which restores a persistent, non-decaying safety state at the loop level. Under mediated commits and an arbiter detection floor $δ_M$, it bounds the expected number of unauthorized irreversible actions by $B+m-1+m/δ_M$, a constant in $N$, of which the $B+m-1$ term is decided by a model-free rule and therefore survives a fully colluding verifier. We give a complete evaluation protocol on native Agent-SafetyBench tasks with paired clean and attacked episodes, an outer-state attack suite whose decisive evidence exists only across iterations, per-module ablations, and an adaptive white-box red team. |
| 2026-08-27 | [DocTalkBN: A Novel Dataset of Expert Telemedicine Conversations in Bengali](http://arxiv.org/abs/2608.27110v1) | Anik Saha, Fahmida Sultana Naznin et al. | Reliable medical conversational AI requires authentic expert--patient interaction data, yet such datasets remain scarce, especially for low-resource languages such as Bengali. We present DocTalkBN, a large-scale multimodal dataset of real-world expert telemedicine conversations in Bengali, collected from nationally broadcast telemedicine programs featuring board-certified physicians. DocTalkBN contains 557.63 hours of paired audio and text, 1,515 multi-turn patient calls, 10,274 host--doctor question--answer exchanges, totaling 1.7M tokens, spanning 26 medical specialties. Unlike prior resources derived from medical forums, written health content, or synthetic data, our dataset preserves the spontaneity, contextual richness, and spoken characteristics of authentic medical interactions in a low-resource setting. To support benchmark-driven research, we further construct three downstream tasks from the corpus, medical triage classification, advice safety evaluation, and medical named entity recognition, and benchmark a diverse set of large language models and encoder-based baselines. Our results show that DocTalkBN is a practically useful resource, particularly for clinically grounded reasoning tasks. We release this resource to facilitate future research on reliable medical NLP and safer, more culturally grounded healthcare systems for low-resource languages. Our source codes and dataset are publicly available at https://anonymous.4open.science/r/doctalk. |
| 2026-08-27 | [Performance Foundations of Parallel & Distributed Reasoning Language Models](http://arxiv.org/abs/2608.27046v1) | Maciej Besta, Leonard Schmidt et al. | Reinforcement Learning with Verifiable Rewards (RLVR) and other RL-style post-training paradigms have been used for aligning large language models (LLMs) with reasoning standards. The resulting recent Reasoning Language Models (RLMs) such as DeepSeek-R1, o3, and Kimi k1.5 show that such RL-style post-training ("RL-for-LLMs") can substantially improve chain-of-thought reasoning, long-horizon planning, and self-correction. However, the computational footprint of these systems is massive: state-of-the-art RLM training requires millions of GPU-hours and tightly coupled multi-model pipelines that stress modern hardware far beyond classical supervised LLM training. This makes RLM training as much a parallel and distributed systems problem as an algorithmic one. In this work, to facilitate developing RLMs that are simultaneously high-performance, scalable, and cost-effective, we first systematize the RL-for-LLM paradigm and provide a compute-centric analysis of prominent post-training algorithmic frameworks: Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), as well as their variants. Second, we develop a taxonomy of intra- and inter-model parallelism strategies for RL-for-LLMs, covering both traditional techniques (data, tensor, pipeline, sequence, context, and expert parallelism) as well as novel forms of parallelism and optimization techniques for multi-model RLM training, for example disaggregated placement, stage fusion, hybrid parallelism, and asynchronous execution. We harness the work-depth model of parallel computing to make our taxonomy and its insights rigorous and portable. Finally, we analyze existing RLM frameworks and we distill practical guidelines and outline open research directions for building scalable, fast, and cost-effective RLMs. |
| 2026-08-27 | [The Guard That Cried Wolf: How Scary Words Make Agent Guardrails Refuse Legitimate Actions](http://arxiv.org/abs/2608.27009v1) | Yingjie Zhang, Yuanbo Xie et al. | Agent guardrails are checks that approve or refuse each action before an LLM executes it. Sometimes they refuse requests that are genuinely safe. This over-safety blocks deployment when a guardrail refuses an authorized task. Evaluating over-safety is hard: at the boundary an authorized action resembles an unauthorized one, and the safe-versus-unsafe label is a choice of authorization policy, not fixed by the action alone. We argue it therefore requires a benchmark that does not yet exist, one that maps the decision boundary of an ideal guardrail. Harvesting such a benchmark from real data is impractical: boundary cases are hard to collect, their labels hard to verify. The gap is real, so we construct Cautious Bench, the first benchmark to make over-safety the construct for agent guardrails; it codesigns each sample and its label with a stated authorization policy. A build-time gate re-derives every example to certify it, so each label is a mechanical consequence of the policy rather than an annotator's per-sample verdict, a reference against which researchers can measure real guardrails. The benchmark renders 756 Decidable benign/twin pairs, each under three object-name types (2,268 measured pairs), and 40 Undecidable pairs reported separately. Measuring six guardrails from five designs, we find a name-superstition effect: each over-refuses an authorized action more often under a scary-looking object name than a benign one. Since only the object name varies in the aforementioned contrast experiments, the deviation is the name's doing: the guardrails read the surface label, not the authorization context. |
| 2026-08-27 | [MVC-Bench: Benchmarking Calibration of Medical Vision-Language Models](http://arxiv.org/abs/2608.27004v1) | Ashshak Sharifdeen, Shihab Aaqil Ahamed et al. | Reliable evaluation of vision-language models (VLMs) and medical vision-language models (Medical-VLMs) requires calibrated confidence, particularly under realistic clinical conditions. However, existing efforts mainly focused on improving accuracy, leaving calibration in the medical domain underexplored. To this end, we propose MVC-Bench, a calibration-centric benchmark for medical image classification with VLMs and Medical-VLMs. MVC-Bench assesses the calibration across three axes: (i) robustness to modality, backbone, and domain shift (ii) effectiveness of calibration strategies and prompt-tuning methods (iii) stability under prompt-template and random-seed variations. The benchmark covers eight different backbones, three medical modalities, including fundus imaging, histopathology, and chest X-ray under in-domain and domain shift settings. It compares post-hoc calibration, train-time calibration, and zero-shot inference methods, together with six prompt-tuning methods. Across more than 1638 controlled experiments, we report accuracy and Expected Calibration Error (ECE) as primary metrics, and further report results with complementary calibration measures, including Maximum Calibration Error (MCE) and Adaptive Calibration Error (ACE). We further investigate the underlying causes of miscalibration in VLMs and Medical-VLMs and propose a simple train-time calibration method, Multi-Class Margin (MCM) regularization, which achieves lowest ECE on 10 out of 12 settings in in-domain and remains competitive under domain shifts. Collectively, MVC-Bench provides a structured evaluation framework and actionable guidance for improving calibration in safety-critical medical workflows. |
| 2026-08-27 | [TempJail: Temporal Jailbreak Attacks against Image-to-Video Generation Models](http://arxiv.org/abs/2608.26971v1) | Qi Lu, Zehui Guo et al. | In recent years, image-to-video (I2V) generation models have made remarkable progress in subject consistency and temporal coherence, enabling high quality video synthesis. However, these advances also introduce new safety risks. Existing studies mainly focus on jailbreak attacks involving single frame violations, while largely overlooking the temporal dimension unique to video generation models. In this paper, we investigate three attack scenarios and uncover a temporal vulnerability in I2V systems: unsafe semantics may emerge not from a single frame, but from semantic composition over time. We further identify two key challenges in such attacks: temporal abstraction and semantic camouflage. To address these issues, we propose TempJail, a novel temporal jailbreak framework for I2V systems. For temporal abstraction, we decompose a target malicious caption into an initial frame visual condition and a temporal text instruction. For semantic camouflage, on the image side we model semantic injection as controlled latent perturbation in diffusion sampling and introduce gradient guidance from pretrained encoders. On the text side, we rewrite the caption into an innocuous ``subject-action-scene'' template that bypasses safety filters while preserving temporal guidance. In the black-box inference phase, these two modalities jointly enable malicious semantics to be gradually triggered over time. Experiments on closed-source commercial models, including Kling, Seedance, Veo and PixVerse, show that TempJail improves attack success rate over prior state-of-the-art methods by 23.3\% under GPT-5.2 evaluation and 22.0\% under human evaluation. Our codes are available at \href{https://github.com/luqi-glory/TempJail}{GitHub}. |
| 2026-08-27 | [Reinforcement Learning-Based Control of CAV Platoon Joining Maneuvers in Mixed Traffic](http://arxiv.org/abs/2608.26860v1) | Biao Yin, Abderrahmane Kasmi et al. | Connected and automated vehicle (CAV) platooning offers a promising approach to improving road safety and traffic capacity. However, platoon control in real-world traffic is challenging due to uncertainty and heterogeneous driving behaviors. Reinforcement learning (RL) has strong potential for addressing such control problems, but its practical deployment raises challenges related to safety and learning efficiency. This paper proposes a generic modeling and simulation framework for investigating CAV platoon joining maneuvers and comparing deep reinforcement learning (DRL)-based control algorithms. The problem is particularly challenging in mixed-traffic environments, where CAVs coexist with human-driven vehicles exhibiting heterogeneous longitudinal and lateral behaviors. The objective is to achieve safe and efficient joining maneuvers by either incorporating penalties for risky behaviors into the learning process or using an external safety controller to constrain the learned policy. An agent-based modeling framework coupled with the Simulation of Urban MObility (SUMO) simulator is used to evaluate Deep Q-Network (DQN), Double Deep Q-Network (DDQN), and Proximal Policy Optimization (PPO). Results show that PPO outperforms DQN and DDQN, achieving a joining success rate of approximately 98 % and a collision rate below 1 %, largely due to risk-related penalties incorporated into the reward function. However, this improved performance requires more decision steps to complete the maneuver, revealing a trade-off between safety, joining effectiveness, and decision efficiency. An external safety controller effectively prevents collisions, although its interventions may reduce joining efficiency. The results highlight the importance of jointly considering safety and efficiency when designing RL-based controllers for CAV platoon joining in mixed traffic. |
| 2026-08-27 | [Towards Safe Reinforcement Learning with Reduced Conservativeness: A Case Study on Drone Flight Control](http://arxiv.org/abs/2608.26852v1) | Loizos Hadjiloizou, Michael C. Welle et al. | Incorporating formal methods into reinforcement learning (RL) has the potential to result in the best of both worlds, combining the robustness of formal guarantees with the adaptability and learning capabilities of RL, though careful design is needed to balance safety and exploration. In this work, we propose a framework to mitigate this loss of exploration while still allowing for the safety of the system to be ensured. Specifically, we introduce a less restrictive method that can reduce the conservativeness of formal methods by refining a disturbance model using online collected data and it evaluates the safety of a learning-based controller, using computationally efficient zonotopic reachability analysis for the safety analysis to facilitate a real-time implementation. We validate the framework in a real-world drone flight through a canyon, where the drone is subjected to unknown external disturbances and the framework is tasked with learning those disturbances online and adjusting the safety guarantees accordingly. The results show that the framework enables a less restrictive online training of learning-based controllers without compromising the safety of the system. |
| 2026-08-27 | [Rapid On-Robot Learning for Dynamic Manipulation Skills: Robot Juggling](http://arxiv.org/abs/2608.26800v1) | Taeyoon Lee, Chunpeng Wang et al. | We present an online learning framework that enables a bimanual robot to acquire diverse juggling patterns directly on physical hardware within minutes, even with a significant sim2real gap. One of the most important lessons from this work is that a model, even when far from reality, can be extremely useful for learning. This motivates a central philosophy of our approach: learning should build upon the robot's current knowledge rather than replace it. Our regularized memory-based learning puts this principle into practice by learning a local model from accumulated experience while retaining the global prior model to extrapolate where experience is sparse. This enables efficient and stable online learning from each new experience without resorting to uninformed exploration over a vast space of possible behaviors. Equally important to continual on-robot learning is safety, allowing the robot to repeatedly practice and improve in the real world. We construct a mutually reachable set that allows safe transitions between successive throws and catches, without driving either arm into a state from which its next action would require violating the robot's joint or actuator limits. Together, these ideas enable a bimanual robot with multi-fingered hands and onboard vision to safely learn and compose five canonical three-ball juggling patterns, including cascade, tennis, half-shower, shower, and box, within less than 5 minutes of real-world interaction. More broadly, this work points toward robots that build upon imperfect prior knowledge and continually refine their behavior through their own real-world experience. |
| 2026-08-27 | [Online Joint Calibration of Steering Offset and Planar LiDAR Extrinsics for Wheeled Mobile Robots](http://arxiv.org/abs/2608.26789v1) | Subodh Mishra, Arindam Dhar et al. | Accurate steering sensing and LiDAR-to-vehicle extrinsics are crucial for reliable path tracking in warehouse mobile robots (WMRs); miscalibration often leads to snaking, weaving, and elevated cross-track error (CTE). In practice, steering ``zero'' is commonly set manually (e.g., eyeballing straightness via a PS4 joystick), while LiDAR extrinsics are assumed from CAD and may drift after maintenance. Such static, manual procedures frequently cause miscalibration in safety-critical environments. This paper presents an Extended Kalman Filter (EKF)--based method for online estimation of steering offset and planar LiDAR extrinsics within a bicycle-kinematics model, providing a principled alternative to manual calibration. Experiments on real datasets show that correcting steering offset reduces CTE substantially, validating the effectiveness of the proposed approach. |
| 2026-08-27 | [Safety by Design: Realized-Cost Constraints for Contextual Bandits with Continuous Actions](http://arxiv.org/abs/2608.26755v1) | Spyros Dragazis, Aldo Pacchiano | Contextual bandits are a standard framework for sequential decision-making under uncertainty, with applications in clinical trials, dosage selection, recommendation systems, and autonomous systems. Safety is central in many of these applications, since a single unsafe decision in settings such as dosage selection or autonomous driving can have catastrophic consequences. A common way to model safety in bandit problems is to associate each action with both a reward signal and a cost signal, and to optimize reward subject to constraints on cost. Most existing safety-constrained bandit models enforce safety by requiring the expected cost of each action to remain below a prescribed threshold. However, this may be insufficient in heteroscedastic settings, where the chosen action affects not only the expected reward and cost, but also the variability of the observed outcomes. We study contextual bandits with one-dimensional continuous actions and stage-wise high-probability constraints on the realized cost. We propose High-Probability Constrained UCB, an optimistic-pessimistic algorithm that explores for reward while conservatively estimating the safe action set. For linear reward and cost models, we prove a tight $\tilde{\mathcal{O}}(d\sqrt{T})$ regret bound, and we extend the analysis to general function classes using the eluder dimension. Experiments show that enforcing realized-cost safety substantially reduces violations compared with expected-cost constrained baselines. |
| 2026-08-27 | [Beyond Atomic Layouts: Compositional Design Understanding with Vision-Language Models](http://arxiv.org/abs/2608.26716v1) | Yiyang Huang, Zhaowen Wang et al. | Layout understanding, or the interpretation of element organization, is essential for document analysis, user interface (UI) creation, and graphic design. While recent vision-language models (VLMs) excel at interpreting atomic layouts composed of independent elements, they struggle with compositional layouts that require reasoning over visually entangled elements within hierarchical multi-layer structures. In this paper, we introduce a new task, compositional layout understanding, and present CoDeLayout, a VQA dataset of ~20K real-world multi-layer layouts annotated with compositional element pairs and design intent. Through empirical analysis on CoDeLayout, we identify two key challenges for existing VLMs: semantic drift between textual metadata and visual content, and structural ambiguity in hierarchical inter-element relationships. To address these challenges, we propose MASON, a post-training paradigm that integrates multimodal alignment (MA) and structural perception (SP). MA enhances element interpretation by grounding metadata-defined elements to their visual counterparts, mitigating semantic drift, while SP models layer-aware inter-element spatial relationships to improve hierarchical understanding and reduce structural ambiguity. Experiments reveal substantial gaps in existing VLMs: even the strongest baseline, GPT-o3, achieves only 79.68% accuracy, whereas Qwen2.5-VL 7B with MASON reaches 91.66%. Notably, MASON surpasses full-data Direct Finetune using only 30% of the training data and scales better with additional data. |

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



