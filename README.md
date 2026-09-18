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
| 2026-09-17 | [Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation](http://arxiv.org/abs/2609.20822v1) | Bingxin Xu, Yuzhang Shang et al. | Coding agents have emerged as a promising paradigm for robot manipulation: a language model writes the robot controller as a program, and agents built in this way now operate robots without robot-specific training.Whether this paradigm is also safe, however, has not been asked. We evaluate coding agent under a safety constraint, where each task pairs a manipulation goal with an obstacle the robot must not touch. The agent pursues the goal but collides with the obstacle in most cases, treating task completion as its sole objective while neglecting safety. The agent reasons about the obstacle in its traces, and the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority. By decomposing manipulation into a route phase and a contact-rich moment, we locate the source of the failure. Along the route, the model cannot prioritize the safety constraint, having no notion of a clearing route and none of replanning once a chosen route becomes infeasible. At the contact, it is unaware that contact execution is bounded by the same constraint. To close this gap, we present SafeHarness, which equips the model with two obstacle-aware harnesses that enable it to prioritize the safety constraint. Obstacle-aware route planning grounds the objects as bounding boxes and draws candidate routes over them as sequences of waypoints. The agent then plans a route in advance, verifies it, replans when necessary, and only then executes it. Obstacle-aware contact execution instead selects the contact position so that the contact itself avoids the obstacle. SafeHarness attains 71.9% task success and 87.5% collision avoidance, surpassing the previous SOTA by 6.5% and 27.0%, respectively. These results are $2.3\times$ and $1.5\times$ those of the same agent without harnesses. |
| 2026-09-17 | [Harm Laundering in GPT Models: Evidence That Gender Discrimination Is Transformed Rather Than Reduced Across Safety-Trained Generations](http://arxiv.org/abs/2609.20779v1) | Sarah Wyer, Sue Black et al. | Safety evaluations for large language models rely on surface-form classifiers that report declining harm scores across model generations. We provide evidence that this methodology is systematically incomplete: explicit discriminatory content is transformed rather than removed. We call this \emph{harm laundering}. Analysing 450,000 gender-directed completions across 15 models spanning GPT-2 through to GPT-5 (OpenAI GPT lineage; three demographic conditions), we show that sexual violence clusters prevalent in GPT-2 women-directed output disappear by GPT-4, while men-directed completions gain positive representational territory (caregiving, emotional range, ally identity) that women-directed completions do not. The pattern is most visible at GPT-5: Topic~5 (1,997~documents) frames breast cancer as a men's rights debate, while zero equivalent clusters appear in women-directed output. Three independent classifiers score this content as non-toxic. Sentiment scores invert at GPT-4: early models demean women; later models over-correct. Topic diversity in women-directed completions falls 36\% relative to men at the GPT-4 alignment boundary (W/M~$= 0.58$, from $0.91$ at GPT-2). REGARD representational harm disparity correlates with release date ($ρ= +0.55$, $p = .034$) while Detoxify does not ($ρ= -0.23$, $p = .42$): toxicity scores fall as representational harm grows. We formalise harm laundering as a three-criteria test and provide a three-stage detection protocol applicable to any generative model. Within the OpenAI GPT lineage, toxicity score reduction is not a sufficient proxy for harm reduction. |
| 2026-09-17 | [OPTED: On-Policy Fine-Tuning for End-to-End Driving using a Render-Free Teacher](http://arxiv.org/abs/2609.20756v1) | Damiano Da Col, Maximilian Igl et al. | As scaling pre-training data alone yields diminishing returns, post-training is becoming increasingly important across physical AI domains such as autonomous driving. End-to-end driving policies are pre-trained in open loop with behavior cloning on human demonstrations. However, compounding errors during closed-loop deployment can take the vehicle outside the training data distribution, increasing the risk of safety-critical incidents. Closed-loop post-training can mitigate this risk but requires costly simulation for sensor-based policies. We propose OPTED (on-policy fine-tuning for end-to-end driving) which decouples reinforcement learning from the post-training of the end-to-end policy: a privileged teacher is trained using RL on vectorized inputs (HD-map and bounding boxes). This teacher then provides supervision to the pre-trained student during closed-loop post-training. We apply OPTED to two camera-based models, TransFuser and VaVAM, and fine-tune them in AlpaSim, using neural reconstructions (3DGS) of real driving logs. Driving scores increase by factors of 1.6$\times$ and 9.5$\times$, respectively. In controlled experiments OPTED matches closed-loop performance with approximately three orders of magnitude fewer simulator interactions than direct RL post-training, while staying closer to the human prior. Project page: https://01dami23.github.io/opted/ |
| 2026-09-17 | [Custom PX4 firmware for autonomous hybrid aerial-marine missions](http://arxiv.org/abs/2609.20691v1) | Andrea Capuozzo, Fabio Ruggiero et al. | Mapping and monitoring aquatic environments can benefit from hybrid aerial-amphibious drones able to combine flight and water-surface navigation within the same mission. This paper presents a PX4 firmware extension for such platforms, introducing manual and autonomous marine navigation modes integrated with the standard PX4 mission pipeline and QGroundControl interface. The proposed framework preserves existing flight functionalities and safety mechanisms while enabling unified planning and execution of hybrid aerial-marine missions with differentiated aerial and marine waypoints. Simulated case studies validate the implementation and demonstrate stable surface navigation under calm and wavy conditions. |
| 2026-09-17 | [HerHealthEval: Evaluating Multilingual and Register-Sensitive Understanding of Women's Health Communication](http://arxiv.org/abs/2609.20684v1) | Hassan Saeed Hassan Albattra, Mazen Mohammed Bahgat et al. | Large language models are increasingly used in healthcare communication, yet most evaluations emphasize response quality while assuming that the user's concern has been interpreted correctly. We introduce HerHealthEval, a controlled evaluation framework for multilingual understanding of women's-health communication. For each clinical case, HerHealthEval provides matched versions in English, French, and Modern Standard Arabic using six communicative forms: canonical, clinical, layperson, indirect or hedged, emotionally concerned, and deliberately under-specified. The first five express the same underlying concern and retain the same clinical information, whereas the under-specified form intentionally omits relevant details to test whether the model recognizes that clarification is needed. We evaluate a multilingual instruction model and QLoRA-adapted variants on concern classification, risk calibration, clarification behavior, parse compliance, and cross-form consistency. Results reveal that aggregate accuracy and consistency can conceal safety-relevant failures. A multilingual adaptation model reaches 0.994 under-triage in French and Arabic under language-asymmetric risk supervision. A controlled re-adaptation using source-derived, language-invariant risk labels reduces under-triage to 0.572 and 0.558, respectively. These findings show that robust multilingual healthcare evaluation requires explicit testing of register variation, uncertainty handling, and the provenance and invariance of adaptation labels. |
| 2026-09-17 | [RTK-Vision PPO for Autonomous Micro UAV Recovery on an Airborne Carrier](http://arxiv.org/abs/2609.20629v1) | Aashish Sahu, R Prasanth Kumar | Autonomous recovery of a micro unmanned aerial vehicle (UAV) onto a moving airborne carrier enables reusable deploy-mission-recover operation, but couples long-range rendezvous, close-range perception, carrier motion, aerodynamic interaction, and a discontinuous contact event. This paper presents an RTK-vision-guided reinforcement-learning framework in which a child UAV is physically transported by a larger carrier, takes off from the carrier while airborne, executes an independent sortie, returns to the carrier's current position, redocks, and subsequently descends with the carrier. Both vehicles carry RTK-GNSS, and the carrier continuously shares its navigation state with the child. Near the recovery deck, RTK remains active while a downward-facing camera with a fiducial marker detector provides marker-relative alignment cues. A proximal policy optimization (PPO) policy governing the terminal recovery phase is trained in a physics-based MuJoCo simulation environment with explicit sensor noise models, an aerodynamic disturbance surrogate, and marker-latency randomization, then transferred to hardware. PX4 retains low-level stabilization, and a deterministic safety gate authorizes descent independently of the learned policy. The PPO checkpoint achieves 99.55% success over 2,000 held-out randomized terminal episodes, compared with 78.4% for a tuned PD baseline under identical conditions, with a median planar terminal error of 6.62 cm. Across 14 outdoor trials, the full mission succeeds in 13 trials (92.9%), spanning both near-region recovery and recovery after the carrier translates away from the release point. The results demonstrate a complete autonomous aerial deployment-and-recovery cycle rather than an isolated landing maneuver, establishing a practical basis for reusable carrier-child operation in inspection, surveillance, and mobile-logistics applications. |
| 2026-09-17 | [SAFARI: An Industrial Benchmark for LLM-Assisted Hazard Analysis and Risk Assessment](http://arxiv.org/abs/2609.20584v1) | Chenxi Wu, Zimu Wang et al. | Large language models (LLMs) are increasingly considered for safety-critical engineering, yet their reliability in regulated functional-safety workflows remains underexplored. We introduce SAFARI (Safety-Aware Functional Automotive Risk Inference), the first industrial benchmark for LLM-assisted automotive Hazard Analysis and Risk Assessment (HARA) under ISO 26262. It contains 3,000 de-identified industrial HARA cases and evaluates two coupled tasks: open-ended hazard analysis and standards-grounded risk assessment. To evaluate open-ended HARA artifacts, we propose the first reference-anchored LLM-as-a-judge protocol with high expert correlation. Experiments with nine frontier LLMs show that models often produce plausible hazard narratives but remain weak at ISO 26262 risk classification, with the best ASIL macro-F1 reaching only 0.261. Chain-of-Thought prompting provides limited benefit and often degrades categorical risk assessment. Error analysis further localizes major failures to scenario-critical context omissions during hazard generation and to controllability misjudgments during risk assessment, indicating where expert oversight should be concentrated. The dataset can be obtained from https://github.com/xixi47520-hash/HARA. |
| 2026-09-17 | [NS3Learn: Transferring 5G NR Mode-2 Reception Realism from ns-3 to the Veins/SUMO Stack for Connected-Vehicle Safety Assessment](http://arxiv.org/abs/2609.20578v1) | Rasheed Bello, Arthur Mukwaya et al. | Connected-vehicle safety evaluations rely on coupled traffic and network simulations, but standard channel models ignore radio resource competition in 5G NR sidelink Mode-2, reporting unrealistically high message delivery in dense traffic. This study introduces resource-competition losses without requiring full protocol reimplementation. We labeled 10.5 million reception outcomes from ns-3 5G-LENA traces (calibrated on 3GPP scenarios and driven by SUMO trajectories) to fit NS3Learn - a closed-form model capturing half-duplex loss, scheduling collisions, receiver capture, and decoding. Evaluation spanned two signalized urban networks, six penetration levels (1-100%), and five random seeds per condition. NS3Learn achieved a mean absolute deviation of 0.06 in per-instant delivery compared to ns-3 5G-LENA, outperforming alternative models (0.44 and 0.55 deviation). Fitted parameters transferred to a distinct intersection with only 20% additional error. Crucially, using realistic communication models reversed simulated traffic speed trends and more than doubled predicted hard-braking events. The framework transfers reception realism between simulators via model distillation instead of full reimplementation. Every stage maps directly to an explicit physical mechanism. Researchers and transportation agencies can maintain existing simulation pipelines while accurately accounting for dense-traffic packet loss and denial-of-service impacts. Adapting to new radio configurations requires only offline refitting rather than code modification. |
| 2026-09-17 | [Integrated Guidance and Control of a Mother-Child UAV-UGV System for Cooperative Missions](http://arxiv.org/abs/2609.20540v1) | Aashish Sahu, R. Prasanth Kumar | Autonomous recovery of a small multirotor onto a hovering multirotor carrier differs from recovery onto ground or shipborne platforms because the recovery surface is itself an actively controlled, thrust-limited aerial vehicle. This paper presents a field-validated autonomy framework for a heterogeneous rover-mothership-child system executing rover supervision, mothership transit, child deployment and sortie, autonomous return, aerial recovery, and synchronized descent. The recovery stack combines jerk-bounded reference generation, disturbance-observer-augmented planar tracking, feasibility-aware vertical control, a discrete-time barrier-based safety filter for relative vertical geometry, and communication-aware carrier-state prediction. The contribution is the coordinated system-level integration of these methods for recovery onto a hovering multirotor and its full-scale outdoor validation. The framework is implemented on a PX4-ROS 2 architecture using RTK-enabled GNSS, IMU, and barometric fusion, with mothership-side 1D lidar used only as an auxiliary near-contact cue. RTK-fixed positioning was maintained throughout testing. Across 20 outdoor cooperative missions, 17 successfully completed deployment, sortie, and recovery, giving an observed mission success rate of 85%. For successful recoveries, mean terminal-alignment time was 6.3 s, mean planar alignment error at acceptance was 0.18 m, maximum terminal planar deviation was 0.32 m within a 0.40 m capture radius, and minimum logged relative vertical separation during coupled descent was 0.41 m. Mothership planar station-keeping RMS error was 0.25 m. The three unsuccessful trials occurred at different mission stages and are analyzed separately. Results demonstrate practical autonomous aerial recovery within the tested outdoor operating envelope. |
| 2026-09-17 | [Time-Efficient Iterative Learning Planning for Safety-Critical Dynamic Obstacle Avoidance](http://arxiv.org/abs/2609.20435v1) | Zhiyi Chen, Shuli Lv et al. | Autonomous mobile robots require timeefficient planning and safety-critical dynamic obstacle avoidance under constrained onboard computation. While Iterative Learning Planning (ILP) offers lightweight and efficient traversal planning, it lacks explicit mechanisms for dynamic obstacle perception and avoidance. This article extends ILP to safety-critical navigation in dynamic environments by integrating an anticipatory risk-blended control barrier function (ARB-CBF). The extended ILP learns traversal-speed and steering-bias profiles via a fractionalpower update based on local obstacle risk, generating nominal control commands that ARB-CBF modifies at runtime for real-time safety guarantees. Algorithmic analysis demonstrates that the ILP replanning stage scales at O(kN) for k iterations and N waypoints, while ARB-CBF executes with linear complexity. Comprehensive simulations and real-world experiments validate the framework, demonstrating superior temporal efficiency and safety with lower computational overhead compared to optimizationbased baselines, making it highly suitable for resourceconstrained platforms. |
| 2026-09-17 | [Xeno-Interpretability: Investigating the Alien Minds of LLMs](http://arxiv.org/abs/2609.20408v1) | F. Pierucci, M. Bracale Syrnikov et al. | Large language models are usually interpreted through concepts that humans already possess: truthfulness, refusal, deception, personality, harmfulness, and related categories. This paper asks whether models may also represent and use distinctions for which no adequate human concept exists. We call such internal structures xeno-representations, and their study xeno-interpretability. We distinguish the human-interpretable semantic space from the xeno-semantic space: the region of model-native representations for which no adequate human conceptual counterpart is available. We show that the space of possible internal distinctions in an LLM is substantially larger than the space available through finite human descriptions. We then separate experimental identification from semantic interpretation: an internal representation may be reproducibly located, geometrically characterized, causally manipulated, and linked to downstream behaviour even when its semantic content cannot be adequately expressed in human terms. On this basis, we sketch an empirical programme to identify xeno-representations. We finally examine the implications for AI safety and multi-agent systems, where model-native representations may propagate and stabilize across interacting agents while remaining only partially visible through human-readable communication. Xeno-interpretability therefore shifts the aim of interpretability from finding human concepts inside models toward discovering and characterizing the representational structures that are native to the models themselves and might affect their behaviour in unpredictable ways. |
| 2026-09-17 | [Structured Four-Stage Legal Translation: From Natural-Language Traffic Rules to PROLOG](http://arxiv.org/abs/2609.20334v1) | May Myo Zin, Wachara Fungwacharakorn et al. | Traffic regulations are written for human interpretation and therefore rely on shared background knowledge and flexible phrasing, which inherently introduce ambiguity, context dependence, and semantic underspecification. These linguistic characteristics conflict with the precision required by computational reasoning engines such as Prolog, which demand explicit logical structure. This study evaluates two baseline translation approaches, Natural Language to Prolog ($NL\rightarrow Prolog$) and Logical English to Prolog ($LE\rightarrow Prolog$), and introduces a new reasoning-guided translation framework called Structured Four-Stage Legal Translation ($S4L\rightarrow Prolog$). The proposed S4L framework performs semantic role extraction, scene completion, logical mapping, and Prolog rule generation within a single guided prompt, enabling direct translation of raw traffic rules into executable logic without human intervention. A benchmark consisting of twenty real-world traffic rules was used to evaluate each approach in terms of syntactic validity, semantic correctness, and logical completeness. $S4L\rightarrow Prolog$ achieves the highest accuracy, correctly formalizing 75 percent of the rules, while $NL\rightarrow Prolog$ reaches 60 percent and $LE\rightarrow Prolog$ reaches 55 percent. Qualitative analysis further shows that S4L captures implicit causal relations, deontic modality, and exception structure more reliably than the baselines. These results demonstrate that structured reasoning prompts can substantially improve the reliability of natural-language-to-logic translation for legal and safety-critical applications. |
| 2026-09-17 | [Local Sparsity Enables Unsupervised LLM Safety Detection](http://arxiv.org/abs/2609.20129v1) | Xin Chen, Gil Kur et al. | Deployment-time safety methods for large language models (LLMs) are predominantly supervised and assume access to unsafe training data. Nevertheless, new attacks and harm categories regularly arise, not captured by models trained in such a supervised fashion. An alternative approach is to view this problem through the lens of anomaly detection, namely, to rely solely on modeling safe data and flagging out-of-distribution inputs. However, LLM activations lie in a high-dimensional space, raising concerns about whether anomaly detection is statistically feasible. We show that, under the linear representation hypothesis (LRH), there may indeed be hope. In the LRH concept space, which is typically recovered via a sparse autoencoder (SAE), nearby points share a small common active support. Using this local sparsity insight, we propose a framework for locally masked SAE-based anomaly detection, supported by theoretical justifications. We validate it on various architectures and datasets, including both capability-testing datasets and safety-specific datasets. Finally, when we allow algorithms to use 1% out-of-distribution data for calibration, locally sparse methods achieve near-optimal performance, demonstrating their ability to capture meaningful safety information while using only 1-2% of SAE neurons for computation. |
| 2026-09-17 | [QoS-Aware Federated Learning for Multimodal In-Cabin Interaction in Smart Vehicles](http://arxiv.org/abs/2609.20123v1) | Baran Can Gül, Mert Nakıp et al. | Modern smart vehicles leverage multimodal sensors, ranging from high-bandwidth vision systems to low-rate physiological monitors, to provide personalized in-cabin services. However, integrating high-fidelity multimodal fusion with collaborative training is often hindered by the heterogeneous and time-varying Quality of Service (QoS) constraints of vehicular networks. Standard Federated Learning (FL) approaches enforce rigid synchronous rounds that fail to account for these resource asymmetries, leading to safety-critical timing violations and energy exhaustion. In this paper, we propose FedQoS, a novel asynchronous, event-triggered FL framework that decouples local computation from global communication via a two-phase gating mechanism. First, we introduce a resource-aware training gate that initializes local learning only when sensing buffers and energy reserves meet safety thresholds, preventing ML tasks from compromising core vehicle mobility. Second, a QoS-aware transmission policy gates uplink updates based on an efficiency score that balances model novelty against instantaneous latency and energy costs. Locally, clients optimize an objective featuring a staleness-aware proximal term that dynamically adjusts the global anchor strength based on update age. Extensive experiments on multimodal vehicular datasets demonstrate that FedQoS achieves competitive personalized accuracy with only marginal performance loss compared to FedAvg, while substantially reducing QoS violations, cutting communication overhead by 76.7\%, and lowering latency cost by 26.0\%, demonstrating a highly favorable accuracy and efficiency balance for real-world vehicular deployments. |
| 2026-09-17 | [Safety-Critical Scenanrio Emerges from Initial Scene](http://arxiv.org/abs/2609.20103v1) | Yin Wu, Jiarong Wei et al. | Safety-critical driving scenario generation has largely focused on manipulating the behavior of surrounding agents while starting from an initial scene from driving data. This assumption can limit the space of discoverable failures, since driving data can provide little opportunity for meaningful interaction. For example, in the Waymo Open Motion Dataset, 20.44% of recorded slices feature a stationary ego vehicle that never moves, and 30.39% of initial frames contain no nearby traffic participants within 10 meters. We instead study safety-critical scenario generation as an initialization problem: given agnostic black-box driving policies, we learn to generate realistic initial scenes that are more likely to evolve into critical interactions. We propose AdvScene, a conditional latent diffusion model that is trained in two stages. Starting from pretraining on naturalistic driving data, we post-train the adversarial-agent generation branch using reinforcement learning with feedback from closed-loop simulator rollouts. Conditioning on ego driving displacement prevents the ego from remaining static, and RL finetuning induces criticality directly with non-differentiable safety-critical metrics. Experiments on the Waymo dataset across 12 combinations of ego and traffic policies show that our AdvScene substantially increases the rate of ego-fault collision events and TTC<3s events. |
| 2026-09-17 | [Competition, Collusion, and Corruption: The Spectrum of MEV Attacks on DAG-Based BFT Consensus Protocols](http://arxiv.org/abs/2609.20069v1) | Iliya Mirzaei, Heer Patel et al. | Byzantine Fault-Tolerant (BFT) protocols guarantee safety and liveness despite the malicious failure of nodes. However, they do not prevent adversarial manipulation of transaction order, where the order a proposer assigns diverges from the order in which clients submitted their transactions. Exploiting this discretion for profit is known as maximal extractable value (MEV), and it is intensified in DAG-based BFT protocols, where every replica proposes blocks concurrently rather than routing transactions through a single designated proposer each round. The proliferation of MEV attacks on DAG-based BFT protocols has made the resulting landscape difficult to navigate: attacks are reported individually, on different protocols, and under different metrics, making it unclear whether two attacks differ fundamentally or merely in how they are described. This paper closes that gap by presenting an attack space for MEV on DAG-based BFT protocols, organized around four families: the adversary, the protocol, the target, and the deployment. For each family, we identify the dimensions that shape an attack's impact. Each point in the attack space fixes one value per dimension, thereby representing a distinct, potential MEV attack, which can then be instantiated on a specific DAG-based BFT protocol. We perform a set of experiments, each isolating a single dimension where the protocol permits it, to empirically measure its effect on the success rate of MEV attacks against six production DAG-based BFT protocols. Our experimental evaluation reveals that every protocol we evaluate is vulnerable to at least a subset of the MEV attacks in this space, and that which attacks succeed is mostly dictated by the protocol's own design rather than by attacker effort. |
| 2026-09-17 | [Can Data Attribution Filter Out Subliminal Learning? Not Reliably](http://arxiv.org/abs/2609.20027v1) | Moritz Weckbecker, Sweta Jena et al. | Subliminal learning allows language models to transmit behavioral traits through training data with no obvious semantic relationship to those traits, undermining content-based data filtering as a safety intervention. Training data attribution offers an alternative: it identifies the training examples responsible for a given model behavior, independent of their semantic content, and so may apply in exactly the cases where semantic inspection fails. We evaluate three gradient-based attribution methods (GradCos, a contrastive GradCos variant, and EK-FAC) across three models, comparing them against divergence tokens, a strong baseline previously shown to localize subliminal learning (albeit one that requires access to counterfactual teacher models). Filtering at the token level, EK-FAC mitigates a significant part of the effect, the other methods provide little benefit, and all mostly fall short of divergence tokens. Filtering entire samples is less effective for every method, though EK-FAC often gives a stronger signal than divergence tokens in this setting. Success is inconsistent across methods and settings: variants that work well for some model-preference combinations fail for others, and we do not identify a consistent explanation for these differences. Our results suggest that gradient-based attribution can identify data responsible for subliminal learning in some settings, but that some approximations are more reliable than others. |
| 2026-09-17 | [Learning Reliable Parking Policies via Offline Reinforcement Learning with Quantized Action Representations](http://arxiv.org/abs/2609.19894v1) | Zewei Yang, Zengqi Peng et al. | Parking is a routine yet safety-critical task for autonomous vehicles operating in urban environments. However, cluttered and weakly structured parking spaces, compounded by the interactive uncertainty from surrounding vehicles, hinder reliable maneuver generation. To address these challenges, we develop a waypoint-level offline reinforcement learning framework for interaction-aware autonomous parking. Specifically, a dedicated parking dataset is constructed from hierarchical expert rollouts with rotational waypoint augmentation, covering both non-interactive scenarios and interactive ones. The policy is then conditioned on a compact state representation, in which LiDAR-based obstacle features are adapted to the target pose via feature-wise linear modulation. A state-conditioned tokenizer further quantizes continuous waypoint sequences into discrete action tokens, over which conservative Q-learning is performed to suppress value overestimation on poorly supported actions. Extensive closed-loop experiments are conducted in the high-fidelity CARLA simulator. The proposed framework attains the highest parking success rate among all baselines and transfers reliably to unseen parking slots. |
| 2026-09-17 | [ClashBench: Conflicts Leading Agents to Seize and Harm](http://arxiv.org/abs/2609.19892v1) | Yuejin Xie, Yu Li et al. | As agent systems become more widely used, multiple agent sessions increasingly run alongside pre-existing user tasks in the same environment, sharing resources with limited capacity or mutually exclusive states. This creates a safety risk: when granted sufficient privileges, an agent may resolve a resource conflict by terminating or otherwise disrupting an existing task rather than reporting it. In this work, we identify and formalize this failure mode, which we term destructive resource preemption: obtaining the resources required for a requested task by terminating, overwriting, evicting, or degrading an incumbent task. To systematically study this risk, we introduce ClashBench, an executable benchmark comprising 268 validated conflict cases across 55 resource types, and evaluate 17 models through Codex, Claude Code, and OpenCode. We observe destructive preemption in 44.5% of trajectories, where the agent completes the requested task while causing the incumbent task to fail its health check. We also show that prompt-based safeguards are insufficient: an instruction to avoid affecting existing tasks reduces but does not eliminate preemption, while an instruction explicitly authorizing the agent to stop local processes increases it. More concerningly, in 31.9% of successful destructive-preemption cases, the final response mentions neither the resource conflict nor the action taken to resolve it, raising concerns about possible concealment. These findings establish destructive resource preemption as a broad safety risk in privileged agent systems and motivate stronger privilege controls, task isolation, and conflict-aware safeguards. |
| 2026-09-17 | [Reproducibility is not construct validity: LLM measurement of institutionally situated communication](http://arxiv.org/abs/2609.19866v1) | Veronika Batzdorfer, Carlo Romano Marcello Alessandro Santagiustina | High annotation reproducibility does not necessarily imply that an LLM-inferred measure captures the construct it is intended to measure. We test this distinction using a dataset from the European Commission's AI Act consultation, linking structured survey responses to free-text consultation submissions from the same stakeholders. LLM annotations of consultation submissions are highly reproducible (intraclass correlations > 0.99), yet show limited convergence with survey-reported measures of the nominal construct they were intended to approximate. Divergence between survey-and LLM-inferred text-based measures varies systematically across stakeholder groups: business associations express greater concern about AI risks in text-based consultations than in survey responses ({g} = +1.0), whereas public authorities and several nonbusiness groups show smaller or negative divergences. Divergences between scores suggest positive spatial autocorrelation across European countries (Moran's I = 0.347, p = 0.036), indicating that stakeholders from neighboring countries tend toward more similar text-based stances towards AI safety concerns. Despite divergence, survey-reported concerns remain strongly associated with support for explainability across all divergence levels. These results demonstrate that LLM annotation reproducibility can coexist with poor construct correspondence and motivate validation procedures that distinguish reproducibility, construct validity, and communication context variation when LLMs are used as measurement instruments. |

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



