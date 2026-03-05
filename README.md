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
| 2026-03-04 | [Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks](http://arxiv.org/abs/2603.04364v1) | Haoyu Liu, Dingcheng Li et al. | Multimodal web agents that process both screenshots and accessibility trees are increasingly deployed to interact with web interfaces, yet their dual-stream architecture opens an underexplored attack surface: an adversary who injects content into the webpage DOM simultaneously corrupts both observation channels with a consistent deceptive narrative. Our vulnerability analysis on MiniWob++ reveals that attacks including a visual component far outperform text-only injections, exposing critical gaps in text-centric VLM safety training. Motivated by this finding, we propose Dual-Modality Multi-Stage Adversarial Safety Training (DMAST), a framework that formalizes the agent-attacker interaction as a two-player zero-sum Markov game and co-trains both players through a three-stage pipeline: (1) imitation learning from a strong teacher model, (2) oracle-guided supervised fine-tuning that uses a novel zero-acknowledgment strategy to instill task-focused reasoning under adversarial noise, and (3) adversarial reinforcement learning via Group Relative Policy Optimization (GRPO) self-play. On out-of-distribution tasks, DMAST substantially mitigates adversarial risks while simultaneously doubling task completion efficiency. Our approach significantly outperforms established training-based and prompt-based defenses, demonstrating genuine co-evolutionary progress and robust generalization to complex, unseen environments. |
| 2026-03-04 | [Efficient Refusal Ablation in LLM through Optimal Transport](http://arxiv.org/abs/2603.04355v1) | Geraldin Nanfack, Eugene Belilovsky et al. | Safety-aligned language models refuse harmful requests through learned refusal behaviors encoded in their internal representations. Recent activation-based jailbreaking methods circumvent these safety mechanisms by applying orthogonal projections to remove refusal directions, but these approaches treat refusal as a one-dimensional phenomenon and ignore the rich distributional structure of model activations. We introduce a principled framework based on optimal transport theory that transforms the entire distribution of harmful activations to match harmless ones. By combining PCA with closed-form Gaussian optimal transport, we achieve efficient computation in high-dimensional representation spaces while preserving essential geometric structure. Across six models (Llama-2, Llama-3.1, Qwen-2.5; 7B-32B parameters), our method achieves up to 11% higher attack success rates than state-of-the-art baselines while maintaining comparable perplexity, demonstrating superior preservation of model capabilities. Critically, we discover that layer-selective intervention (applying optimal transport to 1-2 carefully chosen layers at approximately 40-60% network depth) substantially outperforms full-network interventions, revealing that refusal mechanisms may be localized rather than distributed. Our analysis provides new insights into the geometric structure of safety representations and suggests that current alignment methods may be vulnerable to distributional attacks beyond simple direction removal. |
| 2026-03-04 | [Gaussian Mixture-Based Inverse Perception Contract for Uncertainty-Aware Robot Navigation](http://arxiv.org/abs/2603.04329v1) | Bingyao Du, Joonkyung Kim et al. | Reliable navigation in cluttered environments requires perception outputs that are not only accurate but also equipped with uncertainty sets suitable for safe control. An inverse perception contract (IPC) provides such a connection by mapping perceptual estimates to sets that contain the ground truth with high confidence. Existing IPC formulations, however, instantiate uncertainty as a single ellipsoidal set and rely on deterministic trust scores to guide robot motion. Such a representation cannot capture the multi-modal and irregular structure of fine-grained perception errors, often resulting in over-conservative sets and degraded navigation performance. In this work, we introduce Gaussian Mixture-based Inverse Perception Contract (GM-IPC), which extends IPC to represent uncertainty with unions of ellipsoidal confidence sets derived from Gaussian mixture models. This design moves beyond deterministic single-set abstractions, enabling fine-grained, multi-modal, and non-convex error structures to be captured with formal guarantees. A learning framework is presented that trains GM-IPC to account for probabilistic inclusion, distribution matching, and empty-space penalties, ensuring both validity and compactness of the predicted sets. We further show that the resulting uncertainty characterizations can be leveraged in downstream planning frameworks for real-time safe navigation, enabling less conservative and more adaptive robot motion while preserving safety in a probabilistic manner. |
| 2026-03-04 | [Scalable Evaluation of the Realism of Synthetic Environmental Augmentations in Images](http://arxiv.org/abs/2603.04325v1) | Damian J. Ruck, Paul Vautravers et al. | Evaluation of AI systems often requires synthetic test cases, particularly for rare or safety-critical conditions that are difficult to observe in operational data. Generative AI offers a promising approach for producing such data through controllable image editing, but its usefulness depends on whether the resulting images are sufficiently realistic to support meaningful evaluation.   We present a scalable framework for assessing the realism of synthetic image-editing methods and apply it to the task of adding environmental conditions-fog, rain, snow, and nighttime-to car-mounted camera images. Using 40 clear-day images, we compare rule-based augmentation libraries with generative AI image-editing models. Realism is evaluated using two complementary automated metrics: a vision-language model (VLM) jury for perceptual realism assessment, and embedding-based distributional analysis to measure similarity to genuine adverse-condition imagery.   Generative AI methods substantially outperform rule-based approaches, with the best generative method achieving approximately 3.6 times the acceptance rate of the best rule-based method. Performance varies across conditions: fog proves easiest to simulate, while nighttime transformations remain challenging. Notably, the VLM jury assigns imperfect acceptance even to real adverse-condition imagery, establishing practical ceilings against which synthetic methods can be judged. By this standard, leading generative methods match or exceed real-image performance for most conditions.   These results suggest that modern generative image-editing models can enable scalable generation of realistic adverse-condition imagery for evaluation pipelines. Our framework therefore provides a practical approach for scalable realism evaluation, though validation against human studies remains an important direction for future work. |
| 2026-03-04 | [LUMINA: Foundation Models for Topology Transferable ACOPF](http://arxiv.org/abs/2603.04300v1) | Yijiang Li, Zeeshan Memon et al. | Foundation models in general promise to accelerate scientific computation by learning reusable representations across problem instances, yet constrained scientific systems, where predictions must satisfy physical laws and safety limits, pose unique challenges that stress conventional training paradigms. We derive design principles for constrained scientific foundation models through systematic investigation of AC optimal power flow (ACOPF), a representative optimization problem in power grid operations where power balance equations and operational constraints are non-negotiable. Through controlled experiments spanning architectures, training objectives, and system diversity, we extract three empirically grounded principles governing scientific foundation model design. These principles characterize three design trade-offs: learning physics-invariant representations while respecting system-specific constraints, optimizing accuracy while ensuring constraint satisfaction, and ensuring reliability in high-impact operating regimes. We present the LUMINA framework, including data processing and training pipelines to support reproducible research on physics-informed, feasibility-aware foundation models across scientific applications. |
| 2026-03-04 | [VANGUARD: Vehicle-Anchored Ground Sample Distance Estimation for UAVs in GPS-Denied Environments](http://arxiv.org/abs/2603.04277v1) | Yifei Chen, Xupeng Chen et al. | Autonomous aerial robots operating in GPS-denied or communication-degraded environments frequently lose access to camera metadata and telemetry, leaving onboard perception systems unable to recover the absolute metric scale of the scene. As LLM/VLM-based planners are increasingly adopted as high-level agents for embodied systems, their ability to reason about physical dimensions becomes safety-critical -- yet our experiments show that five state-of-the-art VLMs suffer from spatial scale hallucinations, with median area estimation errors exceeding 50%. We propose VANGUARD, a lightweight, deterministic Geometric Perception Skill designed as a callable tool that any LLM-based agent can invoke to recover Ground Sample Distance (GSD) from ubiquitous environmental anchors: small vehicles detected via oriented bounding boxes, whose modal pixel length is robustly estimated through kernel density estimation and converted to GSD using a pre-calibrated reference length. The tool returns both a GSD estimate and a composite confidence score, enabling the calling agent to autonomously decide whether to trust the measurement or fall back to alternative strategies. On the DOTA~v1.5 benchmark, VANGUARD achieves 6.87% median GSD error on 306~images. Integrated with SAM-based segmentation for downstream area measurement, the pipeline yields 19.7% median error on a 100-entry benchmark -- with 2.6x lower category dependence and 4x fewer catastrophic failures than the best VLM baseline -- demonstrating that equipping agents with deterministic geometric tools is essential for safe autonomous spatial reasoning. |
| 2026-03-04 | [GSeg3D: A High-Precision Grid-Based Algorithm for Safety-Critical Ground Segmentation in LiDAR Point Clouds](http://arxiv.org/abs/2603.04208v1) | Muhammad Haider Khan Lodhi, Christoph Hertzberg | Ground segmentation in point cloud data is the process of separating ground points from non-ground points. This task is fundamental for perception in autonomous driving and robotics, where safety and reliable operation depend on the precise detection of obstacles and navigable surfaces. Existing methods often fall short of the high precision required in safety-critical environments, leading to false detections that can compromise decision-making. In this work, we present a ground segmentation approach designed to deliver consistently high precision, supporting the stringent requirements of autonomous vehicles and robotic systems operating in real-world, safety-critical scenarios. |
| 2026-03-04 | [A Multi-Agent Framework for Interpreting Multivariate Physiological Time Series](http://arxiv.org/abs/2603.04142v1) | Davide Gabrielli, Paola Velardi et al. | Continuous physiological monitoring is central to emergency care, yet deploying trustworthy AI is challenging. While LLMs can translate complex physiological signals into clinical narratives, it is unclear how agentic systems perform relative to zero-shot inference. To address these questions, we present Vivaldi, a role-structured multi-agent system that explains multivariate physiological time series. Due to regulatory constraints that preclude live deployment, we instantiate Vivaldi in a controlled, clinical pilot to a small, highly qualified cohort of emergency medicine experts, whose evaluations reveal a context-dependent picture that contrasts with prevailing assumptions that agentic reasoning uniformly improves performance. Our experiments show that agentic pipelines substantially benefit non-thinking and medically fine-tuned models, improving expert-rated explanation justification and relevance by +6.9 and +9.7 points, respectively. Contrarily, for thinking models, agentic orchestration often degrades explanation quality, including a 14-point drop in relevance, while improving diagnostic precision (ESI F1 +3.6). We also find that explicit tool-based computation is decisive for codifiable clinical metrics, whereas subjective targets, such as pain scores and length of stay, show limited or inconsistent changes. Expert evaluation further indicates that gains in clinical utility depend on visualization conventions, with medically specialized models achieving the most favorable trade-offs between utility and clarity. Together, these findings show that the value of agentic AI lies in the selective externalization of computation and structure rather than in maximal reasoning complexity, and highlight concrete design trade-offs and learned lessons, broadly applicable to explainable AI in safety-critical healthcare settings. |
| 2026-03-04 | [FINEST: Improving LLM Responses to Sensitive Topics Through Fine-Grained Evaluation](http://arxiv.org/abs/2603.04123v1) | Juhyun Oh, Nayeon Lee et al. | Large Language Models (LLMs) often generate overly cautious and vague responses on sensitive topics, sacrificing helpfulness for safety. Existing evaluation frameworks lack systematic methods to identify and address specific weaknesses in responses to sensitive topics, making it difficult to improve both safety and helpfulness simultaneously. To address this, we introduce FINEST, a FINE-grained response evaluation taxonomy for Sensitive Topics, which breaks down helpfulness and harmlessness into errors across three main categories: Content, Logic, and Appropriateness. Experiments on a Korean-sensitive question dataset demonstrate that our score- and error-based improvement pipeline, guided by FINEST, significantly improves the model responses across all three categories, outperforming refinement without guidance. Notably, score-based improvement -- providing category-specific scores and justifications -- yields the most significant gains, reducing the error sentence ratio for Appropriateness by up to 33.09%. This work lays the foundation for a more explainable and comprehensive evaluation and improvement of LLM responses to sensitive questions. |
| 2026-03-04 | [SaFeR: Safety-Critical Scenario Generation for Autonomous Driving Test via Feasibility-Constrained Token Resampling](http://arxiv.org/abs/2603.04071v1) | Jinlong Cui, Fenghua Liang et al. | Safety-critical scenario generation is crucial for evaluating autonomous driving systems. However, existing approaches often struggle to balance three conflicting objectives: adversarial criticality, physical feasibility, and behavioral realism. To bridge this gap, we propose SaFeR: safety-critical scenario generation for autonomous driving test via feasibility-constrained token resampling. We first formulate traffic generation as a discrete next token prediction problem, employing a Transformer-based model as a realism prior to capture naturalistic driving distributions. To capture complex interactions while effectively mitigating attention noise, we propose a novel differential attention mechanism within the realism prior. Building on this prior, SaFeR implements a novel resampling strategy that induces adversarial behaviors within a high-probability trust region to maintain naturalism, while enforcing a feasibility constraint derived from the Largest Feasible Region (LFR). By approximating the LFR via offline reinforcement learning, SaFeR effectively prevents the generation of theoretically inevitable collisions. Closed-loop experiments on the Waymo Open Motion Dataset and nuPlan demonstrate that SaFeR significantly outperforms state-of-the-art baselines, achieving a higher solution rate and superior kinematic realism while maintaining strong adversarial effectiveness. |
| 2026-03-04 | [Monitoring Emergent Reward Hacking During Generation via Internal Activations](http://arxiv.org/abs/2603.04069v1) | Patrick Wilhelm, Thorsten Wittkopp et al. | Fine-tuned large language models can exhibit reward-hacking behavior arising from emergent misalignment, which is difficult to detect from final outputs alone. While prior work has studied reward hacking at the level of completed responses, it remains unclear whether such behavior can be identified during generation. We propose an activation-based monitoring approach that detects reward-hacking signals from internal representations as a model generates its response. Our method trains sparse autoencoders on residual stream activations and applies lightweight linear classifiers to produce token-level estimates of reward-hacking activity. Across multiple model families and fine-tuning mixtures, we find that internal activation patterns reliably distinguish reward-hacking from benign behavior, generalize to unseen mixed-policy adapters, and exhibit model-dependent temporal structure during chain-of-thought reasoning. Notably, reward-hacking signals often emerge early, persist throughout reasoning, and can be amplified by increased test-time compute in the form of chain-of-thought prompting under weakly specified reward objectives. These results suggest that internal activation monitoring provides a complementary and earlier signal of emergent misalignment than output-based evaluation, supporting more robust post-deployment safety monitoring for fine-tuned language models. |
| 2026-03-04 | [Inference-Time Toxicity Mitigation in Protein Language Models](http://arxiv.org/abs/2603.04045v1) | Manuel Fernández Burda, Santiago Aranguri et al. | Protein language models (PLMs) are becoming practical tools for de novo protein design, yet their dual-use potential raises safety concerns. We show that domain adaptation to specific taxonomic groups can elicit toxic protein generation, even when toxicity is not the training objective. To address this, we adapt Logit Diff Amplification (LDA) as an inference-time control mechanism for PLMs. LDA modifies token probabilities by amplifying the logit difference between a baseline model and a toxicity-finetuned model, requiring no retraining. Across four taxonomic groups, LDA consistently reduces predicted toxicity rate (measured via ToxDL2) below the taxon-finetuned baseline while preserving biological plausibility. We evaluate quality using Fréchet ESM Distance and predicted foldability (pLDDT), finding that LDA maintains distributional similarity to natural proteins and structural viability (unlike activation-based steering methods that tend to degrade sequence properties). Our results demonstrate that LDA provides a practical safety knob for protein generators that mitigates elicited toxicity while retaining generative quality. |
| 2026-03-04 | [Measuring AI R&D Automation](http://arxiv.org/abs/2603.03992v1) | Alan Chan, Ranay Padarath et al. | The automation of AI R&D (AIRDA) could have significant implications, but its extent and ultimate effects remain uncertain. We need empirical data to resolve these uncertainties, but existing data (primarily capability benchmarks) may not reflect real-world automation or capture its broader consequences, such as whether AIRDA accelerates capabilities more than safety progress or whether our ability to oversee AI R&D can keep pace with its acceleration. To address these gaps, this work proposes metrics to track the extent of AIRDA and its effects on AI progress and oversight. The metrics span dimensions such as capital share of AI R&D spending, researcher time allocation, and AI subversion incidents, and could help decision makers understand the potential consequences of AIRDA, implement appropriate safety measures, and maintain awareness of the pace of AI development. We recommend that companies and third parties (e.g. non-profit research organisations) start to track these metrics, and that governments support these efforts. |
| 2026-03-04 | [Map-Agnostic And Interactive Safety-Critical Scenario Generation via Multi-Objective Tree Search](http://arxiv.org/abs/2603.03978v1) | Wenyun Li, Zejian Deng et al. | Generating safety-critical scenarios is essential for validating the robustness of autonomous driving systems, yet existing methods often struggle to produce collisions that are both realistic and diverse while ensuring explicit interaction logic among traffic participants. This paper presents a novel framework for traffic-flow level safety-critical scenario generation via multi-objective Monte Carlo Tree Search (MCTS). We reframe trajectory feasibility and naturalistic behavior as optimization objectives within a unified evaluation function, enabling the discovery of diverse collision events without compromising realism. A hybrid Upper Confidence Bound (UCB) and Lower Confidence Bound (LCB) search strategy is introduced to balance exploratory efficiency with risk-averse decision-making. Furthermore, our method is map-agnostic and supports interactive scenario generation with each vehicle individually powered by SUMO's microscopic traffic models, enabling realistic agent behaviors in arbitrary geographic locations imported from OpenStreetMap. We validate our approach across four high-risk accident zones in Hong Kong's complex urban environments. Experimental results demonstrate that our framework achieves an 85\% collision failure rate while generating trajectories with superior feasibility and comfort metrics. The resulting scenarios exhibit greater complexity, as evidenced by increased vehicle mileage and CO\(_2\) emissions. Our work provides a principled solution for stress testing autonomous vehicles through the generation of realistic yet infrequent corner cases at traffic-flow level. |
| 2026-03-04 | [Right in Time: Reactive Reasoning in Regulated Traffic Spaces](http://arxiv.org/abs/2603.03977v1) | Simon Kohaut, Benedict Flade et al. | Exact inference in probabilistic First-Order Logic offers a promising yet computationally costly approach for regulating the behavior of autonomous agents in shared traffic spaces. While prior methods have combined logical and probabilistic data into decision-making frameworks, their application is often limited to pre-flight checks due to the complexity of reasoning across vast numbers of possible universes. In this work, we propose a reactive mission design framework that jointly considers uncertain environmental data and declarative, logical traffic regulations. By synthesizing Probabilistic Mission Design (ProMis) with reactive reasoning facilitated by Reactive Circuits (RC), we enable online, exact probabilistic inference over hybrid domains. Our approach leverages the Frequency of Change inherent in heterogeneous data streams to subdivide inference formulas into memoized, isolated tasks, ensuring that only the specific components affected by new sensor data are re-evaluated. In experiments involving both real-world vessel data and simulated drone traffic in dense urban scenarios, we demonstrate that our approach provides orders of magnitude in speedup over ProMis without reactive paradigms. This allows intelligent transportation systems, such as Unmanned Aircraft Systems (UAS), to actively assert safety and legal compliance during operations rather than relying solely on preparation procedures. |
| 2026-03-04 | [ArthroCut: Autonomous Policy Learning for Robotic Bone Resection in Knee Arthroplasty](http://arxiv.org/abs/2603.03957v1) | Xu Lu, Yiling Zhang et al. | Despite rapid commercialization of surgical robots, their autonomy and real-time decision-making remain limited in practice. To address this gap, we propose ArthroCut, an autonomous policy learning framework that upgrades knee arthroplasty robots from assistive execution to context-aware action generation. ArthroCut fine-tunes a Qwen--VL backbone on a self-built, time-synchronized multimodal dataset from 21 complete cases (23,205 RGB--D pairs), integrating preoperative CT/MR, intraoperative NDI tracking of bones and end effector, RGB--D surgical video, robot state, and textual intent. The method operates on two complementary token families -- Preoperative Imaging Tokens (PIT) to encode patient-specific anatomy and planned resection planes, and Time-Aligned Surgical Tokens (TAST) to fuse real-time visual, geometric, and kinematic evidence -- and emits an interpretable action grammar under grammar/safety-constrained decoding. In bench-top experiments on a knee prosthesis across seven trials, ArthroCut achieves an average success rate of 86% over the six standard resections, significantly outperforming strong baselines trained under the same protocol. Ablations show that TAST is the principal driver of reliability while PIT provides essential anatomical grounding, and their combination yields the most stable multi-plane execution. These results indicate that aligning preoperative geometry with time-aligned intraoperative perception and translating that alignment into tokenized, constrained actions is an effective path toward robust, interpretable autonomy in orthopedic robotic surgery. |
| 2026-03-04 | [When Safety Becomes a Vulnerability: Exploiting LLM Alignment Homogeneity for Transferable Blocking in RAG](http://arxiv.org/abs/2603.03919v1) | Junchen Li, Chao Qi et al. | Retrieval-Augmented Generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge, but its reliance on potentially poisonable knowledge bases introduces new availability risks. Attackers can inject documents that cause LLMs to refuse benign queries, attacks known as blocking attacks. Prior blocking attacks relying on adversarial suffixes or explicit instruction injection are increasingly ineffective against modern safety-aligned LLMs. We observe that safety-aligned LLMs exhibit heightened sensitivity to query-relevant risk signals, causing alignment mechanisms designed for harm prevention to become a source of exploitable refusal. Moreover, mainstream alignment practices share overlapping risk categories and refusal criteria, a phenomenon we term alignment homogeneity, enabling restricted risk context constructed on an accessible LLM to transfer across LLMs. Based on this insight, we propose TabooRAG, a transferable blocking attack framework operating under a strict black-box setting. An attacker can generate a single retrievable blocking document per query by optimizing against a surrogate LLM in an accessible RAG environment, and directly transfer it to an unknown target RAG system without access to the target model. We further introduce a query-aware strategy library to reuse previously effective strategies and improve optimization efficiency. Experiments across 7 modern LLMs and 3 datasets demonstrate that TabooRAG achieves stable cross-model transferability and state-of-the-art blocking success rates, reaching up to 96% on GPT-5.2. Our findings show that increasingly standardized safety alignment across modern LLMs creates a shared and transferable attack surface in RAG systems, revealing a need for improved defenses. |
| 2026-03-04 | [IROSA: Interactive Robot Skill Adaptation using Natural Language](http://arxiv.org/abs/2603.03897v1) | Markus Knauer, Samuel Bustamante et al. | Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability. |
| 2026-03-04 | [Dual-Interaction-Aware Cooperative Control Strategy for Alleviating Mixed Traffic Congestion](http://arxiv.org/abs/2603.03848v1) | Zhengxuan Liu, Yuxin Cai et al. | As Intelligent Transportation System (ITS) develops, Connected and Automated Vehicles (CAVs) are expected to significantly reduce traffic congestion through cooperative strategies, such as in bottleneck areas. However, the uncertainty and diversity in the behaviors of Human-Driven Vehicles (HDVs) in mixed traffic environments present major challenges for CAV cooperation. This paper proposes a Dual-Interaction-Aware Cooperative Control (DIACC) strategy that enhances both local and global interaction perception within the Multi-Agent Reinforcement Learning (MARL) framework for Connected and Automated Vehicles (CAVs) in mixed traffic bottleneck scenarios. The DIACC strategy consists of three key innovations: 1) A Decentralized Interaction-Adaptive Decision-Making (D-IADM) module that enhances actor's local interaction perception by distinguishing CAV-CAV cooperative interactions from CAV-HDV observational interactions. 2) A Centralized Interaction-Enhanced Critic (C-IEC) that improves critic's global traffic understanding through interaction-aware value estimation, providing more accurate guidance for policy updates. 3) A reward design that employs softmin aggregation with temperature annealing to prioritize interaction-intensive scenarios in mixed traffic. Additionally, a lightweight Proactive Safety-based Action Refinement (PSAR) module applies rule-based corrections to accelerate training convergence. Experimental results demonstrate that DIACC significantly improves traffic efficiency and adaptability compared to rule-based and benchmark MARL models. |
| 2026-03-04 | [Cognition to Control - Multi-Agent Learning for Human-Humanoid Collaborative Transport](http://arxiv.org/abs/2603.03768v1) | Hao Zhang, Ding Zhao et al. | Effective human-robot collaboration (HRC) requires translating high-level intent into contact-stable whole-body motion while continuously adapting to a human partner. Many vision-language-action (VLA) systems learn end-to-end mappings from observations and instructions to actions, but they often emphasize reactive (System 1-like) behavior and leave under-specified how sustained System 2-style deliberation can be integrated with reliable, low-latency continuous control. This gap is acute in multi-agent HRC, where long-horizon coordination decisions and physical execution must co-evolve under contact, feasibility, and safety constraints. We address this limitation with cognition-to-control (C2C), a three-layer hierarchy that makes the deliberation-to-control pathway explicit: (i) a VLM-based grounding layer that maintains persistent scene referents and infers embodiment-aware affordances/constraints; (ii) a deliberative skill/coordination layer-the System 2 core-that optimizes long-horizon skill choices and sequences under human-robot coupling via decentralized MARL cast as a Markov potential game with a shared potential encoding task progress; and (iii) a whole-body control layer that executes the selected skills at high frequency while enforcing kinematic/dynamic feasibility and contact stability. The deliberative layer is realized as a residual policy relative to a nominal controller, internalizing partner dynamics without explicit role assignment. Experiments on collaborative manipulation tasks show higher success and robustness than single-agent and end-to-end baselines, with stable coordination and emergent leader-follower behaviors. |

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



