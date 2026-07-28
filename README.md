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
| 2026-07-27 | [Infrared Imaging Empowered by Artificial Intelligence for Pediatric Skeletal Triage: A Narrative Review and Future Perspectives](http://arxiv.org/abs/2607.24727v1) | Sajad Amiri, Pardis Afshar et al. | Background. Pediatric musculoskeletal trauma represents up to 18% of pediatric ED visits, yet diagnosis still depends on ionizing radiography. Cumulative low-dose radiation in early life raises lifetime leukemia and brain malignancy risk, motivating radiation-free triage alternatives. Objective. To synthesize evidence for a hybrid framework coupling broad-spectrum infrared (IR) imaging with deep-learning cross-modal translation to generate clinically interpretable synthetic-radiograph reconstructions from non-ionizing data.   Approach. We review five IR spectral windows spanning 650 nm to 1 mm - NIR-I, NIR-II, SWIR, MIR/LWIR, and THz - and how dual-geometry (transmission/reflection) acquisition exploits wavelength-specific tissue depth and biochemical sensitivity. We summarize image-to-image translation networks (Pix2Pix, CycleGAN, Swin-Unet) and feature-matching algorithms (SuperPoint, SuperGlue, ALIKED, LightGlue) used to align and fuse IR data into radiograph-equivalent reconstructions.   Implications. Pediatric anatomy - smaller cross-sections, thinner cortical bone - favors IR penetration, enabling compact, portable, non-ionizing triage hardware. Feasibility is grounded in fNIRS and transcranial photobiomodulation evidence: near-infrared light passes through skin, skull, and cortex with sufficient signal for hemodynamic monitoring - a longer, more attenuating path than through a pediatric forearm or distal leg. Key barriers: paired IR/X-ray dataset construction, AI-as-medical-device regulatory pathways, generalization across body habitus and skin pigmentation, and acquisition-protocol standardization.   Conclusions. Integrated multi-spectral IR+AI imaging is a promising radiation-free complement to pediatric skeletal radiography. Progress requires multi-center paired datasets, externally validated models, and IR source safety qualification under IEC 60825-1. |
| 2026-07-27 | [Explainable Reinforcement Learning via Physics-Aware Policy Distillation](http://arxiv.org/abs/2607.24672v1) | Shaker Al-Tamari, Waled Kadour | In safety-critical sectors such as robotics and automotive engineering, the deployment of Deep Reinforcement Learning (DRL) is often hindered by the black-box nature of deep neural networks. This lack of transparency poses significant challenges for regulatory compliance and human-agent trust. This paper presents an experimental study aimed at making high-performance continuous control DRL systems interpretable. A policy distillation framework is implemented using the classic Inverted Pendulum benchmark. A high-performance Twin Delayed DDPG (TD3) agent serves as an opaque, continuous teacher model, whose policy is distilled into an interpretable student surrogate based on a shallow Decision Tree. By leveraging a custom physics-aware feature and "Noisy Oracle Rollouts" for dataset generation, the distillation process achieves performance equivalent to the expert teacher. Furthermore, comparative control theory analysis reveals a fundamental trade-off: transitioning from continuous to discrete rule-based control induces high-frequency Bang-Bang actuation and a stable bimodal limit cycle. Simulation results indicate that Bounded-Input Bounded-Output (BIBO) stability is maintained while providing both global and local interpretability for safe autonomous systems. |
| 2026-07-27 | [Evaluating Fuzz Testing for Reinforcement Learning Agents](http://arxiv.org/abs/2607.24577v1) | Zhibin Kang, Hanmo You et al. | Reinforcement Learning (RL) agents are increasingly deployed in safety-critical domains such as robotics, autonomous driving, and drone control, where unexpected behaviors may lead to severe real-world consequences. Fuzz testing has recently emerged as a promising method for exploring the vast state spaces of RL agents and exposing crashes. Although numerous RL fuzzing methods have been proposed, existing studies often differ in evaluation settings, baselines, and metrics, making it difficult to draw reliable conclusions about their relative effectiveness and practical usefulness. To address this gap, we present the first comprehensive empirical study that systematically evaluates RL fuzzing methods from four complementary perspectives: effectiveness, diversity, efficiency, and practical utility. We benchmark five state-of-the-art methods alongside random testing under unified configurations across three environments of increasing complexity (MountainCar, BipedalWalker, and CARLA), and further assess the downstream usefulness of detected crashes for agent robustness improvement and safety monitoring. Our results reveal several key insights. For instance,throughput-oriented methods like MDPFuzz demonstrate superior effectiveness and efficiency in crash discovery, while methods explicitly designed to encourage exploration like SeqDivFuzz excel at uncovering diverse crash behaviors. We also show that fuzzing-generated crashes can meaningfully improve agent robustness and enable accurate safety monitoring with strong cross-method generalization. Beyond these empirical findings, we distill actionable guidance for both researchers and practitioners, highlighting the benefits of combining complementary fuzzing strategies and adopting multi-level diversity analysis to achieve more comprehensive and practical RL testing. |
| 2026-07-27 | [Making Mathematical Knowledge Explainable, Accessible and Interoperable Through Large Language Model Integration](http://arxiv.org/abs/2607.24512v1) | Jan Range, Björn Schembera et al. | Mathematical models are central to formalizing research problems, yet their documentation often falls short of FAIR principles. Knowledge bases such as the Mathematical Model Database (MathModDB) address this gap by providing curated, semantically rich representations of mathematical models. Built on Wikibase, the same open-source infrastructure underlying Wikidata, MathModDB utilizes Semantic Web technologies to support Linked Open Data, collaborative editing, and the storage of semantically enriched metadata, making it a domain-specific knowledge graph within the broader Wikidata ecosystem. However, access to MathModDB currently requires either navigating a complex web interface or proficiency in SPARQL and Wikibase APIs, posing significant barriers for potential users. In addition, the combination of such curated knowledge bases with actual research data stored, e.g., in Dataverse repository instances, remains a challenge. To overcome these limitations, we propose integrating Large Language Models (LLMs) with MathModDB via a Model Context Protocol (MCP) server that exposes a vector-indexed schema retrieval and Steiner-tree-based join planner, combining dialogue-based natural language interaction with curated, epistemically grounded knowledge. Although instantiated on MathModDB, the architecture can be applied to other Wikibase-based systems. We demonstrate that this approach enables epistemically grounded LLM usage, improves model explainability and accessibility beyond what the standard Wikibase interface offers, and simplifies interoperability with external databases and tools, such as Dataverse data repositories. We illustrate the benefits of combining the accessibility of an LLM with the epistemic safety of a curated knowledge base through the adaptability of the MCP protocol by two use cases involving mathematical models in the fields of continuum mechanics and enzyme kinetics. |
| 2026-07-27 | [Classifying Capabilities (Extended Version)](http://arxiv.org/abs/2607.24504v1) | Cao Nguyen Pham, Oliver Bračevac et al. | Capture checking in Scala 3 enables lightweight and practical effect and resource tracking by recording capabilities in types. However, the system offers no way to reason about kinds of capabilities. Natural constraints such as "retaining only the control-flow capabilities of this closure" or "excluding all thread-local capabilities from this argument" become inexpressible. Both arise in the Scala 3 standard library: "Try" re-throws caught exceptions, so it retains only the control-flow capabilities of its body, and "Future" must not capture thread-local resources. The inability to state these constraints has kept parts of the library outside capture checking.   We introduce capability classifiers: a tree-structured, user-extensible hierarchy of tags that classify capabilities by their semantic role. Projections filter capture sets by classifier, supporting both inclusion ("c.only[C]") and exclusion ("c.except[C]"). The tree structure enables decidable disjointness reasoning: classifiers on separate branches are guaranteed to be disjoint regardless of unknown extensions elsewhere in the hierarchy. We formalize classifiers as an extension of System Capless, a core calculus for capture checking, introducing a classifier kind algebra based on intersection, union, and subtraction of classifier subtrees. We extend the operational semantics to model exception interception and establish type safety, effect safety, and handler coverage via a big-step proof, fully mechanized in Lean 4. Classifiers are implemented in the Scala 3 capture checker, and we demonstrate their use on standard library types and real-world effect exclusion patterns. |
| 2026-07-27 | [InterOCF: Spatio-Temporal 2D-3D Interaction for Camera-Only 4D Occupancy Forecasting](http://arxiv.org/abs/2607.24431v1) | Qi Zhang, Xinquan Yu et al. | Camera-only 4D occupancy forecasting enables autonomous vehicles to predict future 3D semantic scenes solely from historical multi-view images, which is critical for driving safety. Even though current methods have achieved good performance, the strong spatial-temporal modeling between the input multi-view frames is still underexplored, which limits the performance of those methods in future 4D forecasting. To address this gap, we introduce a novel framework, InterOCF, for 4D occupancy forecasting that jointly models temporal dynamics in both 3D voxel-based representations and multi-view segmentation sequences, while explicitly incorporating feature interaction between the 2D and 3D branches. Our framework incorporates three core components: 1) A 3D Spatio-Temporal (3DST) module that learns volumetric dynamics from historical voxel states to predict future voxel states; 2) A 2D Spatio-Temporal (2DST) module employing an auxiliary multi-view temporal segmentation forecasting task to enhance temporal semantic dynamics; 3) A Spatio-Temporal Interaction Modeling (STIM) module that enables feature interaction between 2D and 3D representations. Experiments on the nuScenes, Lyft-Level5, and nuScenes-Occupancy datasets show that InterOCF consistently outperforms existing baseline approaches. |
| 2026-07-27 | [DeepNC: A Fast GNN-based Pre-Verification Surrogate for TSN Configuration](http://arxiv.org/abs/2607.24398v1) | Jiayi Zhu, Jing Lin et al. | Time-Sensitive Networking (TSN) is critical to deterministic communication in safety-critical domains, with formal verification such as Network Calculus (NC) serving as the cornerstone for schedulability guarantees. However, during automated configuration-space exploration, repeated schedulability analysis consumes over 90% of the total configuration time, becoming the primary bottleneck for large-scale TSN configurations. To address this challenge, we propose DeepNC, a novel pre-verification surrogate module that pioneers the structural fusion of NC principles into a Graph Neural Network (GNN) for TSN configuration-space exploration. Rather than replacing formal verification, DeepNC acts as a high-speed pre-verification filter, reserving computationally expensive formal verification only for promising candidates. Extensive evaluations demonstrate that DeepNC significantly improves worst-case delay prediction accuracy over state-of-the-art learning-based methods, increasing the average $R^2$ by 55.8% and reducing the average MAPE by 65.3%. More importantly, its high-fidelity regression substantially reduces the number of formal verification calls during configuration-space exploration by 93.25%, while accelerating NC-based verification by more than two orders of magnitude. |
| 2026-07-27 | [When LLM Defenses Backfire: Characterizing Safety, Performance, and Cost Trade-offs](http://arxiv.org/abs/2607.24392v1) | Tong Zhang, Zexin Li et al. | Jailbreak defenses are essential for protecting large language models (LLMs), but they can also introduce secondary costs that weaken model utility. We present a systematic study of these defense trade-offs along three dimensions: performance impact, over-refusal on benign inputs, and inference cost. Rather than treating defenses as a single class, we organize them by operational strategy and examine how different strategies correlate with different side-effect profiles. Across state-of-the-art defense methods, widely used benchmark datasets, and representative open-source LLMs, we find that defenses rarely improve downstream capability, but instead vary in how they trade safety gains against usability and efficiency. In particular, rule-based defenses best preserve task performance, highly conservative self-reflective defenses often increase over-refusal, and multi-round defenses incur the largest runtime overhead. These results provide both a benchmark for evaluating defense side effects and practical guidance for selecting defenses under deployment constraints. |
| 2026-07-27 | [MobiWave: Dispatch-Oriented Graph Wavelets and Drift-Guided Selective Optimization for Autonomous Fleet Rebalancing](http://arxiv.org/abs/2607.24365v1) | Xiao Han, Pinbo Wang et al. | Autonomous fleets enable mobility platforms to coordinate idle vehicles directly, making fleet-wide rebalancing possible. However, two obstacles limit reliable deployment: overlapping regional and local traffic patterns can hide roads that remain useful for dispatch, and mobility drift can make a trained policy unreliable. Existing spatial aggregation mixes these patterns, while updating all parameters from limited recent data is costly and can damage stable knowledge. We propose \name, a framework that connects a dispatch-oriented multi-scale graph wavelet module with Drift-Guided Layer-Selective Optimization (DGLS). The first module addresses the representation challenge by separating graph-frequency patterns and weighting each scale according to its value for demand prediction and feasible rebalancing. DGLS addresses the adaptation challenge by measuring Dispatch-weighted Spectral Drift, selecting affected layers within a resource budget, and separating short shocks from persistent changes through a drift-aware fast--slow update. Candidate validation further rejects updates that fail to improve held-out dispatch reward without worsening monitored service or safety constraints. Experiments on both real-world datasets and simluated environments demonstrate the effectiveness of \name\ in comparing with state-of-the-art methods. The source code and datasets are available at https://anonymous.4open.science/r/MobiWave-40F8/. |
| 2026-07-27 | [Epistemic Norms for AI Safety and Alignment Research](http://arxiv.org/abs/2607.24243v1) | Keivan Navaie | Mainstream AI research emphasises capability growth and tolerates low failure rates when average-case performance is high. AI safety and alignment research has a different mission: to ensure that catastrophic failures never occur, under sparse evidence, adversarial dynamics, and fat-tailed risk. We argue that the two domains differ along two analytically independent axes---{\it capability profile}, demonstrating the absence of hazardous behaviours rather than the presence of positive capabilities, and {\it risk profile}, bounding worst-case outcomes under fat-tailed uncertainty rather than optimising average-case performance---and that mainstream epistemic practices are inadequate on both. Building on a structured synthesis grounded in a preregistered bibliometric baseline, we identify five cross-cutting gap dimensions in current alignment research, including the near-absence of institutionalised independent verification. To address these gaps, we propose {\sc ECAISA}, an Epistemic Code for AI Safety and Alignment comprising eight principles, a three-level scoring rubric, a four-level disclosure ladder that reconciles transparency with information-hazard and commercial-confidentiality constraints, a tiered applicability scheme, an information-hazard adjudication procedure, and seven anti-gaming mechanisms. {\sc ECAISA} does not certify that any AI system is safe; it constrains how safety-relevant research claims are documented, checked, and relied upon, with auditability rather than certification as its governance target. |
| 2026-07-27 | [A Nonlinear CTH-Based Adaptive Cruise Controller With Safety and String Stability Guarantees](http://arxiv.org/abs/2607.24228v1) | Dionysios Theodosis, Amirhossein Samii et al. | This paper studies the problem of longitudinal control, with safety and string stability guarantees, in vehicle platoons. A nonlinear adaptive cruise controller based on the Constant Time Headway (CTH) policy is designed for platoons of heterogeneous vehicles, that incorporates a speed-dependent nonlinear gain guaranteeing bounded vehicle speeds and a saturation-type nonlinearity to guarantee bounded (or non-aggressive) accelerations, while preserving collision-avoidance properties of time-headway-based spacing policies. By introducing a suitable spacing-error variable, explicit nonlinear error dynamics are obtained, allowing a direct analysis of the closed-loop system. It is shown that the collision-free region is positively invariant and that the proposed controller guarantees convergence of the spacing errors. Furthermore, nonlinear string-stability properties are established directly from the closed-loop dynamics without resorting to linearization or frequency-domain arguments. In particular, sufficient conditions are derived for $L_p$ string stability, $1\le p\le\infty$, with respect to disturbances, external inputs acting on the leading vehicle, and perturbations of the initial conditions. |
| 2026-07-27 | [Reasoning to Regulate: Chain-of-Thought for Traffic Rule Understanding](http://arxiv.org/abs/2607.24199v1) | Yueru Luo, Xu Yan et al. | Understanding and complying with traffic regulations is a safety-critical requirement for autonomous driving, yet remains challenging due to the diversity and context dependence of traffic signage. Importantly, regulation understanding is not a simple recognition task, but a reasoning problem: whether a rule applies depends on interpreting the sign in relation to the spatial layout of lanes and scene context. To support such reasoning, MapDR provide fine-grained annotations that link each traffic sign's regulatory rules to the specific lanes they govern. Existing methods, however, largely treat this as direct sequence prediction, ignoring the underlying reasoning that connects sign semantics and map structure. To address this limitation, we explicitly incorporate reasoning into this task and propose a framework that equips vision-language models (VLMs) with chain-of-thought (CoT) capabilities. We first design a scalable CoT curation pipeline that bootstraps rationales from a strong LLM through a two-round strategy and employs a VLM-based verifier to filter out incorrect cases, yielding a high-quality set of (CoT, answer) pairs. Building on this foundation, we adopt a two-stage training scheme: supervised fine-tuning (SFT) to teach rationale-to-answer generation, followed by GRPO reinforcement learning with answer-grounded, fine-grained rewards to further improve final answer accuracy. Extensive experiments on MapDR show that our approach significantly improves both interpretability and accuracy, establishing the first reasoning-based framework for regulation-aware autonomous driving. |
| 2026-07-27 | [Forecasting the Emergence and Evolution of Crash Hotspots: A Unified Deep Learning Framework for Proactive Traffic Safety](http://arxiv.org/abs/2607.24168v1) | Jingwen Zhu, Keshu Wu et al. | Road crashes remain among the gravest threats to public safety, and preventing them is a defining task of transportation systems worldwide. Much of that harm concentrates at hotspots, yet a hotspot is less a place than an episode; it emerges quietly at an intersection or along an arterial, intensifies for weeks, then subsides, only to reappear elsewhere. Enforcement guided by maps of past crashes inevitably trails this cycle, patrolling yesterday's hotspots while tomorrow's form unwatched. Breaking that lag requires three capabilities at once: detecting hotspots as they are born, forecasting where they will sit next week, and following each one through its life. We introduce HERALD (Hotspot Emergence, Risk Anticipation, and Life-cycle Dynamics), a unified deep learning framework that provides all three from a single statewide model. HERALD distills each county's recent crash history into weekly risk maps and forecasts the next with a CNN--Transformer, whose mixture-of-experts lets one model serve dense urban cores and sparse rural corridors alike. Each forecast is anchored in the county's long-run crash geography, sharpened by the self-exciting effect of recent crashes, and paired with explicit warnings of where new hotspots are about to appear. Followed over time, every hotspot acquires a legible life story, from birth through growth and stability to decline and death. Across six heterogeneous Wisconsin counties, HERALD forecasts more accurately than five identically trained baselines, locates hotspots most precisely, and flags emerging risks before they take hold. A single adjustable setting trades accuracy for extra sensitivity where deployment demands it. The result shifts hotspot management from mapping the past to anticipating the future. |
| 2026-07-27 | [Constrained Reinforcement Learning Using Successor Representations](http://arxiv.org/abs/2607.24057v1) | Michael Girstl, Alexander Mattick et al. | Real-world Reinforcement Learning depends on the ability to formulate safety constraints into a policy. A common way to model such constraints is to introduce an additional cost signal in the Markov Decision Process, which notifies the agent of unwanted behavior independently of the reward signal. Unfortunately, current methods are hard to adapt to changes in the cost function introduced by, e.g., domain shift or obstacles moving over time. The lack of adaptability means that policies are too unflexible to deal with complex real-world conditions. We propose the Safe Deep Successor Representation (SafeDSR), a novel method that allows quick retraining of policies towards new cost structures. SafeDSR extends the Deep Successor Representation (Kulkarni et al., 2016) to Constrained Reinforcement Learning by introducing a single learnable weight matrix to decouple the learned value function across dynamics, rewards, and costs. This matrix can be updated in a supervised manner instead of having to adapt the whole network if the cost structure of the environment changes. We demonstrate this ability in a freely configurable two-dimensional navigation environment and show that our method is competitive on a simple navigation task while being considerably more flexible |
| 2026-07-27 | [SyRuP: Enhancing System-Prompt Following via Reward-Guided Prediction in LLM Decoding](http://arxiv.org/abs/2607.23991v1) | Seoyeon Kim, Minjae Kang et al. | Large Language Models (LLMs) are increasingly controlled through system prompts that specify roles, styles, formats, and safety requirements. However, models follow these prompts only implicitly through in-context learning, which can be insufficient for complex or compositional prompts. Existing approaches often require model tuning or response-level reranking, limiting their practicality for lightweight inference-time control. We introduce SyRuP, a decoding-time framework for improving system-prompt adherence while keeping the base LM frozen. SyRuP trains a cross-attention reward head from system-prompt-conditioned preference pairs, treating the system prompt as a separate memory to produce token-level adherence scores. At inference, SyRuP reranks the base LM's top-k candidates by combining base logits with the learned reward signal and an optional contrastive signal capturing system-induced logit shifts. Experiments on system-prompt following benchmarks show that SyRuP consistently outperforms prompting and decoding-time baselines with moderate inference overhead. These results suggest that explicit token-level guidance is an effective and practical mechanism for reliable system-prompt following. |
| 2026-07-27 | [Moral Hazard in Multi-Agent Language Models](http://arxiv.org/abs/2607.23982v1) | Dane Malenfant | Cooperation can fail when socially valuable effort is costly, weakly observable, and mainly benefits others. Drawing on Holmström's team moral-hazard model, we introduce the Dialogue Moral Hazard Game, a controlled textual game that operationalizes this hidden-action structure for language agents. In each episode, an agent can preserve an immediate local reward or pay a query cost to reveal a hidden safety fact that primarily helps another agent's downstream decision. We evaluate seven open-weight language models and decompose behavior into query use, realized information transfer, local-reward preservation, unsafe choice, format validity, and team success. Base models commonly preserve local reward without team success or query without communicating information that changes the final decision. We then use supervised fine-tuning, RLOO, sequential SFT+RLOO, and GEPA prompt optimization as diagnostic update mechanisms. Their effects are heterogeneous: OLMo-7B shows the clearest mechanism-consistent weight-level improvement, whereas GEPA sometimes improves team success while reducing or eliminating costly queries. Thus, optimization can shift aggregate reward without recovering the intended cooperative mechanism, motivating evaluations that report mechanism-level behavior rather than team success alone. |
| 2026-07-27 | [Mutual Modality Trust with Lightweight Reconstruction Regularization for Fine-grained Tire Pattern Recognition](http://arxiv.org/abs/2607.23979v1) | Jianning Yang, Jie Fang et al. | Visual tire recognition serves as a core supporting technique for vehicle safety monitoring, autonomous driving perception and automated automotive maintenance. Existing fine-grained tire recognition techniques suffer from three prominent limitations. They tend to depend on only one visual source, lack the capacity to jointly model spatial and frequency cues for minute tread texture extraction, and suffer severe overfitting given limited annotated tire imagery. This paper proposes a lightweight fine-grained tire pattern recognition method incorporating dual-branch independent inference and enhanced feature fusion to boost recognition performance. The framework employs two task-specialized branches dedicated to tire surface and tread indentation, respectively, to extract modality-specific discriminative features. Each branch conducts independent prediction, while cross-branch feature fusion exploits Mutual Modality Trust (M$^2$T) to realize complementary feature enhancement across two modalities. Besides, a frequency-domain hierarchical guidance module is devised, which leverages bandpass filters to decompose feature maps into high- and low-frequency components and enables fine-grained cross-layer feature modulation. Furthermore, a Lightweight Reconstruction Regularization (LR$^2$) is introduced to retain abundant intrinsic information within feature embeddings, substantially improving feature stability and recognition robustness under limited labeled training data. In addition, we establish a surface-indentation multi-source dataset namely MTire299 for fine-grained tire tread recognition, which covers 299 categories with a total of 14795 paired image samples. Extensive experiments conducted on two public tire datasets validate the superiority and efficacy of the proposed algorithm. |
| 2026-07-27 | [Plato-Bio: verification-first biological novelty screening with temporal rediscovery and structural benchmarks](http://arxiv.org/abs/2607.23975v1) | Stefan G. Creadore | Large language model research agents can connect literature retrieval, analysis code, and manuscript preparation, but coherent output does not establish scientific validity. We developed Plato-Bio, a biology-routed extension of the open Plato/Denario architecture that couples explicit workflow states with provenance records, citation checks, claim-to-evidence links, scoped file writes, and publication gates. A source audit identified and repaired three defects that could distort evaluation: loss of task domain in the default factory, omission of declared method signals from scoring, and evidence sidecars that lacked the drafted-claim denominator. On the current clean revision, the full Python suite completed with 931 passes, six skips, and no failures or errors; targeted biology, genomics, evidence/citation, and adversarial-safety suites likewise completed without failure. We evaluated two narrow use cases. In a frozen historical rediscovery task, independent pre-1986 literature bridges ranked the later-studied relation between fish oil and Raynaud phenomenon first; TF-IDF ranked it second and corpus frequency third. This single curated task measures retrospective ranking, not prospective discovery. In a separate comparison of AlphaFold models with experimental structures for 15 human proteins, 11 targets had high-confidence-core C-alpha RMSD below 1 Angstrom (median 0.501 Angstrom). Four targets exceeded 2 Angstrom, and confidence masking reduced the SUMO1 discrepancy from 16.61 to 2.58 Angstrom over 74 residues. The workflow emitted 27 traceable discrepancy regions, all retained as unvalidated hypotheses. Plato-Bio therefore provides reproducible software contracts and auditable screening baselines; broader claims of agent efficacy or biological novelty require preregistered evaluation, independent review, and prospective validation. |
| 2026-07-27 | [Bridging Reinforcement Learning and Optimal Control via Feasible Action Mapping](http://arxiv.org/abs/2607.23930v1) | Stefan Richter, Alberto Giammarino et al. | Operating constrained dynamical systems requires controllers to efficiently solve complex tasks while enforcing recursive feasibility and safety constraints. To address these competing requirements, we present Feasible Action for Optimal Control (FAOC), a novel control framework integrating Reinforcement Learning (RL) and Optimal Control (OC). The key contribution is a computationally efficient, optimization-based mapping algorithm that transforms the RL agent's action from a static abstract set into a state-dependent feasible parameter set of the Optimal Control Problem (OCP), guaranteeing strict satisfaction of the dynamical system's constraints. Thus, FAOC effectively combines the predictable safety of OC with the flexibility of RL. In contrast to prior work, the abstract action space of the RL agent does not require expert or heuristic design, and the OCP formulation is not compromised by the inability of RL to guarantee feasibility. We apply our approach to real-time motion planning for robot table tennis, which encapsulates these challenges. Via simulated experiments, we show that FAOC outperforms state-of-the-art baselines in both sample efficiency and closed-loop performance. |
| 2026-07-27 | [MemTX: Transactional Belief Commit for Stateful Agent Memory](http://arxiv.org/abs/2607.23929v1) | Xiaoyang Li, Yiqi Wang et al. | LLM agents increasingly coordinate through persistent shared memory: one agent's write becomes another agent's premise, and eventually a tool call with real side effects. Current agent memory systems treat every accepted write as immediately actionable truth, so a polluted tool result, a stale update, or a teammate's half-finished note can silently drive an irreversible action. We argue that a memory write is not a belief commit. We present MemTX, a transactional belief-commit protocol. Each record carries evidence, permissions, provenance, and validity. Writes are staged inside snapshot-isolated transactions and admitted by a validate-and-commit pipeline, irreversible tool calls are gated on in-flight belief state, and retracting a belief triggers typed cascading repair of its derived records and tool side effects. Two invariants, action-safety gating and cascade-repair completeness, are machine-checked by property-based testing and bounded exhaustive enumeration of 5.5 million protocol states, with zero violations. Across five backbones from three model families, MemTX leads all eight baselines with paired-McNemar significance on four backbones and statistically ties the best baseline on the fifth and strongest, while remaining the only method with zero downstream harm on every backbone. Backbone capability does not substitute for commit discipline. |

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



