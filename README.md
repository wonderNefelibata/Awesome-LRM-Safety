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
| 2026-11-20 | [AnimalFarm: Auditing the Generation and Detection of Hateful Images Featuring Anthropomorphic Characters](http://arxiv.org/abs/2611.27905v1) | Junjie Chu, Ye Leng et al. | Anthropomorphic non-human characters have long been used to convey hateful messages targeting human groups, from Nazi-era propaganda to recent AI-generated imagery on social media. However, the safety risks posed by such imagery remain largely unexplored. We thus conduct the first systematic study of hateful image generation and detection involving anthropomorphic non-human characters. We construct a benchmark from conservatively screened source scenes, covering five character types and three image generation tasks: text-to-image generation, image editing, and conversational image generation. Across 14,250 scheduled queries, image generation systems produce images for 92.1\% of queries, of which 95.9\% contain hateful content. We further collect 4,370 hateful images and evaluate five popular hateful-content detectors. Even the strongest detector, GPT-5.5, identifies only 73.9\% of hateful images, while most detect fewer than 58\%. Our analysis reveals that existing detectors often misclassify hateful images as benign when their main subjects are non-human characters. We explore two effective mitigation approaches: a training-free pipeline that maps non-human characters back to human representations before classification and a detector fine-tuned for this type of hateful content. These approaches achieve F1 scores of 0.962 and 0.979, respectively. Our findings expose significant safety gaps in both image generation and hateful-content detection.   Disclaimer: This paper contains hateful text and imagery. |
| 2026-11-20 | [ENC-QA: A Diagnostic Benchmark for Vision-Language Models on Electronic Navigational Charts](http://arxiv.org/abs/2611.27858v1) | Jiyoon Pyo, Jacob Arndt et al. | Nautical charts encode safety-critical maritime information through standardized symbols, attributes, text, and heterogeneous geometries. Interpreting these charts and assessing the navigational significance of successive chart updates requires recognizing nautical features, determining their spatial relationships, and applying navigational knowledge. Vision-language models (VLMs) offer a promising approach to nautical-chart interpretation, yet systematic evaluation of their fine-grained nautical feature recognition, chart-based spatial reasoning, and application of navigational knowledge to assessing chart changes remains limited. We introduce ENC-QA, a diagnostic benchmark comprising 77,312 questions derived from 6,612 National Oceanic and Atmospheric Administration (NOAA) Electronic Navigational Charts (ENCs). The benchmark evaluates four complementary capabilities: ENC Perception, Spatial Reasoning, Joint Perception and Spatial Reasoning, and Application. Of these, 8,227 questions form a progressive subset that decomposes selected chart-change cases into linked questions progressing from change detection to criticality assessment, enabling case-level analysis of how models use perceptual and spatial evidence to determine the navigational significance of chart changes. Evaluations of 10 open-weight VLMs reveal substantial limitations in fine-grained nautical feature recognition and feature-grounded spatial reasoning. Controlled evidence interventions show that relevant perceptual and spatial information improves change-criticality accuracy. Our findings establish ENC-QA as a diagnostic resource for nautical-chart understanding and safety-critical navigational decisions. We will release the benchmark and evaluation code upon publication. |
| 2026-11-20 | [Humanoid robots for safety-critical electrical switching: A perception-informed runtime constraint filter](http://arxiv.org/abs/2611.27845v1) | Songyang Liu, Yu Hou et al. | Humanoid robots can reduce personnel exposure in electrical switching, but safe control is challenging because the tool must reach a target switch next to similar devices, where an imprecise operator command can contact a non-target switch. This study proposes a runtime safety filter that checks each operator command during virtual-reality teleoperation of a humanoid robot. The filter integrates a scene model estimated from the robot head camera and a target-conditioned control barrier function quadratic program that keeps the tool away from the non-target switch while passing safe commands unchanged. In real-world experiments on a Unitree G1 humanoid robot, with six rounds at each of two standing positions, target operations succeed in all ten rounds with an unchanged target and no non-target actuation. A simulation study achieves 9/9 safe completions, compared with 5/9 for raw teleoperation, demonstrating the feasibility of the method for humanoid switching operations. |
| 2026-11-20 | [Beyond Self-Critique: Epistemic Convergence Control for Multi-Agent Research Verification](http://arxiv.org/abs/2611.27839v1) | Chrisantus Eze | Multi-agent systems for scientific claim verification increasingly rely on role-structured LLM debate to pressure-test hypotheses. However, when debating agents are instantiated from the same underlying base model, they share identical latent representations, making self-critique bias structurally inevitable. Static fixed-iteration protocols compound this by either halting prematurely before evidence stabilizes or wasting inference compute after convergence. We present \sys{}, a multi-agent research verification framework that addresses both failure modes. First, we introduce an asymmetric role architecture comprising an Advocate, Skeptic, and Arbitrator with opposing epistemic directives, producing significantly lower role-embedding similarity (0.852 vs.\ 0.918 for symmetric Multi-Agent Debate [MAD], $p = 5.82 \times 10^{-8}$) and surfacing more contested claims (22.0\% vs.\ 15.4\%). Second, we propose the Belief-Delta ($\DB$) convergence criterion, an evidence-grounded stopping rule that eliminates over-iteration (0.0\% vs.\ 62.5\% for Fixed-$3$) while cutting LLM calls by 2.4$\times$. Third, each supported claim is anchored to a verbatim literature snippet audited by an independent NLI classifier, achieving a 21.7\% hallucination rate (comparable to 25.9\% for a Self-Refine critic, $p = 0.61$) while establishing auditable evidence grounding. Across 200 runs on 40 research hypotheses evaluated with DeepSeek-R1-32B, \sys{} balances compute efficiency with rigorous epistemic verification. |
| 2026-11-20 | [Generalist Action Critic: Harnessing Frontier Models in Autonomous Driving](http://arxiv.org/abs/2611.27804v1) | Yun Yan, Shizhi Hu et al. | Frontier models have demonstrated remarkable general-purpose reasoning capabilities, opening new possibilities for intelligent decision making in the physical world. In autonomous driving, many recent approaches seek to leverage these capabilities through additional adaptation for action output, particularly in scenarios where existing models underperform. We introduce Generalist Action Critic (GAC), a zero-shot framework that combines frontier models' general-purpose reasoning with the planning competence of specialized driving models through an inference-time harness. In GAC, specialized planners generate multiple candidate trajectories and make an initial selection, while a frozen frontier model serves as an action critic that assesses the selected trajectory against scene evidence and either retains it or reselects from the available candidates. The harness combines camera images with spatial overlays, provides driving context and reasoning principles through textual instructions, and regulates the information available at each stage to support trajectory assessment and reselection. On HUGSIM, GAC raises DrivoR's HD-Score to 43.2, achieving state-of-the-art performance among methods trained without additional simulation data. GAC also consistently improves closed-loop driving safety across three planners, with relative HD-Score gains of up to 32.1%. Experiments on NAVSIM v1 further reveal how GAC improves planning performance by correcting up to 52.13% of eligible unsafe trajectory selections while introducing new failures in only 0.67% of originally high-quality selections. These results demonstrate the potential of harnessing frontier models to improve driving safety through inference-time cooperation with specialized planners. |
| 2026-11-20 | [ChildAI-Gov: An Evidence-Linked AI System for Assessing EdTech Policies Against Children's Digital Rights](http://arxiv.org/abs/2611.27799v1) | Naveen Sigalapalli, Huizhi Liang et al. | We present ChildAI-Gov, an explainable AI system for assessing whether education-technology (EdTech) policies respect children's digital rights. Generative AI is reaching children's education faster than its governance can keep pace, yet checking a tool against children's rights remains manual, expert-dependent, and slow. We propose the Child Rights Impact Score (CRIS), an evidence-linked measure across five dimensions: safety, privacy, transparency, autonomy, and non-discrimination. ChildAI-Gov reads an EdTech tool's privacy or AI policy and scores each dimension by combining a large-language-model judgement (gpt-4o-mini) with a deterministic rule check. A retrieval-augmented generation (RAG) component grounds every score in the governing provisions, retrieved from a curated knowledge base of children's rights frameworks (the EU AI Act, UNICEF guidance, and others), so each score is tied to the evidence behind it and an auditor can trace which provision it rests on. We also construct AIPolicy-EdTech, a dataset of 57 real EdTech policies. On it, the five dimensions cohered as a single measure (Cronbach's alpha = 0.79), and grounding made scoring stricter while preserving rankings; an ungrounded baseline was far more lenient, passing nearly half the policies where ChildAI-Gov passed none. It offers evidence-linked decision support on documented commitments, not observation of real-world behaviour or determinations of legal compliance. |
| 2026-11-20 | [The Four-Step Action Loop with Feedback: A teachable perception-action framework for everyday self-regulation](http://arxiv.org/abs/2611.27795v1) | Sundeep Kapila | Everyday stress is largely a learned pattern: a familiar cue, a fast reading of threat, an automatic reaction and a body that stays activated. Body-based practices such as mindfulness, interoceptive training, Somatic Experiencing and yoga target these patterns, but they are usually taught as techniques without a map of the mechanism they act on. Pain neuroscience education suggests that understanding such a mechanism can itself change experience. No comparable model exists for everyday self-regulation. This paper proposes the Four-Step Action Loop with Feedback, developed over six cohorts of a workshop for non-clinical adults. The loop describes any action as four steps: Sense (exteroception, proprioception and interoception), Process (perception as inference, weighted by context and safety), Choose (habitual reaction or goal-directed response after a pause) and Execute (autonomic and somatic output). Feedback from each outcome acts beneath all four steps, through synaptic plasticity, habit formation and allostatic load. The paper integrates the perception-action cycle, information-processing stages of choice, interoceptive inference, the autonomic space model and the defence cascade. It places a set of practices on the loop: full sensory awareness, grounding, orientation, resourcing, the pause, reduced stimulus and choice, slow movement and relaxation, each with a proposed mechanism and an assessment of the evidence. It also describes the model's translation into four practitioner skills. Seven testable propositions are derived for future evaluation. The model is offered as a teachable framework for wellbeing, not a treatment. |
| 2026-11-20 | [Neural Defend ARCAS 290M: A Robust Foundation Model for AI-Generated Audio Detection](http://arxiv.org/abs/2611.27784v1) | Sivashankar Selvarajan, Piyush Verma et al. | Voice cloning now needs only seconds of audio, and detectors tuned to one dataset often fail when the generator, channel, or language changes. We introduce ARCAS 290M, Neural Defend's foundation model for detecting AI-generated audio. A single frozen production model, with no benchmark-specific fine-tuning, generalizes across text-to-speech, voice-conversion, and codec-language-model generators, telephone and Bluetooth channels, recording conditions, and languages. We evaluate it on the complete official test sets of ten public benchmarks, scoring every recording exactly as released, and against commercial systems on independent leaderboards. ARCAS 290M ranks first among all systems on the Podonos leaderboard and has the lowest equal error rate among commercial detectors on ASVspoof5. It separates real from AI-generated speech almost perfectly on ASVspoof5, SpeechFake, and ShiftySpeech, and detects nearly all AI-generated speech in Hindi, Tamil, Bengali, and Telugu. On every benchmark with a published zero-shot comparison, including in-the-wild audio, partial fakes, and the conditions hardest for the field, it surpasses the strongest reported detector by a wide margin. These results establish ARCAS 290M as a general-purpose foundation for AI-generated audio detection in fraud prevention, platform safety, and media verification. |
| 2026-11-20 | [Learning Inside Types: Epistemic Typestates for Safe Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2611.27631v1) | Lorenzo Bacchiani, Mario Bravetti et al. | Type disciplines make programs safe by construction: components checked against each other's types compose into programs that cannot go wrong in the ways the types exclude. Types suit reinforcement learning: a type constrains what an agent may choose, not how it chooses, so the guarantee holds for every policy a learner produces. We introduce epistemic typestates, a type discipline for safe multi-agent reinforcement learning. An agent's type is a protocol whose states record its beliefs, the worlds it cannot rule out from its own observations and actions, and permits only actions that are safe in every world of the belief, whatever the others do within their public types. Agents learn inside their types, which are built and checked one agent at a time and run without communication. We prove type soundness and team safety, give a synthesis that is sound and complete, build types from guards when worlds are infinite, and show that shields are a special case. All proofs are mechanised in Rocq. On nine benchmarks, no typed learner causes a safety violation, and task performance does not differ significantly from that of two recent multi-agent shielding methods. Where a contract-based method fails at five and ten agents, types for ten agents are built in under a second. In the two continuous benchmarks, Lagrangian learners cause thousands of violations in every training run. |
| 2026-11-20 | [Potential emergence in LLM agents: a statistical physics perspective](http://arxiv.org/abs/2611.27618v1) | Zhuo-Yang Song, Hua Xing Zhu | There has been significant recent interest in understanding the behaviour of LLM agents from the perspective of human preferences or corpus statistics. In this paper we propose agent states as intermediate variables through which post-training can be measured and predicted. From a statistical physics perspective, we study the emergence of a global potential which governs the transition of state in LLM agent reasoning. We introduce a partition function to model the ideal KL-regularised transition distribution and track potential structure, mean reward and state entropy. We demonstrate our approach in two Qwen3.5-9B tasks: safety-test selection and physics-reasoning auditing. We observe that the potential structure strengthens during post-training as predicted from our framework, while joint changes in reward and state entropy provide consistency evidence. States thus provide a quantitative description of how post-training organises agent behaviour. |
| 2026-11-20 | [Bringing AI Assistance to the Beznau Nuclear Power Plant: A Hierarchical System of LLM Agents for Reactor Monitoring](http://arxiv.org/abs/2611.27556v1) | Wenqi Zhou, Leonardo Baldo et al. | Beznau Unit 1 in Switzerland, in operation since 1969, is the most experienced nuclear reactor still running worldwide. Five decades of incremental modernization have left its monitoring systems, alarm logs, variable catalogs, technical specifications and investigation reports in separate silos that were never designed to be queried together. As a result, investigating a core instrumentation anomaly means slow, manual gathering of knowledge scattered across databases, documents, and experienced colleagues. We present a hierarchical multi-agent assistant built on large language models (LLMs) to support these investigations and reduce the need for tedious manual searches in SAP systems. An orchestrating agent breaks down an engineer's question and hands the parts to specialized sub-agents, each with read-only access to one source. Domain expertise is written as natural-language procedures that plant engineers can read, review and revise. The system is strictly advisory, with every answer traced back to its sources. We also report organizational lessons from introducing agentic AI into a culture built on safety, traceability and conservative change. Beyond the technical architecture, the Beznau case study examines how an assistant can be introduced into a setting where evidence must be traceable, changes carefully reviewed, and operational decisions retained by plant personnel, paving the way for the responsible use of agentic AI in industrial plant monitoring. |
| 2026-11-20 | [Where Delay Compensation Belongs in a Decentralized Mean-Field-Game Swarm with a Control-Barrier Safety Filter](http://arxiv.org/abs/2611.27540v1) | Jan Mikołajczyk, Igor Rostkowski et al. | Large robot teams are often run as a plan-then-execute stack. We study a decentralized instance for drone swarms, a mean-field-game (MFG) routing field with a per-agent control-barrier-function (CBF) safety filter, when communication is limited by a shared TDMA channel. Neighbors are known only through noisy range and bearing measurements exchanged in round-robin slots, so the frame, and with it the age of every estimate, grows with swarm size. In simulation this delay reaches both layers but breaks only the safety layer. Without delay, swarms of up to 100 agents do not collide, even with each measurement's noise held for a whole frame; with it, stale estimates at the CBF cause collisions beyond a frame period of 125 to 175 ms, or 75 to 100 ms at 100 agents with held noise, and swarm size sets where that limit lies and how steeply collisions rise. At the MFG, staleness makes the local density noisier without biasing it, and throughput drops by at most a tenth. We then compensate with lightweight on-board predictors, from constant-velocity fits to a small learned network, that estimate each neighbor's position from its last broadcasts. A layer-split experiment shows where prediction is needed: at the CBF it restores safety and extends the tolerated frame period by 1.4 to 1.7 times with an untrained constant-velocity fit and about 2.3 with the learned network when the noise is redrawn each step; with noise held for a frame the two extend it about equally, while at the MFG it only refines the congestion correction. A constant wider safety margin, the simplest alternative, makes collisions more frequent in a bottleneck. With the routing solved offline, that layer is robust to the delay because it sees neighbors only through a crowd average; compensation belongs in the safety filter, switched on only above the uncompensated limit, since below it the predictor's own error adds collisions. |
| 2026-11-20 | [The Assistant Persona Differs Across Languages in Activation Space](http://arxiv.org/abs/2611.27504v1) | Daniel Yoo, Naama Rozen et al. | Large language models behave differently across languages, and are noticeably less safe in lower-resource languages. This gap is well established behaviorally, but not mechanistically. Directions in activation space, including the assistant persona direction, are almost always extracted from English and assumed to generalize across languages. We extract the assistant persona direction in 31 languages in Qwen3-32B, and find that the resulting axes differ substantially: their cosine similarity to the English axis ranges from 0.378 to 0.911. Moreover, each language's assistant persona direction and its magnitude are correlated with the language's resource level. We also find that the model's default assistant state is farther from the helpful assistant pole in lower-resource languages. Finally, steering against a language's own axis amplifies persona-based jailbreak more than steering against the English axis. These results indicate that the model does not carry a single assistant representation across languages. More broadly, the assistant persona may need to be an explicit target of multilingual alignment. |
| 2026-11-20 | [Trained on Harm, Still Refusing: A Training-Time Defense for Code Generation](http://arxiv.org/abs/2611.27495v1) | Miaoqing Tian, Ruohan Li et al. | Fine-tuning enables safety-aligned code language models to adapt to diverse downstream tasks. However, training on untrusted data can severely undermine their ability to refuse harmful requests. Notably, we find that models comply with harmful requests even though harmfulness information remains highly decodable from their internal representations. Moreover, an initial preference for refusal does not necessarily translate into sustained refusal behavior throughout generation. Together, these findings reveal a critical gap between recognizing harmful requests and consistently refusing to fulfill them. To address this gap, we introduce SAFEHOLD, a novel provider-enforced defense framework that preserves safety alignment during harmful fine-tuning by treating refusal as a sustained generation process rather than a single response decision. Specifically, SAFEHOLD preserves the aligned model's full-vocabulary response distribution at the initial generation position through decision-boundary anchoring, while supervising the complete refusal sequence through refusal-path reinforcement. We conduct extensive experiments evaluating safety, utility, and internal mechanisms. SAFEHOLD reduces average ASR by 72.03 percentage points relative to undefended fine-tuning, outperforming five SOTA defenses by 24.32 to 70.87 percentage points. Meanwhile, it improves average HumanEval pass@1 by 10.24 percentage points over undefended fine-tuning, while maintaining an average XSTest over-refusal rate of only 2.73 percent. Overall, these results demonstrate that SAFEHOLD effectively mitigates safety degradation during harmful fine-tuning while preserving benign-task utility. |
| 2026-11-20 | [Dugong: Composing Uncertified DAG Consensus](http://arxiv.org/abs/2611.27473v1) | George Danezis, Lefteris Kokoris-Kogias et al. | There is a significant number of uncertified DAG consensus protocols spanning partial synchrony and asynchrony as well as crash, Byzantine, and hybrid fault models, yet each protocol is designed, proved, and implemented from scratch. We introduce Dugong, a single parameterized protocol that unifies uncertified DAG consensus protocols. Its core algorithm, the Decider can implement eight existing protocols and yields four new ones by just tuning its parameters. Given the property that DAGs decouple throughput from latency Dugong presents a simple trade-off between latency and resilience to network and node faults. An operator can adapt the parameters of their protocol epoch-by-epoch to adjust the resilience without needing to develop a new protocol or worry about throughput. Around the Decider, Dugong layers production feature mechanisms (adaptive leader scheduling, checkpoints, responsive liveness, and crash recovery) that provably compose with each other and with the Decider's configurations. We formalize Dugong in Lean 4, proving safety and liveness under the established definitions of Byzantine atomic broadcast, and identify and repair gaps in published proofs. Finally, we implement Dugong as a reusable Rust library paired with a discrete-event network simulator, enabling direct latency comparisons across network and fault models. |
| 2026-11-20 | [Poisoning Attacks and Collaborative Defense Framework for LightRAG-Based Construction Regulation Knowledge](http://arxiv.org/abs/2611.27460v1) | Yingyan He | Retrieval-augmented generation (RAG) is widely used in construction, yet the security of building-regulation knowledge bases has received little study. This thesis reports a pre-registered experiment on one fully instrumented LightRAG system built on the UK Approved Documents. Using a 150-item benchmark labeled with safety consequences and tampering directions, it evaluates four attacks (text layer, graph layer, fabricated amendment narrative and a defense-aware adaptive attacker) together with a three-layer retrieval-graph-generation defense in a full factorial design. Text-level poisoning succeeded, with success rising with the budget; graph-level poisoning was less effective, contrary to the registered hypothesis. The full defense met the registered criterion and six of the seven configurations met its threshold, but those containing the generation layer did so by refusing 69-80 percent of poisoned queries, costing up to 74 points of clean accuracy, whereas provenance-based retrieval filtering cost no measurable clean accuracy in the configuration used. A second experimental batch showed that excluding a legitimate unregistered amendment caused no measurable loss of accuracy either, as its wording reached the generator through the graph's entity and relationship descriptions, the route the poison took, which no layer inspects. Interactions were strong, so no layer's contribution can be isolated. A reported zero attack success rate is not evidence of detection unless the refusal rate is reported with it: zero arises either because provenance filtering removes every document outside the manifest without judging its content, or because the system refuses. Layered defenses must therefore be evaluated factorially, and regulatory RAG assistants require corpus admission control and version-aware verification before they can support safety-critical decisions. |
| 2026-11-20 | [Mechanism Parity for Simulation-Based Validation of Autonomous-Driving Perception](http://arxiv.org/abs/2611.27447v1) | Danial Safaei, Siddartha Khastgir et al. | Simulation is central to intelligent-vehicle validation, but visual realism or output agreement between a real scene and its synthetic counterpart does not prove that a perception system used the same evidence in both domains. This paper addresses that mechanism-parity gap for paired real-synthetic autonomous-driving perception. It unifies Decisive-Feature Fidelity, which compares decisive evidence maps in a frozen system under test (SUT), with Concept-Decisive Feature Fidelity, which projects decisive evidence into semantic scene concepts for diagnosis. The framework is a validation diagnostic rather than a standalone safety certificate: it asks whether output-consistent simulation also preserves the mechanism by which the perception model reaches that output. The empirical evidence comprises ten retrained diffusion ControlNet generators evaluated on a deterministic exact-geometry KITTI-Virtual KITTI 2 subset. Generated counterparts use a 1248-by-376-pixel padded inference canvas and are cropped back to the exact KITTI dimensions before evaluation. The final benchmark reports target-image, output-consistency, and mechanism metrics recomputed from a single audited output set, then adds metric-disagreement cases, bootstrap uncertainty, threshold sensitivity, and concept/cache integrity audits. Across the benchmark, output and appearance fidelity are at most weakly and inconsistently rank-correlated with decisive-mechanism distance for all three frozen SUTs, and a constructed positive control shows the estimator detects a known planted mechanism that output agreement misses, supporting decisive-evidence diagnostics for intelligent-vehicle simulation validation. All numerical claims are restricted to the declared exact-geometry benchmark, ten generators, and three frozen SUTs. |
| 2026-11-20 | [Cleave: Severing Guardrails and Auditing Spectral Coupling in Open-Weight LLMs](http://arxiv.org/abs/2611.27426v1) | Feng Huang, Qiang Huang et al. | Open-weight Large Language Models (LLMs) are widely downloaded for their strong coding and reasoning capabilities and deployed locally beyond the provider's control. To prevent misuse, developers embed safety guardrails into these models before release. However, we find that these guardrails rely on the residual writing weights to trigger refusals. Based on this insight, we propose Cleave, a training-free attack that directly edits these weights to remove safety restrictions while preserving general capabilities. With Cleave, malicious users can either remove all guardrails or selectively lift only cybersecurity restrictions. Because this selective removal retains up to a 99% refusal rate on other harmful requests and avoids weight rank collapse, it evades both behavioral spot-checks and static weight audits. Across five open-weight LLMs from 7B to 320B parameters, Cleave achieves up to a 67.97% attack success rate on seven safety benchmarks, substantially outperforming existing white-box baselines. On 356 cybersecurity tasks and 15 live vulnerable services, it further boosts executable exploit generation by up to 24.0 times and compromises up to 66.7% of the targets. To counter this threat, we propose a dual-stage defense that combines pre-release subspace diagnostics with post-release weight-coupling audits, achieving 100% detection recall at a 1% false-positive rate. |
| 2026-11-20 | [Jagged Robustness: Uneven Safety Transfer and Emerging Vulnerabilities under LLM Quantization](http://arxiv.org/abs/2611.27424v1) | Ramneek Agnihotri, Yingte Xu et al. | Quantization enables efficient inference in language models through model parameter compression. This work audits the impact of quantization on capability and safety domains including model safeguards, instruction-following, ethics and bias. Through benchmarks, we find that model safeguards and bias can degrade under quantization, while other capabilities such as truthfulness remain more stable in Qwen3-4B. We define this uneven transfer of LLM capabilities under quantization as Jagged Robustness. We find multiple factors driving Jagged Robustness: task complexity, evaluation method and generation coherence. Moreover, we verify the degradation of model safeguards by observing increased compliance with unsafe requests under quantization, and demonstrate universal adversarial suffix attacks that are benign for the full model but jailbreak quantized models. The findings indicate that it is risky to assume that the performance and robustness can transfer evenly across different capability domains from full models to quantized models. |
| 2026-11-20 | [Testing for Coordination Failures in Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2611.27368v1) | Dennis Gross, Jørn Eirik Betten et al. | Deep parallel cooperative multi-agent reinforcement learning (CMARL)} trains multiple agents that all act at every time step to optimize a common objective. Like single-agent RL, CMARL can produce unsafe behaviors that violate implicit safety requirements not captured by the reward, and these behaviors are hard to explain. However, testing CMARL policies for safety has received less attention than the safety testing of single-agent RL policies. In this paper, we propose DisQ, a test case selection method that identifies failure-revealing states in value-based CMARL by exploiting a signal specific to trained Q-learning agents. The signal is the Q-value gap, an explainable metric that flags a state when the agents' estimates of the achievable return diverge by more than a calibrated threshold. DisQ selects such states for execution and skips those without a gap. Our experiments in four CMARL benchmark environments show that, in most of the evaluated configurations, DisQ improves failure detection over single-agent RL testing methods applied to CMARL, namely random testing and search-based testing with genetic algorithms. Our main contributions are the Q-value gap, an indicator of potential miscoordination computable from a single state without executing the system, and DisQ, a test case selection method that adapts existing single-agent testing methods to CMARL. |

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



