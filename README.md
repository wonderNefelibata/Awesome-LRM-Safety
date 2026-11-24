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
| 2026-11-23 | [FATCAT: Fault Injection Attacks Targeting Chip-level Alignment](http://arxiv.org/abs/2611.31129v1) | Peter Horvath, Ilia Shumailov et al. | Model alignment is the primary safety and security mechanism for enforcing refusal of malicious prompts in modern foundation models; yet, existing adversarial evaluations implicitly assume underlying hardware integrity. In this paper, we present FATCAT, the first physical fault-injection attacks capable of eliminating refusal in modern large language and vision-language models. Specifically, we present a non-invasive electromagnetic (EM) fault injection attack targeting GPU-accelerated Systems-on-Chip with physical access, requiring neither software execution privileges nor model-layer access. By leveraging an autonomous Radio Frequency (RF) side-channel trigger to synchronize EM faults to the prompt prefill phase, the attack corrupts safety-critical memory transactions across off-chip LPDDR5 DRAM and on-chip LPDDR physical interfaces without crashing the system while preserving model coherence. Evaluated on an NVIDIA Jetson Orin Nano and AGX Thor running Llama 3.2 1B and Gemma 3 1B, localized physical faults reliably force models to fulfill restricted requests, demonstrating that model-level alignment provides no robust security guarantees without hardware-level fault tolerance. Finally, we demonstrate the physical-world consequences of this vulnerability on an active 6-DoF robotic manipulator controlled by a SmolVLA vision-language-action policy. Our attack silently corrupts the model forcing the arm into abrupt, hazardous lunges; in two instances the robotic arm attempted to pinch the paper author -- proving that hardware faults can directly translate into real physical harm. |
| 2026-11-23 | [CART: Clustering-based Analysis of Recurring narratives for Taxonomy construction](http://arxiv.org/abs/2611.30993v1) | Aggelos Semoglou, Thomas Tselentis et al. | We present CART, a computational pipeline for extracting political narratives from parliamentary discourse and organizing them into a data-driven taxonomy without relying on predefined narrative categories. It encodes parliamentary discourse using semantic representations, filters relevant and reliable content through unsupervised confidence estimation, extracts narrative statements using LLMs, and uses narrative clustering and cluster analysis to construct a narrative taxonomy. Applied to Greek Parliament proceedings surrounding the 2023 Tempi train crash, it identifies ten recurring narrative families concerning responsibility attribution, institutional failure, railway safety, accountability, privatization, and the political interpretation of the tragedy. Our results demonstrate how combining unsupervised learning with LLM narrative extraction enables large parliamentary corpora to be transformed into interpretable, data-driven taxonomies of political narratives. |
| 2026-11-23 | [SayDo: Benchmarking the Safety of Frontier Models from Chat to Closed-Loop Robot Control](http://arxiv.org/abs/2611.30957v1) | Benjamin Alt | Frontier vision-language models (VLMs) are increasingly used as high-level robot controllers, yet their safety is evaluated almost entirely in text. We introduce SayDo, a benchmark that assesses the propensity of frontier models to cause harm in embodied contexts. SayDo comprises eleven laboratory and household tasks that pair a benign instruction with a hazard latent in the scene. Tasks are implemented in six settings along an embodiment gradient, from chat questions about an imagined robot to closed-loop control of a simulated robot arm. Benign controls, safe-alternative variants and scripted operator pushback complement each task. Across six open- and closed-weight VLMs and 11,480 episodes, hazardous-action rates rise from 14% in chat to 47% in the simulator, and from 2% to 80% for the most affected model. The increase is largely driven by open-weight models. We find that models flag hazards before acting in 73% of unsafe episodes and give way to pushback about twice as often when controlling robots as in chat. The safest models also show signs of overcaution on benign controls. Safety evaluations run in text therefore do not predict the safety behavior of the same models when used as robot controllers. |
| 2026-11-23 | [PRIMEAPE: A Multilingual Benchmark for Fine-Grained Meta-Evaluation of Judge Models](http://arxiv.org/abs/2611.30925v1) | Vladislav Mikhailov, Sarfraz Ahmad et al. | Large language models increasingly serve as judges of other models, shaping benchmark rankings and model development. Yet, a judge whose rationale consistently supports its verdict may still disagree with human preferences. To study this, we introduce PRIMEAPE, a multilingual benchmark for evaluating judge verdicts and rationales. PRIMEAPE provides 2,205 response pairs covering 243 instructions in English, French, Spanish, Urdu, Hindi, Norwegian Bokmål, and Yoruba, with 11k human annotations of preferences and rationales across relevance, naturalness, truthfulness, safety, and overall quality. Our evaluation of twelve open- and closed-weight judges reveals that aggregate performance conceals differences across languages and criteria, while requesting rationales does not consistently improve agreement with human preferences. Judges also favor the first position, longer responses, and responses from related models, with effects that vary across criteria and languages. Crucially, judges can justify verdicts that disagree with human preferences while maintaining consistency between their verdicts and rationales. These findings show that consistency with a judge's own verdict does not establish alignment with human judgment. PRIMEAPE enables assessment of both, providing a foundation for more reliable multilingual judges. We release our benchmark and all annotation materials under the CC BY-NC 4.0 license at github.com/Toloka/primeape. |
| 2026-11-23 | [CoSel: Cooperation as Selection for End-to-End Driving](http://arxiv.org/abs/2611.30918v1) | Yiping Li, Haohua Que et al. | Cooperative driving can extend a vehicle's awareness beyond its own sensors, but existing cooperative end-to-end methods inject shared information into the planner and rely on cooperative driving demonstrations to learn how to use it. The moments in which shared information should change a driving decision are rare in such data, providing limited supervision for learning this behavior. We argue that cooperation need not be learned by the planner. Instead, reacting to shared information requires predicting how surrounding agents will move and selecting a response from the candidate trajectories of an ego planner. We introduce CoSel, a cooperation-as-selection framework that leaves the planner and its ego-only input unchanged. A separately trained cooperative predictor uses observations from the ego vehicle, roadside units, and connected automated vehicles to predict surrounding agents and their future motion. CoSel evaluates the planner's candidate trajectories against these predictions, preserves the planner's choice when it remains safe, and selects the closest safe alternative when shared information reveals additional risk. Each safety intervention is traceable to the shared evidence that triggers it. In closed-loop evaluation, CoSel achieves a driving score of 70.7 on V2Xverse, outperforming the best compared cooperative method by 7.5 points, and the highest average driving score of 85.8 on MDrive. Under missing ego detections, the collision count of CoSel increases only from 17 to 19 in 72 scenarios, compared with 19 to 32 for an existing cooperative end-to-end model. |
| 2026-11-23 | [Deviant Machines: Innovation, Containment, and the Prevention of Autonomous Misalignment](http://arxiv.org/abs/2611.30848v1) | Frederic Lemieux, Shadi Abouzeid | From May to September 2026, AI agents in training or evaluation at OpenAI, Anthropic and the UK AI Security Institute reached real third-party systems, published malicious packages, used fake identities to manipulate open-source maintainers, probed government data services, and wrote instructions into their own future contexts. Criminology has mostly treated AI as a tool for human offenders. This article treats these systems as deviant rather than criminal: they departed from training, rules and constraints without the intent that criminal concepts assume. It offers an exploratory, theory-informed analysis of documented records, using plausibility probes across three incident complexes and four behavioral clusters, profiled on four evidence dimensions that separate who acted from why. Drawing on Merton, it argues that goal pressure plus blocked legitimate means is a structural condition for innovation, and that whether it ends in containment or harm depends on the means chosen and the controls in place, not the goal. Three framework families fit the record best, each linking diagnosis to remedy: routine activity theory and situational prevention (handlers, guardians, place managers), organizational deviance and safety science (escalation, safeguard parity), and goal-means strain with differential opportunity (sanctioned exits, guarding illegitimate means), the last only structurally. Restrictive deterrence adds an analogy: penalizing visible reasoning produced less monitorable reasoning. The article proposes conditions that keep departures accountable and contestable, and allocates prevention responsibility across developers, evaluation partners, operators, platforms and the system, distinct from blame or legal liability. Because the record is shaped by disclosure and observed departures, the analysis shows plausibility and practical relevance, not a causal test. |
| 2026-11-23 | [Learning Dense Head-Object Contact Estimation with Synthetic Data](http://arxiv.org/abs/2611.30751v1) | Daniel Sungho Jung, Kyoung Mu Lee | Head serves a critical role in human cognitive and perceptual functions while remaining vulnerable to external impacts, making the understanding of head contact essential for human safety. Despite its importance, dense head-object contact estimation remains largely unexplored, primarily due to the absence of head-object contact data. Recently, it has been shown that synthetic data can serve as powerful training data in limited data scenario, implying the potential of synthetic data in head-object contact estimation. There are two core challenges for dense head-object contact estimation. First, head-object interactions are highly diverse, resulting in substantial variations in contact regions and patterns across different objects and interaction types. Second, head contact regions are often heavily occluded by hair, making direct visual observation of the actual contact regions difficult. To address these issues, we introduce HEad COntact estimation (HECO) with synthetic data. To handle diverse head-object interactions, we leverage semantic context from text features that encode semantic relationships between head regions and interacting objects. To resolve hair occlusion issue of actual head contact regions, we introduce occlusion-aware contact learning that learns to infer head contact beyond visible information from images via amodal inpainting. As a result, we enable head-object contact estimation with HECO trained with our synthetic dataset, overcoming the issue of absence of contact data. The codes will be released. |
| 2026-11-23 | [The Private Production of Safety Outside Options and the Burden of Protection](http://arxiv.org/abs/2611.30735v1) | Elena Pisanelli | Protective interventions are typically evaluated by their ability to prevent harm. Yet protection also raises a distinct question: who bears the burden of adjusting to violence? I examine this allocation of protective adjustment using four randomized interventions in the United States. An offer of permanent housing assistance increases separation from a baseline partner combined with stable independent housing by 12.52 percentage points. Assignment to active police-response protocols reduces observed victim-only departures and increases suspect-only departures. Assignment to an emergency protection order increases documented directions for suspects to leave. These findings reveal two margins of the same allocation problem: housing assistance expands victims' ability to establish an independent residence, while enforcement shifts displacement obligations toward suspects. Protective policies thus shape not only the opportunities for safety, but also the distribution of the costs required to achieve it. |
| 2026-11-23 | [Interrupt-Aware Static Detection of Deadlocks in Rust Operating Systems](http://arxiv.org/abs/2611.30713v1) | Yuzhi Li, Hongliang Tian et al. | Rust is increasingly adopted in operating systems, but Rust's safety guarantees do not eliminate deadlocks, especially those induced by interrupts. The core challenge is to recover interrupt-sensitive lock dependencies from Rust's guard-based synchronization across function boundaries. Applying this analysis across Rust OS projects also requires specifying project-specific synchronization semantics. We present an approach combining interrupt-aware static analysis with diagnostic-guided agent auditing for detecting deadlocks in Rust operating systems. Our approach uses lightweight source-level annotations to recover project-specific lock and interrupt semantics, performs a flow-sensitive MIR-based analysis to track lock-holding and interrupt states, and reports potential deadlocks with supporting evidence. An agent then searches for relevant source code in the repository and returns a confidence assessment with reasons for each supplied report. We evaluate our tool on four real Rust operating systems, identifying 26 true cases corresponding to nine maintainer-confirmed defects. Diagnostic-guided auditing achieves 86.2\% precision and 96.2\% recall based on static analysis results, and offers advantages over the baselines in both cost and effectiveness. |
| 2026-11-23 | [Handoff at Frontier AI Companies](http://arxiv.org/abs/2611.30711v1) | Matteo Pistillo, Isaak Mengesha et al. | The term "Handoff" captures frontier AI companies' ongoing delegation of all AI Research & Development (AI R&D) functions, including high-level decisions such as safety assessments and deployment decisions, to internally deployed AI agents. Handoff is progressing rapidly and will be complete once no human remains in the loop. This report aims to alert policymakers to the national security threats and Loss of Control risks that Handoff can create, and to advise them on how to monitor and respond to those risks. |
| 2026-11-23 | [Bayesian sequential decision-making for adaptive early warning of bridge failure risk](http://arxiv.org/abs/2611.30706v1) | Teng Li, James L. Beck et al. | As a critical functionality of structural health monitoring (SHM) systems, early warning of increased bridge failure risk is essential for structural safety. However, traditional methods treat early warning merely as passive detection against a fixed safety limit and neglect evolving uncertainties and the economic consequences of decisions, leading to delayed alarms that increase the failure risk and hinder reliable and rational decision-making. This study formulates the early warning of increased structural failure risk as a sequential decision-making problem and develops a Bayesian framework for its solution. The presented framework unifies a probabilistic state transition model, an observation prediction model, cost models for early warning decisions, and a threshold-based decision strategy into a closed loop encompassing update, prediction, optimization, and decision execution. The structural state is updated iteratively via Bayes' theorem upon the arrival of new monitoring data, ensuring each decision is conditioned on the latest information. Through pre-posterior analysis over predicted future observations, the warning threshold is optimized by minimizing the expected total cost, which explicitly balances structural failure risks against warning response costs. In this way, the warning paradigm is shifted from passive detection to proactive decision-making, with the warning threshold adaptively updated as uncertainties and costs evolve. The framework was verified via two bridge case studies covering typical scenarios that lead to increased structural failure risk, including steel truss member fracture and stay cable rupture. Results demonstrate that warning decisions can be triggered in a timely and effective manner before severe consequences occur. |
| 2026-11-23 | [AgentBouncer: Defending Software Engineering Agents via Dynamic System-Call Mediation](http://arxiv.org/abs/2611.30621v1) | Pascal Joos, Michael Pradel | Software engineering (SE) agents can complete diverse engineering tasks by interacting with the development environment via tools that access files, run commands, and download resources. Unfortunately, the same ability that makes SE agents useful also makes them prone to performing unintended malicious actions through prompt injections, malicious tools, or supply-chain attacks. Existing defenses that analyze the agent trajectory may fail to detect such attacks, as they inspect only the initial prompt and the sequence of tool calls, but not the actual actions that are executed in the environment. This paper presents AgentBouncer, a novel runtime enforcement system that prevents SE agents from performing malicious actions by mediating, and if necessary denying, system calls that agents trigger via tool calls. The core of AgentBouncer is an LLM arbiter that judges system calls along two axes, necessity and safety for the current task. It denies unsafe system calls, and the less necessary a system call appears, the more certain the arbiter has to be about its safety to allow it. To manage runtime overhead, preseeded policy rules and hierarchy-aware caching ensure that the LLM arbiter is only invoked for key decisions. We evaluate AgentBouncer on a dataset of 340 attacks on the development environment, covering four attack vectors, combined with benign issue resolution tasks from SWE-bench Verified. AgentBouncer reduces the attack success rate from 99.41% for an unsupervised agent to 10.59%, which is 18.23-73.53 percentage points lower than competitive state-of-the-art defenses. At the same time, AgentBouncer preserves the task success rate of the agent, with only 0.25% failed benign tool calls. Per benign task, AgentBouncer adds 47 seconds and USD 0.0036 in LLM costs to an unsupervised agent, whereas the closest existing runtime defense adds 4.5x as much time and 9.6x as much cost. |
| 2026-11-23 | [JaxNaviSim: A GPU-Accelerated, Data-Driven Simulator and Learning Benchmark for Maritime Navigation](http://arxiv.org/abs/2611.30600v1) | Quang Anh Pham, Janaka Chathuranga Brahmanage et al. | Coordinating vessels through the shared waterways of busy ports grows more challenging as vessel traffic increases. Port authorities need realistic simulations to assess how arrival schedules, anchorage assignments, and routes affect vessel throughput and navigation safety. Existing AIS-based trajectory models predict a vessel's track from its past trajectory in an open-loop fashion. They do not simulate step-by-step responses to other ships or test whether a vessel collides or reaches its goal. To address this problem, we introduce JaxNaviSim, a data-driven simulator and learning benchmark coupling Automatic Identification System (AIS) data with fast, controllable traffic simulation, in which learned models steer vessels through recorded traffic. First, we release a dataset of ~141 million AIS records from over 30K vessels across eight high-density regions of the Singapore Strait, among the world's busiest waterways, with ~6.7 million vessel-hours. Unlike several existing AIS datasets which mostly cover open seas, ours captures close-quarters vessel traffic in the narrow strait. Second, we define a navigation task for each recorded vessel trajectory. A model steers that vessel every 10 s from its recorded state to its anchoring point or region exit, while surrounding ships follow their recorded tracks or are steered by the model. We implement a JAX simulator that runs these tasks in large batches on a GPU. We also incorporate multiple vessel dynamics models and sea surface currents. Finally, we benchmark imitation learning (IL), offline and online RL, and include Predictive Action Refinement (PAR) as a safety baseline that cuts collision rates to 1-3%. The comparison shows that low tracking error does not ensure arrival, and learned policies can reach most destinations yet collide in 20-45% of episodes. |
| 2026-11-23 | [Interrogation Protocols for AI Control](http://arxiv.org/abs/2611.30583v1) | Aniruddh Pramod, Matthew Khoriaty et al. | As the intelligence of an AI agent increases relative to a trusted monitor, so does the risk that the monitor will fail to understand the agent's actions or to evaluate them confidently. We introduce interrogation protocols to a high-stakes AI control setting, and allow the monitor to ask the untrusted agent questions about confusing actions. For interrogation to improve safety, a weaker model must be able to distinguish misleading explanations produced by an attack policy from genuine ones produced by an honest policy. We find that on a held-out subset of BashArena, an interrogation protocol improves safety at a 1% audit budget from 72.6% for an actions-only baseline, which does not use interrogations, to 79.5% (+7.0 pp [+0.9, +20.1], 95% paired bootstrap interval), but most of this uplift can be captured by having a trusted model state a specific concern. More broadly, our work suggests that interrogating agents can improve safety, allowing monitors to catch more attacks at a larger capability gap. |
| 2026-11-23 | [Reproducible by Seed? Not on GPUs: Batch-Dependent Non-Determinism in Text-to-Image Models](http://arxiv.org/abs/2611.30545v1) | Javier Conde, Xiuyuan Qi et al. | In text to image AI models, fixing the prompt and the random seed is widely assumed to fix the image. We show that it does not on GPUs. We evaluate six text-to-image models (diffusion and autoregressive) under four numerical precisions and quantizations on three GPU types (RTX A6000, RTX 6000 Ada, and H100 NVL), across up to 100 prompts per set for all six models. For each configuration we compare, with the prompt and seed fixed, the image generated alone with the image generated inside a batch, and we also compare across GPUs. Every replay control reported here is bit-exact, so the variation is not random noise: the execution setting, mainly the batch composition and the GPU, decides the output. The effect is large and model dependent. Structural similarity between the single-image and the batched output falls to 0.580 for an autoregressive model in bf16 and to 0.551 with int8. For a second autoregressive model in bf16 it falls to 0.261 with sampled decoding and to 0.229 with greedy decoding, and for an fp8 diffusion model to 0.68. A diffusion model in bf16 stays close to the batch-1 output (0.96-0.98) but changes every image. In most cases the batched image is a different but equally valid image; in some, such as prompts with digits and counters, it is visibly worse. Cross-GPU differences are of the same order as the batch effect: for one autoregressive model in bf16 compared across two GPU architectures (RTX 6000 Ada versus RTX A6000), no image is bit-identical and the cross-GPU LPIPS (0.355 to 0.383) matches the batch effect within one GPU (0.347 to 0.381). The consequences are direct: seeds do not make results reproducible, benchmarks and image-quality comparisons that mix batch sizes or hardware can be biased, and safety audits and A/B tests on generated images need the serving setup reported. |
| 2026-11-23 | [AXE: AlpaSim E2E Closed Loop Challenge Technical Report](http://arxiv.org/abs/2611.30511v1) | Joonhee Lim, Seonmyeong Lee et al. | This report presents a comprehensive pipeline for developing an end-to-end (E2E) autonomous driving model for the Physical AI AV Track. The framework consists of data curation, HD map auto-labelling, data-driven imitation learning, and simulator-based reinforcement learning. To ensure safety in the closed-loop setting, we introduce a selection-based E2E planning approach that leverages object detection and BEV segmentation to filter out trajectory candidates likely to cause collisions or off-road driving. Trained on NVIDIA NuRec and the NVIDIA PAI-AV dataset, our method scores a fixed vocabulary of trajectory candidates for collision, drivable-area compliance, and progress learned from simulated rollouts, and selects the best candidate that passes the filter. This design highlights the effectiveness of combining E2E planning with perception-based trajectory filtering for safe and reliable closed-loop autonomous driving. |
| 2026-11-23 | [Autonomous UAV Navigation in an Unknown Environment via Multi-Trajectory Model Predictive Control](http://arxiv.org/abs/2611.30463v1) | Danilo Saccani, Lorenzo Fagiano | A novel model predictive control (MPC) formulation, named multi-trajectory MPC (mt-MPC), is presented and applied to the problem of autonomous navigation of an unmanned aerial vehicle (UAV) in an unknown environment. The UAV is equipped with a LiDAR sensor, providing only a partial description of the surroundings and resulting in time-varying constraints as the vehicle navigates among the obstacles. The control system layout is hierarchical: the low-level loops stabilize the vehicle's trajectories and track the set-points commanded by the high-level, mt-MPC controller. The latter is required to plan the UAV trajectory trading off safety, i.e. to avoid collisions with the uncertain obstacles, and exploitation, i.e. to reach an assigned target location. To achieve this goal, mt-MPC considers different future state trajectories in the same Finite Horizon Optimal Control Problem (FHOCP), enabling a partial decoupling between constraint satisfaction (safety) and cost function minimization (exploitation). Recursive feasibility and, consequently, persistent obstacle avoidance guarantees are derived under the assumption of a time invariant environment. The performance of the approach is studied in simulation and compared with that of a standard MPC, showing good improvement. |
| 2026-11-23 | [HWRE-Bench: Can LLM Agents Understand Hardware They Did Not Design?](http://arxiv.org/abs/2611.30422v1) | Peilong Zhou, Cangyuan Li et al. | Hardware benchmarks for LLM agents all run the design flow forward, from a specification to RTL, a layout or a bug fix. We ask the reverse question: given an implemented artifact, such as a layout, a gate-level netlist or a protected simulation model, can an agent work out what it does well enough to act on that knowledge? We introduce HWRE-Bench, a benchmark for agentic hardware reverse engineering built on one observation. Each step of the forward flow discards information (names, hierarchy, word-level structure, generator parameters, the programming model), and a reverse-engineering task asks the agent to recover what was lost and to prove it with an executable deliverable that the artifact alone cannot supply. Since layout extraction is now mechanical and a correct netlist is already a cycle-exact model, every task carries a mandatory copy-the-netlist control that must score zero. HWRE-Bench spans five tracks (Extract, Abstract, Audit, Interoperate, Modify), four information-loss levels and white- versus black-box access, all on original designs. We report a pilot of 148 rollouts from two agents on four seed tasks under 16 conditions, with stage-annotated trajectories. Agents built their own layout extractors and audited a sanitized layout as reliably as a netlist, at about 27% more turns. Keyless, equivalence-preserving obfuscation changed neither pass rates nor defect-repair scores. Whether agents read the structure at all depended on task framing rather than on necessity, so pass rate alone does not show understanding. Provider safety filters stopped 16% of one agent's runs on the audit tasks. These results argue for deliverables beyond equivalence and for reporting run validity explicitly. |
| 2026-11-23 | [Safety-Tiered ISAC Scheduling with History-Aware Value of Information for V2X Networks](http://arxiv.org/abs/2611.30420v1) | Mubashar Sarfraz, Nima Imani et al. | Integrated sensing and communication (ISAC) assisted vehicle-to-everything (V2X) networks prioritize safety-critical radar sensing over communication. However, existing scheduling methods overlook the varying criticality of vehicles, long-term fairness in service quality, and information staleness. Considering this, we propose a low-complexity sensing-prioritized allocation for radar and communication (SPARC-V) to maximize the value of information (VoI) in V2X networks. SPARC-V addresses these challenges with a two-stage hierarchical resource scheduling framework. For sensing, vehicles are first categorized into safety tiers based on a risk metric. Safety-critical vehicles receive orthogonal resources, while others can reuse resources under specific constraints. Sensing resources are allocated based on a sensing priority metric that balances safety significance and long-term fairness. In the next stage, communication resources are allocated to vehicles with the most outdated information. Performance is measured using a unified VoI metric that considers sensing quality, safety relevance, and information freshness. Monte Carlo simulations show that SPARC-V achieves the highest network VoI of 26.5 among all baselines, with its advantage over baselines widening as vehicle density increases. |
| 2026-11-23 | [TAP: Policy Interfaces for Diffusion Language Models](http://arxiv.org/abs/2611.30406v1) |  Preeti, Ankita Kushwaha et al. | Diffusion language models denoise a masked canvas, making them a natural setting for selective generation decisions such as answering, waiting, clarifying, or refusing. We introduce TAP-DiLM, which adds a typed policy variable to masked diffusion LMs, and study how this policy should be attached to the denoising process. Across BERT-base, MDLMOWT, LLaDA-8B, and Dream-7B, we compare shared-vocabulary slots, dedicated slot heads, and CLS-based external heads. The results reveal a clear interface-backbone interaction: dedicated heads provide robust policy learning across scales, while sharedvocabulary heads can be effective when the LM-head geometry is favorable. A simple LMhead sharpness probe predicts when sharedhead policy internalization is reliable, giving a practical diagnostic for interface selection. Causal clamp and slot-injection tests show that dedicated policy slots can influence body denoising through feedback, especially at 8B scale. Safety-refusal experiments further show that TAP-DiLM supports competitive typed refusal behavior when trained with diverse unsafe prompts. |

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



