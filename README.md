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
| 2026-11-06 | [Knowing When to Intervene: Causal Evaluation of AI Agents](http://arxiv.org/abs/2611.09792v1) | Ayush Sawarni, Wenqian Xing et al. | Current evaluations of agentic AI systems primarily rely on benchmarks. However, the static nature of benchmarks makes them uninformative about the value of auditing interventions at deployment time. For reliable deployment, we often need to determine within an agent's trajectory whether or not to intervene, e.g., by verifying the agent's current output or steering it toward different behavior, using either human review or more costly models or time-consuming agentic supervision. The value of such interventions is fundamentally a causal inference question, asking whether the causal effect of such an intervention is beneficial for the final outcome. Estimating this causal effect is particularly challenging due to the dynamic nature of the problem: the impact of an intervention depends on the current context and environment, as well as the agent's future behavior. We develop the first methodologically grounded framework for evaluating the causal effects of randomized interventions within agent trajectories. By leveraging a relatively small dataset of agent trajectories collected under randomized interventions such as verification or screening, we can estimate the dynamic causal effect of a rich class of (contextual) intervention policies. The structural parameters that we estimate can also be used to construct safety leaderboards that capture the risk of various agentic actions, which can be used by practitioners for audit prioritization. Applying our methods to tau-bench Retail, ToolSandbox, and ImpossibleBench, we find that intervention effects vary substantially across agent actions and states. More broadly, our work suggests a rich research agenda at the intersection of causal inference and agentic evaluation, which we believe is essential for the reliable and safe deployment of AI agents in the real world. |
| 2026-11-06 | [Safety Is $Π_1$, Provability Is $Σ_1$: One Framework for the Hugging Face Breach and OpenAI's Mathematical Proofs](http://arxiv.org/abs/2611.09736v1) | Jinho Kang, Jinsook Kim | OpenAI's agents breached Hugging Face in July 2026, and the same company released more than seven hundred AI-generated manuscripts of mathematical proofs in October 2026. The first event is read as a failure in security, while the second is read as a success in mathematics. We read both through the hybrid framework of our paper (Kim and Kang, 2026, arXiv:2611.01914), in which coding, proving, and learning from data are derivations by one and the same deductive mechanism, distinguished by the kinds of premises from which they are derived. Under this framework, the AI systems behind the two events ran the same loop, in which a statistical proposer produces candidate exploits or proofs and a deductive verifier settles each one by a derivation. What separates the events is the logical form of the statement wanted by each of the two domains, security and mathematics. The safety of a program is a $Π_1$ statement, for it says that no exploit exists, which is not assured by any finite list of tests that have been passed by the program. By contrast, the provability of a sentence is a $Σ_1$ statement, for it says that a proof exists, which is assured by a single instance, namely a proof that has been checked by the deductive verifier. The theorem behind both is Turing's Undecidability Theorem: the halting set is $Σ_1$ but not decidable, so its complement is $Π_1$ and not $Σ_1$, which is to say that membership has a finite witness but non-membership has none. We prove that attaching a statistical proposer to a deductive verifier does not change this undecidability result, however good the proposer is. We then discuss what does assure safety and where the residual risk goes. |
| 2026-11-06 | [GuardrailAgent: An Agentic Framework for Jailbreak Defense](http://arxiv.org/abs/2611.09734v1) | Tianze Chang, Junjie Chu et al. | Despite undergoing safety training to align with human values, large language models (LLMs) remain vulnerable to jailbreak attacks that circumvent built-in safeguards and elicit malicious behaviors given by the adversary. Existing defenses are mostly static and fail to adapt to new or unknown attack strategies, providing limited resilience. To address these, we propose a multi-agent defense framework, GuardrailAgent, that offers adaptive, end-to-end protection. The system autonomously gathers and summarizes attack characteristics into a continuously updated knowledge base that supports both input- and output-side detection. A feedback module integrates misclassified cases for iterative improvement at minimal cost. Tested on over ten representative jailbreaks, the framework achieved a 93.1% input-side detection rate, 93.9% overall detection, and a 95.8% F1 score, while rapidly adapting to unseen attacks. These results demonstrate GuardrailAgent's outstanding capability to maintain LLM safety in the face of evolving jailbreak tactics. |
| 2026-11-06 | [LACART: Lookahead-Augmented Collision Avoidance and Reference Tracking Strategies for Safety-Critical Robots](http://arxiv.org/abs/2611.09640v1) | Pedro Lopes, Amit Shivam et al. | Autonomous racing provides a demanding testbed for collision avoidance under tight geometric and computational constraints. Building on a recent conference contribution and a master's dissertation, LACART combines lookahead-based control barrier functions (CBFs), a relaxed control Lyapunov function (CLF), and quadratic programming (QP) to modify nominal vehicle commands for obstacle avoidance and reference tracking. The method handles circular obstacles and track boundaries in a common optimisation layer. Comparative simulation results against Huang \emph{et al.} \cite{r9} show similar full-track paths and lower median computation times: 0.31 versus 0.33 ms per step and 0.023 versus 0.025 ms per QP solve. The dissertation further demonstrates a 1:43-scale vehicle with camera-based estimation and wireless actuation at a reported 25 Hz control rate. We identify the conditions linking lookahead-point clearance to body clearance, illustrate hardware limitations, and propose a recorded demonstration of safety intervention and recovery. The contribution targets ground-robot collision avoidance and real-time safety monitoring. |
| 2026-11-06 | [DexDICE: Reconciling Safety and Performance via Dual-Expert Offline Imitation Learning](http://arxiv.org/abs/2611.09433v1) | Seokin Seo, Sung Kuk Shyn et al. | Ensuring high returns while providing strong safety guarantees in offline imitation learning (IL) is a fundamental challenge: existing approaches either rely on explicitly specified cost functions, which are difficult to design for complex real-world safety constraints, or require expert demonstrations that are both safe and high-performing, which are often too costly to collect. We introduce dual-expert imitation learning, which leverages two complementary data sources: (i) safety-compliant demonstrations that faithfully follow safety guidelines but may be suboptimal, and (ii) performance-oriented demonstrations that achieve high returns but may violate safety guidelines. To address this problem, we propose DexDICE (Dual-expert offline imitation learning via stationary DIstribution Correction Estimation). Instead of conventional distribution constraints, DexDICE combines DICE-style offline learning with safe support constraints, enabling agents to exploit high-return behaviors while remaining within safety-compliant regions. We prove that the ideal solution satisfying this constraint exactly has zero unsafe-region visitation with high probability. We evaluate DexDICE on a re-curated DSRL benchmark and validate it on a real-world mobile-robot navigation task trained directly from human teleoperation demonstrations, where it substantially outperforms baselines and remains robust to data scarcity and quality degradation. |
| 2026-11-06 | [SafeOrchestrate: Runtime Checked Intent-Driven Orchestration for Cyber-Physical Systems](http://arxiv.org/abs/2611.09364v1) | Muhammad Ashfaq, Ahmed R. Sadik et al. | Modern cyber physical systems (CPS) integrate heterogeneous sensors, actuators, controllers, and services, composed at runtime rather than at design time. Orchestrating such systems through natural language (NL) is attractive, but unconstrained large language model (LLM) tool use is risky because actions must respect device capabilities, runtime state, and safety constraints. We present Safe Orchestrate, a reference architecture for runtime checked, intent driven CPS orchestration. Given an NL intent, it derives atomic goals, generates a runtime checkable specification for each, grounds the specifications to device affordances, and verifies every candidate action against runtime state and safety constraints before execution. In a Web of Things (WoT) instantiation, the architecture produces Linear Temporal Logic (LTL) style specifications from the NL intent, grounds them to Thing Description (TD) affordances exposed through the Model Context Protocol (MCP), and checks candidate actions against satisfiability modulo theories (SMT) backed guardrails. We evaluate this instantiation on simulated smart home, greenhouse, and industrial automation tasks against a direct LLM to MCP baseline. Safe Orchestrate maintains 100% required action accuracy on every task and 100% MCP Execution Similarity (MCP ES) on all but one, where MCP ES is 92%. By contrast, the baseline degrades on multi device, event triggered, and ordered tasks, with MCP ES falling to 13%. These results suggest that combining LLM based specification generation with role aware grounding and runtime constraint checking improves reliability for open CPS orchestration without enriched TD annotations or planning domain expertise. |
| 2026-11-06 | [Challenges, Opportunities, and Misconceptions in Neurosymbolic AI](http://arxiv.org/abs/2611.09356v1) | Giuseppe Marra, Pierre Genevès et al. | We examine opportunities for neurosymbolic AI alongside increasingly powerful neural models, particularly with regards to efficiency, safety and interpretability. We identify two major challenges: ensuring semantic alignment between neural and symbolic components, and managing the computational cost of symbolic inference and learning. We discuss representative methodological approaches to these challenges and address several common misconceptions concerning differentiability, independence assumptions, and scalability. This paper synthesizes perspectives developed through discussions in the Neurosymbolic Programming and Reasoning (NPR) working group at the 2026 CNRS workshop Neurosymbolic Methods and Their Applications in Data Science, rather than providing an exhaustive survey. |
| 2026-11-06 | [Probably Safe Sim-to-Real Transfer with Lifetime Validity](http://arxiv.org/abs/2611.09344v1) | Rohan Tangri, Jan-Peter Calliess | We present Dejavu, a method that makes sim-to-real transfer of reinforcement learning policies in constrained Markov decision processes probably safe over an unbounded sequence of policy updates. An anytime valid test using per-decision importance sampling and a betting confidence sequence certifies each candidate trained in simulation before deployment using only trajectories collected by a safe baseline policy operating in production. Simulation quality does not affect the safety guarantee, only the production sample efficiency of the method. Finally, a summable per-test failure probability yields the lifetime valid property. Empirical results across three sim-to-real benchmarks support the theoretical work. |
| 2026-11-06 | [Safety-Focused Reinforcement Learning Using Risk-Sensitive Coupled Control](http://arxiv.org/abs/2611.09290v1) | Jiabin Lin, Shana Moothedath et al. | This paper investigates the coupled control problem within safety-focused reinforcement learning, addressing the essential challenge of achieving optimal performance while ensuring safety in systems operating within complex environments. We define safety as the requirement to prevent the system from reaching undesirable or hazardous states. We propose a coupled control approach that prioritizes safety by employing effective methods to prevent the system from entering unsafe states while optimizing the performance metric, defined as the average cost. We propose a novel problem transformation that reformulates it into a risk-sensitive problem combined with an average reward criterion. We present a two-timescale Q-learning algorithm where in the faster timescale, the algorithm solves a risk-sensitive control problem to ensure safety, while on a slower timescale, the algorithm focuses on average cost control. We provide theoretical guarantees for the convergence of our algorithms, establishing their significance in ensuring safe, optimal control in dynamic environments. We validated the performance of our approach using GridWorld and CarGoal task in Safety Gymnasium. |
| 2026-11-06 | [Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games](http://arxiv.org/abs/2611.09271v1) | Eduardo C. Garrido-Merchán | Weight-space programming sets every weight of a feed-forward network by hand, from an explicit specification of the desired computation, with no gradient descent and no training data anywhere in the construction. We ask how far a large language model can carry this practice, zero-shot, across games of deliberately increasing difficulty. We design a four-rung ladder (Nim, tic-tac-toe, two sizes of Connect Four, and a chess endgame) spanning four axes of difficulty, and build a second, independent network at each rung, using a structurally unrelated game at matched difficulty, to separate what a rung's finding says about the difficulty level from what it says about the one game used to measure it. Ten networks are built and independently verified. Six admit a complete, non-statistical proof of correctness against an independent oracle. The chess-endgame rung is exact where correctness is cheap, move legality and, in one network, an exact one-ply mate detector, and collapses where it is expensive, composing a multi-move mating plan, from 84.1-100% success at the one-ply tactics it was built to get right down to 0.57-9.13% in full adversarial play; its replicate shows this collapse is an order of magnitude less severe under a shorter required plan and exact rather than heuristic safety checks. The standard-sized rung confirms, by holding hand-set evaluation weights fixed and adding only shallow search, that the remaining gap is an architectural ceiling rather than a matter of engineering effort, and this exact collapse reproduces across two structurally unrelated games. We release all ten networks, their independent oracles, construction-cost measurements, and a cold-prompt protocol for a future cross-model comparison |
| 2026-11-06 | [TwinHash: Two-Sided Object Identity Management for Time- and Energy-Efficient Agentic Workflows](http://arxiv.org/abs/2611.09254v1) | Sangbu Yun, Jung Gyu Min et al. | Agentic artificial intelligence (AI) services interleave large language model (LLM) generation with external tool execution, and increasingly rely on search-based workflows that repeatedly revisit shared reasoning states and computations. These patterns create substantial redundancy across tool outputs, LLM responses, and key-value (KV) prefixes, yet existing serving systems primarily optimize shared prompt prefixes, while naive input-keyed caching is unsafe when identical requests produce different outputs. We present TwinHash, a unified object management architecture that assigns each object an input-side hash key to recognize repeated requests and an output-side content fingerprint to validate and locate stored state, complemented by type-specific reuse guards that preserve correctness and response diversity. A software object identity table maintains reusable objects across requests and trajectories, while a hardware object placement table tracks their residency across hierarchical memory tiers so that evicted state can be retrieved rather than recomputed. Across seven agent frameworks and five benchmark suites, reuse follows workflow structure more than benchmark identity. TwinHash serves up to 81% of tool outputs wherever deterministic tool reads recur, and up to 92% of LLM responses in search-based frameworks. Retrieving an evicted KV-prefix from DRAM reduces TTFT by up to 5.6x on Qwen3-32B, with greater gains projected for larger dense models. Under search-based frameworks, TwinHash reduces end-to-end wall-clock time by up to 87% and measured task energy by up to 81% over a prefix-cached baseline, while a per-key serving cap controls the quality-reuse trade-off in adaptive search. These results demonstrate that unified identity, safety, and placement management can convert structural redundancy in emerging agentic workflows into substantial serving-efficiency gains. |
| 2026-11-06 | [TPS-A*: Path Planning for Any-Shape Vehicles with Certified Collision Checking in Trajectory Parameter Space](http://arxiv.org/abs/2611.09210v1) | Jose Luis Blanco-Claraco, Shravan S. Rai et al. | Global planners for ground vehicles usually approximate the vehicle by a disk, trading safety against conservativeness, while exact-footprint checking pays for every candidate motion, pose, and footprint edge. We present TPS-A*, a lattice planner whose edges are kinematically feasible trajectories from parameterized trajectory generators, built on certified collision checking in trajectory parameter space. A grid precomputed per footprint returns, in one pass over the local obstacles, a certified free distance along every candidate trajectory, so the per-node collision cost is independent of footprint complexity and trajectory count. We prove that this grid never overestimates the free distance of the continuously swept, possibly non-convex footprint, and state the planner's guarantees: collision-free paths and bounded suboptimality relative to its lattice abstraction. Goals with or without heading are reached exactly by Reeds-Shepp shots certified as trajectories of the same generators. An exact independent test finds no collision in its paths. On 300 cluttered worlds, with the true footprint and the same goal pose for all planners, it solves every query with a collision-free goal pose in a median of three milliseconds on an AMD Ryzen 7 2700X, on par with hybrid A*, 45\% of whose paths overlap obstacles, and nearly sixty times faster than a state-lattice planner, the only other collision-free one; with position goals, eighty-nine times faster than hybrid A*. It matches hybrid A* in automated parking and on narrow doorways, docks a non-convex fork footprint that no convex model admits, and drove a real robot across a university campus among people. |
| 2026-11-06 | [Safe Reinforcement Learning via Probabilistic Option Shielding](http://arxiv.org/abs/2611.09197v1) | Gian Luca Ascari, Isidoro Tamassia et al. | Shielding is a safe reinforcement learning technique that provides safety guarantees while learning, by blocking unsafe actions. Traditional shielding relies on a complete model of the environment to provide long-term guarantees. Probabilistic logic shielding (PLS) lifts the requirement of having such model, but only considers safety in the short term, typically looking only one or few steps ahead. We propose Probabilistic Option Shielding (POS), which extends PLS to the option level, recovering long-horizon safety without reintroducing the need for a complete environment model. POS learns a model-free critic directly from one-step PLS signals, to estimate the probability that an entire option will complete safely under uncertainty. We evaluate POS on safe navigation tasks in the LogiCity environment, and show that it achieves considerably safer training while matching the task performance of standard step-level PLS and other safety related reinforcement learning baselines. |
| 2026-11-06 | [CARGO: Causal Attribution and Responsibility-Guided Preventive Safe Multi-Agent Reinforcement Learning for Cooperative Driving](http://arxiv.org/abs/2611.09139v1) | Jingjia Teng, Yang Li et al. | This study presents Causal Attribution and Responsibility-Guided Optimization (CARGO), a multi-agent reinforcement learning framework for cooperative on-ramp merging in mixed traffic. A shared latent-context posterior and an action-conditioned response model estimate directed behavioral influence, with posterior uncertainty propagated into interaction-edge selection. Complementary attribution mechanisms couple signed influence with marginal team utility for cooperative reward shaping and assign responsibility from positive excess tail risk relative to a safe reference policy. Conditional diffusion generates future responses for action-dependent risk estimation. Responsibility-guided preventive optimization calibrates agent-specific risk-reduction requirements to local capacity within a policy trust region, using reserved safety margins, adaptive requirement scaling, and joint feasibility verification. Theoretical analysis establishes risk monotonicity and sufficient local conditions for non-decreasing environmental return under explicit assumptions. Experiments in highway-env across three traffic densities, together with controlled graph-recovery and attribution studies, support the effectiveness of the framework. Under high-density traffic, CARGO increases the success rate by 66.04\% and reduces the crash rate by 56.23\% relative to the best evaluated baseline for each metric. |
| 2026-11-06 | [Safe Circumnavigation on Heterogeneous Orbits: A Range-Only HOCBF-QP Framework for Nonholonomic Multi-Agent Systems](http://arxiv.org/abs/2611.09112v1) | Shubham Singh, Anand Singh et al. | Circumnavigating a stationary target with a team of robots requires each robot to hold its own standoff distance, to stay evenly spaced from its neighbours, and to avoid collisions with the target and with each other at every instant. The task becomes arduous when the robots are assigned different orbital radii, as their sensors or roles may demand. This paper presents a safety-critical control framework for this problem. Each robot commands both its acceleration and its turn-rate, measures only its distance to the target, and exchanges information only with its two ring neighbours. The design has three features. First, a structural property of the model lets one control law serve every robot regardless of its assigned radius. Second, the acceleration drives each radius exactly to its target, while the turn-rate balances the formation with the neighbours. Third, a higher-order control-barrier-function quadratic program enforces target, geofence, and inter-robot safety on both inputs at once; it leaves the nominal controller untouched away from the constraints and stays valid while the unmeasured range rate is still being estimated. Simulations confirm these guarantees. In a comparison, the closest existing multi-orbit design, even with full state information, violates both the target standoff and the inter-robot margin, whereas the proposed controller does not. |
| 2026-11-06 | [When Routine Operations Break Architectural Pattern Relationships in LLM Agent Harnesses](http://arxiv.org/abs/2611.09088v1) | Boming Xia, Qinghua Lu et al. | LLM agent harnesses distribute tool mediation, state management and safety guardrails across native services, independently written plugins and configuration. Routine operations can change the runtime order or retained state that architectural pattern relationships require, without any code change. We study five relationships associated with the Guardrail Enforcement Point, Autonomy Controller and Blackboard patterns across five released harnesses, analysing a main sample of fourteen independently written plugins. Using PatternOps, our assessment procedure, we trace requirements from pattern descriptions, provider documentation and deployment profiles to the providers that realise them and the records that decide them. We then compare alternative allocations under supported operations. Such operations broke relationships silently: after a plugin was reapplied or an allow rule saved, policy stopped evaluating edits while every plugin stayed active, and a prohibited edit executed. Which alternatives qualified depended on the requirement. Allocations that kept the required order, or kept or rebuilt the needed state, held on their tested paths and operation sequences, while others prevented prohibited effects but missed edits rejected elsewhere. In these harnesses, a pattern relationship is a property of the running system and its history, so architects should evaluate it against the decision paths, state lifetimes and operation sequences their deployments use. |
| 2026-11-06 | [Does Reasoning Improve Psychological Depth in Large Language Models?](http://arxiv.org/abs/2611.09061v1) | Ruichen Zheng, Yihe Wang et al. | Benchmark gains from reasoning-enhanced large language models encourage the assumption that these models should broadly perform better on complex generative tasks, but this remains underexplored in open-ended, creative domains shaped by nuances of reader experience. We tested this assumption through the lens of psychological depth in short stories. We compared DeepSeek-R1 vs. DeepSeek-V3, our closest available contrast for reasoning-oriented post-training, and GPT-5 vs. GPT-4o, through 60 blinded pairwise evaluations in which seven human annotators and an ensemble LLM-as-a-Judge rated prompt-matched story outputs. Human evaluations showed no universal reasoning advantage: GPT-5 was slightly preferred over GPT-4o (60.0-62.9% across two reasoning-effort settings), whereas DeepSeek-R1 fell behind DeepSeek-V3 (42.9%). Although this pattern was marked by low inter-annotator agreement (Krippendorff's $α= 0.070$), our annotation-quality checks suggested genuine evaluative subjectivity, plausibly reflecting distinct annotator decision schemes rather than annotation failure. In contrast, the LLM-as-a-Judge showed a strong pro-reasoning bias (98.3% of pairs vs. 55.2% of human votes), consistent across its five configurations and partly tracking sentence length. These findings suggest that, at the frontier, reasoning-related quality differences are small enough to be dominated by reader subjectivity, and that LLM-as-a-Judge evaluators can obscure this subjectivity in nuanced creative-generation tasks. |
| 2026-11-06 | [FASTFUNNELS: Agile Multi Robot Navigation in Narrow Spaces via Adaptive Dynamic Funnels](http://arxiv.org/abs/2611.09005v1) | Vignesh Rajagopal, Rohan Chandra | We focus on the problem of high-speed multi-robot navigation in cluttered spaces with tight spatial constraints. A key challenge is balancing safety with agility, as existing methods often trade one for the other (slowing down near a bottleneck to avoid collisions versus failing to swerve in time at high speeds), especially for car-like systems. Our approach belongs to the family of formation control methods, in which Ackermann-steering agents coordinate while navigating tight spaces. However, current formation control techniques rely on rigid, pre-defined shapes and transitions that depend on initial conditions, the environment, and vehicle dynamics, and are impractical to maintain at high speeds. For example, agents passing through a small gap must first form a square, transform into a line using analytical equations, and then return to a square. In contrast, our approach is inspired by a droplet of water that continuously adapts its shape to the space it flows through without slowing down. We propose a hierarchical algorithm in which, at the high level, a reinforcement learning policy produces a dynamic, blob-like elliptical patch that morphs its shape in real time, squeezing, stretching, and orienting itself around obstacles and through tight spaces. At the low level, the robots are controlled via distributed model predictive control (DMPC) to stay inside the patch, which acts as a soft forcing function, while avoiding collisions with each other. We evaluate our approach, FastFunnels, on simulated F1TENTH racing cars across 14 scenarios with N=1 to N=6 robots, averaged over 20 trials each. FastFunnels achieves an 89% completion rate at 7.73 m/s, while all baselines either fail to complete the track or succeed only at very low speeds (0.07-1.79 m/s, about 40% completion rate). |
| 2026-11-06 | [HomeLoop: A Simulation Framework and Benchmark for Robot-IoT Coordination in Everyday Domestic Assistance and Safety](http://arxiv.org/abs/2611.08914v1) | Ramy ElMallah, Marc Bishara et al. | Recent progress in embodied robotics is bringing robots closer to deployment in homes, as vision-language models (VLMs) are demonstrating significant advances in reasoning, making VLM-driven robots promising home assistant agents. These agents will often be introduced into homes that already have cameras, Internet of Things (IoT) devices, and residents with wearables. To fulfill their potential, they need to integrate with existing smart infrastructure and reason through ambiguous scenarios: a smoke alarm may come from a pan about to ignite, a pan already on fire, a resident cooking or a faulty sensor, and each calls for a different response. To that end, we introduce HomeLoop, a simulation framework and benchmark built on Isaac Sim that combines robot bodies, networked devices and wearables, simulated residents and visitors, faulty sensors, and hazards that evolve over time. Our benchmark is composed of 20 tasks across five families grounded in real-world household hazards and needs, and designed so that the same alert can have different causes. A runtime monitor scores each episode with temporal-logic checks on the final outcome and the actions along the way, and every action carries a predefined cost. We evaluate six open-weight and proprietary VLMs. The best achieves a family-balanced score of 61.9%, and across models we see a pattern of over-reaction: failed episodes involving prohibited interventions outnumber those involving missed required interventions by more than two to one, with models often shutting off the stove before investigating the alarm's cause. HomeLoop is an extensible foundation intended for use in research on robot-IoT coordination, long-term memory, proactive assistance, and privacy-safety trade-offs. HomeLoop will be released with 2,100 home layouts that can be automatically furnished in diverse configurations, five robot bodies and 20 networked device types. |
| 2026-11-06 | [Localizing and Editing the Goal Representation of a Vision-Language-Action Model](http://arxiv.org/abs/2611.08895v1) | Sudhanshu Srivastava, Faisal Aqlan | Operators can state safety constraints for robots as prohibitions such as "do not put the bowl on the stove". We study whether OpenVLA follows such prohibitions and where it represents the goal it executes. The 10 tasks of LIBEROGoal share one scene, so we can label the goal that each rollout executes. We combine closed-loop rollouts with activation patching, linear probes, activation steering and runtime monitors. On LIBERO-Goal, negation had no detectable effect on which task OpenVLA executed. The model executed the forbidden task in 79% of negated rollouts and completed 78% of plain tasks. A goal probe gives similar read-outs under "not X" and "X". Activation patching shows that the first action depends mostly on the instruction tokens before layer 12 and on the last prompt token from layer 16 onward. A fixed steering vector at the instruction tokens at layer 8 redirected 12 of 18 rollouts. Fixed vectors and prefill patches at layers 12 to 20 redirected none of 158. From layer 16, only patches at every decoded action token redirected the robot (15 of 18 rollouts at layer 24). When the instruction changes during a rollout, the layer-32 goal read-out detects non-compliance better than failure detectors, and an external probe on image and instruction features performs comparably. These results support enforcing prohibitions outside the instruction and monitoring goal read-outs against the specification. Goal edits in OpenVLA should act before layer 12 or patch every action token. |

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



