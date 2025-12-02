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
| 2025-12-01 | [ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation](http://arxiv.org/abs/2512.02013v1) | Chenyang Gu, Jiaming Liu et al. | Vision-Language-Action (VLA) models have recently emerged, demonstrating strong generalization in robotic scene understanding and manipulation. However, when confronted with long-horizon tasks that require defined goal states, such as LEGO assembly or object rearrangement, existing VLA models still face challenges in coordinating high-level planning with precise manipulation. Therefore, we aim to endow a VLA model with the capability to infer the "how" process from the "what" outcomes, transforming goal states into executable procedures. In this paper, we introduce ManualVLA, a unified VLA framework built upon a Mixture-of-Transformers (MoT) architecture, enabling coherent collaboration between multimodal manual generation and action execution. Unlike prior VLA models that directly map sensory inputs to actions, we first equip ManualVLA with a planning expert that generates intermediate manuals consisting of images, position prompts, and textual instructions. Building upon these multimodal manuals, we design a Manual Chain-of-Thought (ManualCoT) reasoning process that feeds them into the action expert, where each manual step provides explicit control conditions, while its latent representation offers implicit guidance for accurate manipulation. To alleviate the burden of data collection, we develop a high-fidelity digital-twin toolkit based on 3D Gaussian Splatting, which automatically generates manual data for planning expert training. ManualVLA demonstrates strong real-world performance, achieving an average success rate 32% higher than the previous hierarchical SOTA baseline on LEGO assembly and object rearrangement tasks. |
| 2025-12-01 | [Bounded treewidth, multiple context-free grammars, and downward closures](http://arxiv.org/abs/2512.01973v1) | C. Aiswarya, Pascal Baumann et al. | The reachability problem in multi-pushdown automata (MPDA) has many applications in static analysis of recursive programs. An example is safety verification of multi-threaded recursive programs with shared memory. Since these problems are undecidable, the literature contains many decidable (and efficient) underapproximations of MPDA.   A uniform framework that captures many of these underapproximations is that of bounded treewidth (tw): To each execution of the MPDA, we associate a graph; then we consider the subset of all graphs that have a wt at most $k$, for some constant $k$. In fact, bounding tw is a generic approach to obtain classes of systems with decidable reachability, even beyond MPDA underapproximations. The resulting systems are also called MSO-definable bounded-tw systems.   While bounded tw is a powerful tool for reachability and similar types of analysis, the word languages (i.e. action sequences corresponding to executions) of these systems remain far from understood.   For the slight restriction of bounded special tw, or "bounded-stw" (which is equivalent to bounded tw on MPDA, and even includes all bounded-tw systems studied in the literature), this work reveals a connection with multiple context-free languages (MCFL), a concept from computational linguistics. We show that the word languages of MSO-definable bounded-stw systems are exactly the MCFL.   We exploit this connection to provide an optimal algorithm for computing downward closures (dcl) for MSO-definable bounded-stw systems. Computing dcl is a notoriously difficult task that has many applications in the verification of complex systems: As an example application, we show that in programs with dynamic spawning of MSO-definable bounded-stw processes, safety verification has the same complexity as in the case of processes with sequential recursive processes. |
| 2025-12-01 | [Rectifying LLM Thought from Lens of Optimization](http://arxiv.org/abs/2512.01925v1) | Junnan Liu, Hongwei Liu et al. | Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (Rectifying Process-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors. |
| 2025-12-01 | [Provably Safe Model Updates](http://arxiv.org/abs/2512.01899v1) | Leo Elmecker-Plakolm, Pierre Fasterling et al. | Safety-critical environments are inherently dynamic. Distribution shifts, emerging vulnerabilities, and evolving requirements demand continuous updates to machine learning models. Yet even benign parameter updates can have unintended consequences, such as catastrophic forgetting in classical models or alignment drift in foundation models. Existing heuristic approaches (e.g., regularization, parameter isolation) can mitigate these effects but cannot certify that updated models continue to satisfy required performance specifications. We address this problem by introducing a framework for provably safe model updates. Our approach first formalizes the problem as computing the largest locally invariant domain (LID): a connected region in parameter space where all points are certified to satisfy a given specification. While exact maximal LID computation is intractable, we show that relaxing the problem to parameterized abstract domains (orthotopes, zonotopes) yields a tractable primal-dual formulation. This enables efficient certification of updates - independent of the data or algorithm used - by projecting them onto the safe domain. Our formulation further allows computation of multiple approximately optimal LIDs, incorporation of regularization-inspired biases, and use of lookahead data buffers. Across continual learning and foundation model fine-tuning benchmarks, our method matches or exceeds heuristic baselines for avoiding forgetting while providing formal safety guarantees. |
| 2025-12-01 | [NeuroHJR: Hamilton-Jacobi Reachability-based Obstacle Avoidance in Complex Environments with Physics-Informed Neural Networks](http://arxiv.org/abs/2512.01897v1) | Granthik Halder, Rudrashis Majumder et al. | Autonomous ground vehicles (AGVs) must navigate safely in cluttered environments while accounting for complex dynamics and environmental uncertainty. Hamilton-Jacobi Reachability (HJR) offers formal safety guarantees through the computation of forward and backward reachable sets, but its application is hindered by poor scalability in environments with numerous obstacles. In this paper, we present a novel framework called NeuroHJR that leverages Physics-Informed Neural Networks (PINNs) to approximate the HJR solution for real-time obstacle avoidance. By embedding system dynamics and safety constraints directly into the neural network loss function, our method bypasses the need for grid-based discretization and enables efficient estimation of reachable sets in continuous state spaces. We demonstrate the effectiveness of our approach through simulation results in densely cluttered scenarios, showing that it achieves safety performance comparable to that of classical HJR solvers while significantly reducing the computational cost. This work provides a new step toward real-time, scalable deployment of reachability-based obstacle avoidance in robotics. |
| 2025-12-01 | [Beyond SFT: Reinforcement Learning for Safer Large Reasoning Models with Better Reasoning Ability](http://arxiv.org/abs/2512.01848v1) | Jinghan Jia, Nathalie Baracaldo et al. | Large reasoning models (LRMs) extend large language models by generating explicit chain-of-thought (CoT) reasoning, significantly improving mathematical and logical problem solving. However, this explicit reasoning process also introduces new safety risks, as unsafe behaviors often emerge within intermediate reasoning trajectories, even when final answers appear harmless. Existing safety alignment approaches primarily rely on supervised fine-tuning (SFT) over safety-oriented long CoT datasets. While intuitive, we find that SFT produces inconsistent safety improvements, degrades reasoning ability, and generalizes poorly across model families. These limitations suggest that purely supervised approaches are insufficient for robust safety alignment in LRMs. To address this, we investigate reinforcement learning (RL) as a complementary optimization framework for LRM safety training. Unlike SFT, RL directly optimizes model policies with reward feedback, enabling more adaptive and stable alignment. Extensive experiments across multiple model families and benchmarks show that RL achieves stronger and more consistent safety gains while maintaining reasoning competence. Further analysis of reflection dynamics and token-level entropy reveals that RL suppresses unsafe exploratory reasoning while preserving reflective depth, leading to safer and more reliable reasoning processes. |
| 2025-12-01 | [OpenREAD: Reinforced Open-Ended Reasoing for End-to-End Autonomous Driving with LLM-as-Critic](http://arxiv.org/abs/2512.01830v1) | Songyan Zhang, Wenhui Huang et al. | Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks. |
| 2025-12-01 | [Search for Peak Structures in the Stochastic Gravitational-Wave Background in LIGO-Virgo-KAGRA O1-O4a Datasets](http://arxiv.org/abs/2512.01776v1) | Catalina-Ana Miritescu, Mario Martinez et al. | We present a dedicated search for gravitational-wave backgrounds with nontrivial peak structures using data from the first three and the initial part of the fourth observing runs of the LIGO-Virgo-KAGRA network. The analysis is motivated by a variety of early-Universe models characterized by signals with multiple peaks. We introduce a model independent parameterization of double-peaked spectra based on the superposition of two normalized broken power laws and perform a Bayesian inference study using the LIGO-Virgo-KAGRA isotropic cross-correlation data. While no statistically significant evidence for a multi-peak background is found, the analysis provides constraints on the inter-peak slopes in correlation with the signal amplitude. These results exhibit LIGO-Virgo-KAGRA's ability to probe signals beyond a single peak structure and establish a foundation for future targeted searches for nontrivial gravitational waves background spectral shapes in future observing runs and the advanced detector era. |
| 2025-12-01 | [Beware of Reasoning Overconfidence: Pitfalls in the Reasoning Process for Multi-solution Tasks](http://arxiv.org/abs/2512.01725v1) | Jiannan Guan, Qiguang Chen et al. | Large Language Models (LLMs) excel in reasoning tasks requiring a single correct answer, but they perform poorly in multi-solution tasks that require generating comprehensive and diverse answers. We attribute this limitation to \textbf{reasoning overconfidence}: a tendency to express undue certainty in an incomplete solution set. To examine the effect, we introduce \textit{MuSoBench}, a benchmark of multi-solution problems. Experiments show that the conventional short chain-of-thought (Short-CoT) prompting paradigm exhibits pronounced overconfidence, whereas the emerging long chain-of-thought (Long-CoT) approach mitigates it through iterative exploration and self-reflection. We further characterise observable behaviours and influential factors. To probe the underlying cause, we propose the \textbf{cognitive-rigidity hypothesis}, which posits that overconfidence arises when the reasoning process prematurely converges on a narrow set of thought paths. An attention-entropy analysis offers preliminary support for this view. These findings provide tools for assessing the completeness of LLM reasoning and highlight the need to move evaluation beyond single-answer accuracy toward comprehensive exploration. |
| 2025-12-01 | [Integrating Artificial Intelligence and Mixed Integer Linear Programming: Explainable Graph-Based Instance Space Analysis in Air Transportation](http://arxiv.org/abs/2512.01698v1) | Artur Guerra Rosa, Felipe Tavares Loureiro et al. | This paper analyzes the integration of artificial intelligence (AI) with mixed integer linear programming (MILP) to address complex optimization challenges in air transportation with explainability. The study aims to validate the use of Graph Neural Networks (GNNs) for extracting structural feature embeddings from MILP instances, using the air05 crew scheduling problem. The MILP instance was transformed into a heterogeneous bipartite graph to model relationships between variables and constraints. Two neural architectures, Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) were trained to generate node embeddings. These representations were evaluated using Instance Space Analysis (ISA) through linear (PCA) and non-linear (UMAP, t-SNE) dimensionality reduction techniques. Analysis revealed that PCA failed to distinguish cluster structures, necessitating non-linear reductions to visualize the embedding topology. The GCN architecture demonstrated superior performance, capturing global topology with well-defined clusters for both variables and constraints. In contrast, the GAT model failed to organize the constraint space. The findings confirm that simpler graph architectures can effectively map the sparse topology of aviation logistics problems without manual feature engineering, contributing to explainability of instance complexity. This structural awareness provides a validated foundation for developing future Learning to Optimize (L2O) agents capable of improving solver performance in safety-critical environments. |
| 2025-12-01 | [Dynamic Log-Gaussian Process Control Barrier Function for Safe Robotic Navigation in Dynamic Environments](http://arxiv.org/abs/2512.01668v1) | Xin Yin, Chenyang Liang et al. | Control Barrier Functions (CBFs) have emerged as efficient tools to address the safe navigation problem for robot applications. However, synthesizing informative and obstacle motion-aware CBFs online using real-time sensor data remains challenging, particularly in unknown and dynamic scenarios. Motived by this challenge, this paper aims to propose a novel Gaussian Process-based formulation of CBF, termed the Dynamic Log Gaussian Process Control Barrier Function (DLGP-CBF), to enable real-time construction of CBF which are both spatially informative and responsive to obstacle motion. Firstly, the DLGP-CBF leverages a logarithmic transformation of GP regression to generate smooth and informative barrier values and gradients, even in sparse-data regions. Secondly, by explicitly modeling the DLGP-CBF as a function of obstacle positions, the derived safety constraint integrates predicted obstacle velocities, allowing the controller to proactively respond to dynamic obstacles' motion. Simulation results demonstrate significant improvements in obstacle avoidance performance, including increased safety margins, smoother trajectories, and enhanced responsiveness compared to baseline methods. |
| 2025-12-01 | [Bayesian Ambiguity Contraction-based Adaptive Robust Markov Decision Processes for Adversarial Surveillance Missions](http://arxiv.org/abs/2512.01660v1) | Jimin Choi, Max Z. Li | Collaborative Combat Aircraft (CCAs) are envisioned to enable autonomous Intelligence, Surveillance, and Reconnaissance (ISR) missions in contested environments, where adversaries may act strategically to deceive or evade detection. These missions pose challenges due to model uncertainty and the need for safe, real-time decision-making. Robust Markov Decision Processes (RMDPs) provide worst-case guarantees but are limited by static ambiguity sets that capture initial uncertainty without adapting to new observations. This paper presents an adaptive RMDP framework tailored to ISR missions with CCAs. We introduce a mission-specific formulation in which aircraft alternate between movement and sensing states. Adversarial tactics are modeled as a finite set of transition kernels, each capturing assumptions about how adversarial sensing or environmental conditions affect rewards. Our approach incrementally refines policies by eliminating inconsistent threat models, allowing agents to shift from conservative to aggressive behaviors while maintaining robustness. We provide theoretical guarantees showing that the adaptive planner converges as credible sets contract to the true threat and maintains safety under uncertainty. Experiments under Gaussian and non-Gaussian threat models across diverse network topologies show higher mission rewards and fewer exposure events compared to nominal and static robust planners. |
| 2025-12-01 | [Velocity-Adaptive Access Scheme for Semantic-Aware Vehicular Networks: Joint Fairness and AoI Optimization](http://arxiv.org/abs/2512.01571v1) | Xiao Xu, Qiong Wu et al. | In this paper, we address the problem of fair access and Age of Information (AoI) optimization in 5G New Radio (NR) Vehicle to Everything (V2X) Mode 2. Specifically, vehicles need to exchange information with the road side unit (RSU). However, due to the varying vehicle speeds leading to different communication durations, the amount of data exchanged between different vehicles and the RSU may vary. This may poses significant safety risks in high-speed environments. To address this, we define a fairness index through tuning the selection window of different vehicles and consider the image semantic communication system to reduce latency. However, adjusting the selection window may affect the communication time, thereby impacting the AoI. Moreover, considering the re-evaluation mechanism in 5G NR, which helps reduce resource collisions, it may lead to an increase in AoI. We analyze the AoI using Stochastic Hybrid System (SHS) and construct a multi-objective optimization problem to achieve fair access and AoI optimization. Sequential Convex Approximation (SCA) is employed to transform the non-convex problem into a convex one, and solve it using convex optimization. We also provide a large language model (LLM) based algorithm. The scheme's effectiveness is validated through numerical simulations. |
| 2025-12-01 | [Deep FlexQP: Accelerated Nonlinear Programming via Deep Unfolding](http://arxiv.org/abs/2512.01565v1) | Alex Oshin, Rahul Vodeb Ghosh et al. | We propose an always-feasible quadratic programming (QP) optimizer, FlexQP, which is based on an exact relaxation of the QP constraints. If the original constraints are feasible, then the optimizer finds the optimal solution to the original QP. On the other hand, if the constraints are infeasible, the optimizer identifies a solution that minimizes the constraint violation in a sparse manner. FlexQP scales favorably with respect to the problem dimension, is robust to both feasible and infeasible QPs with minimal assumptions on the problem data, and can be effectively warm-started. We subsequently apply deep unfolding to improve our optimizer through data-driven techniques, leading to an accelerated Deep FlexQP. By learning dimension-agnostic feedback policies for the parameters from a small number of training examples, Deep FlexQP generalizes to problems with larger dimensions and can optimize for many more iterations than it was initially trained for. Our approach outperforms two recently proposed state-of-the-art accelerated QP approaches on a suite of benchmark systems including portfolio optimization, classification, and regression problems. We provide guarantees on the expected performance of our deep QP optimizer through probably approximately correct (PAC) Bayes generalization bounds. These certificates are used to design an accelerated sequential quadratic programming solver that solves nonlinear optimal control and predictive safety filter problems faster than traditional approaches. Overall, our approach is very robust and greatly outperforms existing non-learning and learning-based optimizers in terms of both runtime and convergence to the optimal solution across multiple classes of NLPs. |
| 2025-12-01 | [LLM2Fx-Tools: Tool Calling For Music Post-Production](http://arxiv.org/abs/2512.01559v1) | Seungheon Doh, Junghyun Koo et al. | This paper introduces LLM2Fx-Tools, a multimodal tool-calling framework that generates executable sequences of audio effects (Fx-chain) for music post-production. LLM2Fx-Tools uses a large language model (LLM) to understand audio inputs, select audio effects types, determine their order, and estimate parameters, guided by chain-of-thought (CoT) planning. We also present LP-Fx, a new instruction-following dataset with structured CoT annotations and tool calls for audio effects modules. Experiments show that LLM2Fx-Tools can infer an Fx-chain and its parameters from pairs of unprocessed and processed audio, enabled by autoregressive sequence modeling, tool calling, and CoT reasoning. We further validate the system in a style transfer setting, where audio effects information is transferred from a reference source and applied to new content. Finally, LLM-as-a-judge evaluation demonstrates that our approach generates appropriate CoT reasoning and responses for music production queries. To our knowledge, this is the first work to apply LLM-based tool calling to audio effects modules, enabling interpretable and controllable music production. |
| 2025-12-01 | [Formal Verification of Noisy Quantum Reinforcement Learning Policies](http://arxiv.org/abs/2512.01502v1) | Dennis Gross | Quantum reinforcement learning (QRL) aims to use quantum effects to create sequential decision-making policies that achieve tasks more effectively than their classical counterparts. However, QRL policies face uncertainty from quantum measurements and hardware noise, such as bit-flip, phase-flip, and depolarizing errors, which can lead to unsafe behavior. Existing work offers no systematic way to verify whether trained QRL policies meet safety requirements under specific noise conditions.   We introduce QVerifier, a formal verification method that applies probabilistic model checking to analyze trained QRL policies with and without modeled quantum noise. QVerifier builds a complete model of the policy-environment interaction, incorporates quantum uncertainty directly into the transition probabilities, and then checks safety properties using the Storm model checker.   Experiments across multiple QRL environments show that QVerifier precisely measures how different noise models influence safety, revealing both performance degradation and cases where noise can help. By enabling rigorous safety verification before deployment, QVerifier addresses a critical need: because access to quantum hardware is expensive, pre-deployment verification is essential for any safety-critical use of QRL. QVerifier targets a potential classical-quantum sweet spot: trained QRL policies that execute efficiently on quantum hardware, yet remain tractable for classical probabilistic model checking despite being too slow for real-time classical deployment. |
| 2025-12-01 | [Multi-Path Collaborative Reasoning via Reinforcement Learning](http://arxiv.org/abs/2512.01485v1) | Jindi Lv, Yuhao Zhou et al. | Chain-of-Thought (CoT) reasoning has significantly advanced the problem-solving capabilities of Large Language Models (LLMs), yet conventional CoT often exhibits internal determinism during decoding, limiting exploration of plausible alternatives. Recent methods attempt to address this by generating soft abstract tokens to enable reasoning in a continuous semantic space. However, we find that such approaches remain constrained by the greedy nature of autoregressive decoding, which fundamentally isolates the model from alternative reasoning possibilities. In this work, we propose Multi-Path Perception Policy Optimization (M3PO), a novel reinforcement learning framework that explicitly injects collective insights into the reasoning process. M3PO leverages parallel policy rollouts as naturally diverse reasoning sources and integrates cross-path interactions into policy updates through a lightweight collaborative mechanism. This design allows each trajectory to refine its reasoning with peer feedback, thereby cultivating more reliable multi-step reasoning patterns. Empirical results show that M3PO achieves state-of-the-art performance on both knowledge- and reasoning-intensive benchmarks. Models trained with M3PO maintain interpretability and inference efficiency, underscoring the promise of multi-path collaborative learning for robust reasoning. |
| 2025-12-01 | [Reinventing Clinical Dialogue: Agentic Paradigms for LLM Enabled Healthcare Communication](http://arxiv.org/abs/2512.01453v1) | Xiaoquan Zhi, Hongke Zhao et al. | Clinical dialogue represents a complex duality requiring both the empathetic fluency of natural conversation and the rigorous precision of evidence-based medicine. While Large Language Models possess unprecedented linguistic capabilities, their architectural reliance on reactive and stateless processing often favors probabilistic plausibility over factual veracity. This structural limitation has catalyzed a paradigm shift in medical AI from generative text prediction to agentic autonomy, where the model functions as a central reasoning engine capable of deliberate planning and persistent memory. Moving beyond existing reviews that primarily catalog downstream applications, this survey provides a first-principles analysis of the cognitive architecture underpinning this shift. We introduce a novel taxonomy structured along the orthogonal axes of knowledge source and agency objective to delineate the provenance of clinical knowledge against the system's operational scope. This framework facilitates a systematic analysis of the intrinsic trade-offs between creativity and reliability by categorizing methods into four archetypes: \textit{Latent Space Clinicians}, \textit{Emergent Planners}, \textit{Grounded Synthesizers}, and \textit{Verifiable Workflow Automators}. For each paradigm, we deconstruct the technical realization across the entire cognitive pipeline, encompassing strategic planning, memory management, action execution, collaboration, and evolution to reveal how distinct architectural choices balance the tension between autonomy and safety. |
| 2025-12-01 | [Personalized optimization of pediatric HD-tDCS for dose consistency and target engagement](http://arxiv.org/abs/2512.01406v1) | Zeming Liu, Mo Wang et al. | High-definition transcranial direct current stimulation (HD-tDCS) dosing in children remains largely empirical, relying on one-size-fits-all protocols despite rapid developmental changes in head anatomy and tissue properties that strongly modulate how currents reach the developing brain. Using 70 pediatric head models and commonly used cortical targets, our forward simulations find that standard montages produce marked age-dependent reductions in target electric-field intensity and systematic sex differences linked to tissue-volume covariation, underscoring the profound limitations of conventional uniform montages. To overcome these limitations, we introduce a developmentally informed, dual-objective optimization framework designed to generate personalized Pareto fronts summarizing the trade-off between electric-field intensity and focality. From these optimized solutions, we derive two practical dosing prescriptions: a dose-consistency strategy that, for the first time, enforces fixed target intensity across individuals to implicitly mitigate demographic effects, and a target-engagement strategy that maximizes target intensity under safety limits. Both strategies remain robust to large conductivity variations, and we further show that dense HD-tDCS solutions admit sparse equivalents without performance loss under the target-engagement strategy. We also find that tissue conductivity sensitivity is depth-dependent, with Pareto-front distributions for superficial cortical targets most influenced by gray matter, scalp, and bone conductivities, and those for a deep target predominantly shaped by gray and white matter conductivities. Together, these results establish a principled framework for pediatric HD-tDCS planning that explicitly accounts for developmental anatomy and physiological uncertainty, enabling reliable and individualized neuromodulation dosing in pediatric populations. |
| 2025-12-01 | [Accelerating Probabilistic Response-Time Analysis: Revised Critical Instant and Optimized Convolution](http://arxiv.org/abs/2512.01381v1) | Hiroto Takahashi, Atsushi Yano et al. | Accurate estimation of the Worst-Case Deadline Failure Probability (WCDFP) has attracted growing attention as a means to provide safety assurances in complex systems such as robotic platforms and autonomous vehicles. WCDFP quantifies the likelihood of deadline misses under the most pessimistic operating conditions, and safe estimation is essential for dependable real-time applications. However, achieving high accuracy in WCDFP estimation often incurs significant computational cost. Recent studies have revealed that the classical assumption of the critical instant, the activation pattern traditionally considered to trigger the worst-case behavior, can lead to underestimation of WCDFP in probabilistic settings. This observation motivates the use of a revised critical instant formulation that more faithfully captures the true worst-case scenario. This paper investigates convolution-based methods for WCDFP estimation under this revised setting and proposes an optimization technique that accelerates convolution by improving the merge order. Extensive experiments with diverse execution-time distributions demonstrate that the proposed optimized Aggregate Convolution reduces computation time by up to an order of magnitude compared to Sequential Convolution, while retaining accurate and safe-sided WCDFP estimates. These results highlight the potential of the approach to provide both efficiency and reliability in probabilistic timing analysis for safety-critical real-time applications. |

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



