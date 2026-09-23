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
| 2026-09-22 | [Optimal Sequential Annotations for Off-Policy Evaluation](http://arxiv.org/abs/2609.26707v1) | Woojin Chae, Ezinne Nwankwo et al. | Offline reinforcement learning and off-policy evaluation evaluates dynamic treatment rules based on retrospectively collected data prior to deployment. In recent AI applications, state and reward information is recorded as complex text or image, which recent AI advancements such as LLM-as-a-judge can label with unknown bias. Expert annotation may be available but at a higher cost. For example, safety classification via cheap but imperfect classifiers vs. expensive expert review. We show how a limited budget for ground-truth data-annotation can be used via doubly-robust OPE with missing rewards, and we optimize variance-optimal annotation probabilities for sequential off-policy evaluation, where the target policy value is estimated from annotated data. We characterize the optimal annotation probabilities for sequential forward-monotone annotation protocols, and provide a feasible batch-adaptive implementation. Our work is motivated by a collaboration with a homelessness services nonprofit that writes casenotes for individuals over time. Our method can be used to unlock trustworthy inference from casenote data and answer new inferential questions such as: how does expanding outreach effort over time affect progress towards a housing application and improvement in housing placement? In simulations and on two real datasets - casenotes from the nonprofit and human-preference votes from LMArena - we see reductions in RMSE of 34-65% for housing placement and 17-68% for progress towards a housing application at budgets of 40% of full annotation and above, and by 55-62% at every budget on LMArena. |
| 2026-09-22 | [From Alignment to Access Control: A Framework for GenAI Policy Enforcement](http://arxiv.org/abs/2609.26682v1) | Nathalie Baracaldo | Generative AI (GenAI) applications have flourished enabling users to chat with large language models, and to create agents to act on their behalf for a variety of tasks. The pace of development of capabilities in this field is incredibly fast with security and safety taking a back seat. Unfortunately, the slower pace at which security and safety mechanisms have evolved has led to real incidents. Policy enables the definition of desirable behavior of applications, and for that reason, it is a cornerstone of making systems secure and compliant. Policy however means different things to different practitioners creating confusion and siloed solutions that are not adequate for compliance. This paper takes a tour of the good, the bad and the ugly when it comes to policy enforcement in GenAI applications. We propose a methodology to systematically analyze and dissect existing approaches to define and enforce policy found in the wild. Based on this principled analysis, we provide recommendations and call for action for the community to address.   This paper is a companion extension of USENIX Security 2026 Enigma talk titled "From Alignment to Access Control: A Unified View of GenAI Policy Enforcement" by the author Nathalie Baracaldo. |
| 2026-09-22 | [Autonomous Quantum Transport Measurements of 2D Semiconductors by an AI Agent](http://arxiv.org/abs/2609.26661v1) | Brandon Bauer, Matthew Whalen et al. | Artificial-intelligence (AI) agents are beginning to enter experimental laboratories, automating experiments and accelerating scientific discovery. Herein, we introduce an AI-driven workflow in which an AI agent performs multi-step, multi-day quantum transport measurements end-to-end. Specifically, given brief instructions, the agent starts by planning the multi-step measurements, then safely operates the cryogenic instruments, analyzes the data, and concludes with a final report. We demonstrate this AI workflow on multiple monolayer and bilayer MoS2 devices. Through autonomous measurement campaigns lasting up to six days, the agent determined the conduction-band spin-orbit coupling energy in monolayer MoS2, and mapped a layer- and valley-resolved phase diagram in bilayer MoS2. This experimental workflow is implemented through the FermiLink agent harness, which emphasizes instrumental safety and the reliability of the measurement and analysis. The framework is general and can be readily adapted to other types of experiments, representing a step toward self-driving laboratories. |
| 2026-09-22 | [Dynamic Slack-Aware Clocking for Near-Threshold Tensor Processing Units (TPUs)](http://arxiv.org/abs/2609.26644v1) | Muhammad Usman Nadeem, Sanghamitra Roy et al. | Operating Tensor Processing Units (TPUs) in the near-threshold computing (NTC) region significantly reduces energy consumption but introduces high delay sensitivity to process variation and data activity. Conventional designs typically rely on a conservative, fixed global clock to ensure safety, which leaves large portions of timing margin unexploited as most operations finish well before the clock edge. We propose Dynamic Slack-Aware Clocking (DSAC), a proactive framework that replaces worst-case timing with operation-specific adjustments. DSAC employs lightweight Hamming-Distance, Most-Significant-Bit, and Hybrid predictors to estimate the delay sensitivity of individual multiply-accumulate (MAC) operations and classify them into three timing tiers. These tiers are enforced locally via dummy-hold cycles under a fixed global reference clock, enabling fine-grained timing adaptation without global clock retuning or frequency scaling. A closed-loop feedback controller monitors timing violations and updates tier thresholds at runtime to maintain resilience. Experiments on quantized DNN benchmarks demonstrate that the MSB predictor maintains high inference accuracy, with an average loss of only 1% even at aggressive performance points. Furthermore, DSAC achieves up to 1.55X better energy efficiency at 2.15X frequency scaling compared to a baseline TPU, while incurring an area overhead as low as 13%. |
| 2026-09-22 | [NavSafe-$\infty$: Benchmarking Closed-Loop Driving Safety in Photorealistic Environments](http://arxiv.org/abs/2609.26618v1) | Yuxin Bao, Hongwei Ruan et al. | End-to-end (E2E) driving policies have progressed rapidly on open-loop (OL) benchmarks, yet OL evaluation cannot reveal whether a policy withstands compounding errors, recovers from failures, or interacts safely with surrounding actors. We introduce NavSafe-$\infty$, a photorealistic closed-loop (CL) benchmark of 280 scenarios spanning 28 event types, each with success and failure criteria defined within a structured traffic-safety taxonomy, which yields category-level capability scores for Traffic Crashes, Vulnerable Road User Crashes, Traffic Violations, and Traffic Incidents. Evaluating 20 E2E policies, we find that OL gains do not reliably transfer to CL safety. Analyzing two common remedies further shows that passive demonstration perturbation helps mainly when CL rollouts stay near its perturbed training states, and that OL reinforcement-learning fine-tuning exhibits reward hacking by trading safety margin for ego progress, which CL feedback amplifies into compounding safety-critical errors. Together, these results demonstrate the blind spot of OL benchmarks indicating CL safety success. The benchmark and an extensible toolbox for customizable event curation and policy diagnosis will be open-sourced and maintained to facilitate future research. |
| 2026-09-22 | [Generalizing Manipulation Skills with a Local Coding Agent](http://arxiv.org/abs/2609.26499v1) | Raman Talwar, Elias Nijs et al. | Today, progress in open-weight language models enables systems capable of writing, executing and debugging code while still running on a single workstation. Most language-driven robots give the model a fixed action interface or a trained policy. Generalizing to a new task therefore means more engineering effort or more data collection, both time-consuming. We investigate whether a local open-weight vision-language model can control a robot and one-shot generalize to new variations of a task without new human programming or training. We let a local open-weight VLM, Qwen3.8-27B, drive a UR3e robotic arm from a coding-agent harness. It writes and runs its own code above a service that implements kinematics, safety limits and classic computer vision techniques. We investigate if this system is capable of generalizing to unseen tasks. Specifically, we test it on nine tasks built from children's toys designed to probe generalization capability across various object characteristics: color, size, shape, and task variation of those. With five trials for each task, we observe generalization in 30 out of 45 trials with durations ranging from 3.4 to 67.5 minutes depending on task complexity. We further test if there is a speedup when an agent is asked to redo the task after successful completion. This resulted in a 50% reduction in duration, indicating that there is self-improvement over time. Finally, we expose the limitations of a local coding agent. We believe that solving those limitations combined with further investigation of self-improvement over time points at a direct path toward real-world deployment of a local coding agent. |
| 2026-09-22 | [Benchmarking Robots for Everyday Environments: From Lab Experiments to Real-World Operations](http://arxiv.org/abs/2609.26490v1) | Raphael Memmesheimer, Martina Overbeck et al. | This study introduces an interdisciplinary framework for benchmarking robots deployed in public environments, addressing the gap between traditional laboratory metrics and real-world benchmarking requirements. We evaluate three distinct robots across diverse use cases - outdoor park cleaning, pedestrian underpass cleaning, and interactive library assistance - each representing unique challenges in public daily life. Over a three-year benchmarking process (2023-2025) comprising seven benchmarking events, a consensus workshop and six on-site evaluations (two per use case), we utilized realistic indoor and outdoor test environments to assess not only technical performance but also the broader implications of deploying robots in unstructured, human-centric settings. An expert panel, spanning robotics, human-robot interaction, safety, and economics, systematically developed and refined an evaluation concept to analyze the transition from laboratory prototypes to operational systems. Our findings highlight critical factors for successful deployment, including task fulfillment, interaction quality, safety, and economic feasibility. This work provides actionable insights for researchers and practitioners aiming to bridge the gap between robotic innovation and real-world applicability. |
| 2026-09-22 | [Frugal Collective Perception: Context-Aware Adaptive Reporting for Safety-Critical C-ITS](http://arxiv.org/abs/2609.26470v1) | Romain Tessier, Bruno Monsuez et al. | Ensuring safety and scalability in Collective Perception Service (CPS) remains a key challenge for Cooperative Intelligent Transport Systems (C-ITS). Conventional CPS enhances perception by broadcasting Collective Perception Messages (CPMs). However, its reliance on transmitting a potentially large volume of context-irrelevant information at high frequency leads to network congestion, processing delays, and poor scalability. We propose a Context-Aware Adaptive Filter that dynamically adjusts CPM content and transmission frequency based on contextual relevance and situational criticality. By prioritizing safety-critical objects and interactions, the proposed approach prevents information overload while preserving timely updates for decision-making. An end-to-end SUMO--Artery simulation evaluates safety, decision-making efficiency, and communication cost under different computational capacity tiers and transmission rates. Results show that the proposed adaptive filtering mechanism achieves safety performance comparable to conventional CPMs transmitted at the maximum allowed frequency (10~Hz), while reducing communication volume by over 93\% and preventing queue saturation. This demonstrates that context-aware adaptivity enables CPS to remain both scalable and safety-compliant across heterogeneous computing platforms. |
| 2026-09-22 | [Combining Hierarchical Cognitive Process with Process Supervision for Interpretable Scene Safety Understanding](http://arxiv.org/abs/2609.26399v1) | Zhiyun Jiang, Hanyong Wang et al. | Scene safety understanding plays a life-or-death role in situational awareness in various critical domains. Traditional methods that rely on learning direct mappings between scenes and safety levels often lack interpretability, limiting their reliability in critical applications. An effective approach to overcoming this challenge lies in interpreting human cognitive processes and equipping machine models with analogous cognitive capabilities. This work explores an effective way of integrating scene safety cognitive process modeling and process supervision. Specifically, we first construct a hierarchical cognitive safety structure, which motivates the development of a novel, high-quality scene safety understanding dataset based on multi-step reasoning with process labels. This dataset serves both as a benchmark and a resource to improve the safety reasoning capabilities of Large Language Models (LLMs), while also enabling a granular analysis of intermediate reasoning steps through information flow and saliency-based techniques. Building upon this foundation, we introduce a modular and flexible process supervision framework that reflects the hierarchical nature of human cognition. This framework leverages LLMs as the core architecture and incorporates Low-Rank Adaptation(LoRA) and Mixture-of-Experts (MoE) strategies to enable specialization and collaboration among expert modules, each tasked with specific sub-processes of the overall reasoning chain. Systematic experimental evaluations and analyses confirm that our framework exhibits superior interpretability and performance characteristics compared to traditional approaches. |
| 2026-09-22 | [Learning to Defer with Guidance on Real World Medical Data](http://arxiv.org/abs/2609.26384v1) | Emma Sun, Joshua Strong et al. | Medical image interpretation is high-volume and time-consuming, and while AI interpretation can reduce workload, fully autonomous deployment carries potential safety concerns and low specificity may in practice lead to increased clinician workload. Learning to Defer (L2D) addresses this by selectively routing cases between autonomous prediction and human experts by learning from input features and AI model and human performance. While theoretical guarantees have been proven for L2D, its performance has not been validated on real-world medical datasets with human reader annotations. We evaluate the predictor-rejector formulation of two-stage L2D, where the AI predictor model is fixed and separate from the trainable routing or rejector model, on Collab-CXR, a multilabel chest X-ray dataset with multiple human annotations per case. This is the first work to look at L2D in the context of real-world medical imaging data with human annotations. We further introduce a new setup, L2D with Guidance, where the decision space is extended to three choices: predict autonomously, defer to a human expert, or defer to a human expert and provide AI guidance. We compare multiple rejector architectures and loss functions, and different input feature availabilities. This is reproduced on two larger datasets, VinDr-CXR and CheXpert. Our results show that two-stage L2D with Guidance outperforms classic two-stage learning to defer, as well as human-alone, AI-alone and AI-guided human baselines. Notably, this performance is achieved with simpler loss functions compared to formally defined L2D surrogate loss functions in current literature. |
| 2026-09-22 | [Collective Tube Model Predictive Control With Distribution-Free Joint Safety Certificates](http://arxiv.org/abs/2609.26339v1) | Giuseppe C. Calafiore | Data-calibrated stochastic MPC typically builds separate risk margins for many events along the horizon, such as times, facets, state/input components, or obstacles, and then combines them with a union bound. This approach is valid, but it does not match the key object used in the tube-MPC recursive-feasibility proof, which shifts a complete error tube. This paper develops collective tube MPC (CT-MPC), where the calibrated uncertainty object is the finite-horizon prediction-error trajectory. A reusable trajectory tube is calibrated offline, its cross-sections define deterministic Pontryagin tightenings online, and the certified violation event is that a fresh prediction-error trajectory leaves the tube. For linear systems with additive uncertainty and fixed ancillary feedback, we prove joint state-input safety over the prediction horizon, one-step recursive feasibility from an explicit shifted candidate, a finite-deployment risk bound, and a practical value-decrease inequality. The finite-sample certificate is distribution-free under split calibration and has beta-binomial form; its complexity is the certified number of residual trajectories that can define the tube, rather than the number of horizon-constraint blocks. We also give implementable shift-compatibility tests for polytopic tubes and a stable-compression fallback for irregular tube designers. Numerical experiments compare CT-MPC with Bonferroni tightening, a sample-envelope tube, and a joint-in-time conformal MPC baseline. The collective tube reduces deterministic tightening while preserving empirical safety and recursive feasibility. |
| 2026-09-22 | [SafeLoop: Risk-Aware Rollback for Vision-Language-Action Manipulation](http://arxiv.org/abs/2609.26313v1) | Zeyu Lou, Tianran Zhang et al. | Recent vision-language-action (VLA) models are promising for general-purpose manipulation, but long-horizon execution remains fragile. Small state-estimation or control errors can lead to irreversible failures (e.g., collisions and object drops). Avoiding these risks requires a proactive safety mechanism capable of anticipating hazards. In this paper, we introduce SafeLoop, a non-invasive external wrapper that adds hazard prediction and rollback-based recovery to a VLA model without changing its parameters. SafeLoop trains a risk predictor from vision and proprioception to output four values: the probability and time-to-hazard for body collisions and for object failures. A lightweight controller then chooses one of three actions based on the predicted risk: continue execution (noop), save a safety checkpoint (record), or retreat in joint space (rollback). Rollback moves the robot back to a recent safe waypoint and queries the base policy again, which may yield an alternative continuation. Across 24 LIBERO tasks (16 random seeds each) and three real-robot tasks (25 rollouts each), SafeLoop achieves a stronger overall safety-success trade-off than alternative methods, reducing hazard cases by roughly 70% while preserving task success and the base-policy control rate. Project code is available at https://github.com/Loule0-0/SafeLoop/tree/release/safeloop. |
| 2026-09-22 | [The Minkowski Wrap: A Relativistic Speed Limiter](http://arxiv.org/abs/2609.26311v1) | Nicoletta Prencipe, Başak Sakçak et al. | In special relativity, a particle can experience constant acceleration, but at the same time, its motion is constrained by the velocity limit imposed by the speed of light $c$. Inspired by this principle, we propose a method for enforcing velocity bounds in control systems by replacing $c$ with the maximum attainable speed of the system. We refer to this as the ``Minkowski wrap," the operation of deforming the phase portrait of a system so as to enforce desired speed limits. We apply this idea to shape the input generated by a state-feedback stabilizing controller and time-optimal controller considering controlling a double integrator system. By applying Pontryagin's Maximum Principle, we show that the time-optimal control of a wrapped double-integrator system is bang-bang. The proposed method transforms the classical double-integrator dynamics into a ``wrapped" system that respects velocity bounds without the need for clipping, offering an explicit nonlinear feedback control strategy conducive to safety applications. |
| 2026-09-22 | [On the security and privacy of LLMs in Mobility](http://arxiv.org/abs/2609.26295v1) | Mauro Conti, Lorenzo Perinello et al. | The mobility sector is undergoing a paradigm shift driven by advances in Generative Artificial Intelligence. With a global market valued at approximately 2.9 trillion dollars annually, considering only cars, the integration of these technologies has the potential to impact more than 1.5 billion vehicles worldwide. As Large Language Models (LLMs) are increasingly adopted in mobility, concerns about cybersecurity, privacy, and reliability emerge. Accordingly, this paper surveys current applications and assesses these challenges. Since the European AI Act classifies transportation AI as high risk, we derive nine technical classes from its requirements to assess current research and future deployments. Our findings show that research mainly studies GPT and Llama models (over 50\% of reviewed works) and traffic applications while largely neglecting security, privacy, and reliability. This gap extends to AI Act compliance: among 35 reviewed works, only one includes a partial vulnerability assessment and one a partial risk management system. We identify a clear gap between strong optimization performance and regulatory adherence, suggesting compliance is limited less by technology than by a focus on static performance over lifecycle safety, and underscoring an urgent need for security-by-design in safety-critical intelligent transportation systems. |
| 2026-09-22 | [VideoX-Qwen: Data-Centric Instruction-Based Video Editing](http://arxiv.org/abs/2609.26015v1) | JJiahang Li, Dingbao Shao et al. | Progress in general-purpose video editing depends on constructing large-scale paired supervision and effectively adapting video-generation backbones to instruction-driven editing. Unlike video generation, video editing must execute a requested transformation while preserving unrelated subjects, scene structure, motion, and temporal continuity. We present VideoX-Qwen, an integrated data-construction and model-training framework for general instruction-based video editing. Our scalable production pipeline organizes specialized generation and understanding models into complementary routes for addition, removal, replacement, and attribute editing, followed by quality screening and instruction enrichment. It produces more than 1.2 million directional video-editing records, including over 400,000 records in each major task group, with an automatic acceptance rate of 89%. The resulting corpus provides broad and structured coverage of common editing operations through a unified source-instruction-target interface. We further develop a unified Qwen-Wan editor that combines multimodal semantic conditioning with dense source-video latent guidance. A progressive image-video training strategy aligns the multimodal instruction interface, adapts the video generator to source-conditioned editing, and refines output quality with selected high-resolution data. In a 100-example comparison with UniVideo and Kling O1, VideoX-Qwen achieves the best mean result on nine of eleven reported metrics, including instruction following, editing quality, content preservation, structural and perceptual similarity, and video-distribution quality. Together, the large-scale data-production system and unified training framework provide a practical foundation for more capable instruction-driven video editing. |
| 2026-09-22 | [Overload-Robust Latency in 5G-TSN: A HoL-Enhanced Hybrid Lyapunov Approach for 3GPP Indoor Factory Environments](http://arxiv.org/abs/2609.26011v1) | Kouros Zanbouri, Md Noor-A-Rahim et al. | Private 5G networks are a key enabler for flexible industrial automation, especially when used in conjunction with Time-Sensitive Networking (TSN) technology. In this context, radio schedulers must multiplex safety-critical control traffic with bandwidth-hungry sensing streams over a fixed spectrum allocation. This paper proposes a Head-of-Line (HoL) Enhanced Hybrid Lyapunov scheduler for 5G-TSN networks that augments a drift-plus-penalty queue-stability core with an explicit head-of-line delay term and a class-isolation mechanism. The scheduler is evaluated in a 3GPP Indoor Factory scenario with standardized 3GPP fading, spatial consistency, and clutter blockage, using Automated Guided Vehicles (AGVs) generating concurrent URLLC, eMBB, and mMTC flows mapped to dedicated QoS-flow bearers. A fleet-size sweep of 5--30 AGVs on a fixed 20\,MHz carrier reveals a scheduler-independent capacity threshold at approximately 12 vehicles, verified by resource-block saturation. Below the threshold, the proposed scheduler is competitive with the strongest delay-aware baselines and its head-of-line term halves the URLLC deadline-miss ratio relative to the plain Lyapunov formulation. Beyond the threshold, it degrades selectively where the baselines collapse: at $2.5\times$ overload it delivers $1.8\times$ more URLLC traffic than the proportional-fair and delay-budget-aware baselines with a $\approx 4$--$7\times$ shorter 99th-percentile latency, resolving the capacity shortfall in favour of the critical classes instead of spreading it across the traffic mix, at a quantified cost in aggregate cell throughput. The results position Lyapunov-based scheduling as an attractive overload-robustness mechanism for industrial 5G deployments that must remain dependable under unexpected load conditions. |
| 2026-09-22 | [Safety-Constrained Model Predictive Control for an Omnidirectional Walking Assistive Robot Using Control Barrier Function](http://arxiv.org/abs/2609.25994v1) | Andrea Fortuna, Marta Lorenzini et al. | Providing safe and effective mobility assistance plays a crucial role in restoring independence and enhancing the quality of life for individuals with motor impairments. In this context, robotic walking assistive devices have recently emerged as promising solutions to provide physically compliant interaction while ensuring user safety and support. This paper presents a novel control framework for an omnidirectional Walking Assistive Robot (I-WANDER) that integrates a Control Barrier Function (CBF) formulation into a Model Predictive Control (MPC) scheme to explicitly enforce collision-avoidance safety constraints while optimizing for energy efficiency and smooth human-robot collaboration. The method was experimentally evaluated with 12 healthy participants performing two different walking tasks using both the proposed CBF-based MPC controller (CB-MPC) and a variable admittance controller (AC). The first task involved structured navigation through a U-shaped corridor, whereas the second consisted of a single-obstacle avoidance task performed blindfolded to ensure the obstacle was unexpected. Comparative results show that the CB-MPC architecture significantly reduces energy consumption and mechanical work (p < 0.01) without compromising motion smoothness, while also decreasing the number of obstacle collisions. Overall, the findings highlight the potential of the proposed control architecture to enhance both safety and efficiency in robotic walking assistance. |
| 2026-09-22 | [Governed AI-Agent Coordination for Dementia Care: Architecture, Safety Contracts, and Evidence-Derived Workflow Verification](http://arxiv.org/abs/2609.25956v1) | Francesca Medda, Hui Gong | Dementia care increasingly involves connected sensors, medication devices, electronic records, and assistive technologies. Interoperability can transport observations but cannot maintain an accountable care state, reconcile evidence, determine who may act, or verify resolution. The shift from large language models to agentic engineering creates a systems opportunity: an external runtime can maintain memory across episodes, plan over goals and constraints, invoke tools, observe outcomes, and enforce governance. This paper presents Governed Closed-loop Agent Coordination (GCAC), an architecture for bounded agent participation in community dementia-care workflows. Evidence on care-coordination failures and policy obligations is translated into traceable system requirements. GCAC separates observation, governed memory, planning, deterministic policy enforcement, execution, and outcome monitoring through a typed event-memory-decision-action-outcome contract. A reference harness evaluates 18 evidence-derived traces covering missing records, medication conflict, caregiver reports, service failure, consent change, stale state, duplicate events, untrusted text, and suspected acute neurological change. GCAC satisfies all 18 contract oracles with zero policy-violating tool calls and correctly preserves obligations, rejects stale state, creates human hand-offs, and records workflow closure. Event-threshold and stateless-planner controls satisfy 2/18 and 1/18 oracles, respectively. Component ablations localise failures to the removed memory, policy, or versioning function. The results establish architectural conformance rather than clinical effectiveness and show how agentic systems can automate reconciliation, routing, documentation, and follow-up while preserving human authority over consequential care decisions. |
| 2026-09-22 | [Towards Systematic Qualification of Vision-Language Models for Automotive Perception Systems](http://arxiv.org/abs/2609.25945v1) | Malsha Ashani Mahawatta Dona, Konstantinos Rokanas et al. | The field of Artificial Intelligence has been adopted for many application domains. Vision Language Models are one of the recently advanced AI techniques that have been explored to support automotive features such as vehicle perception, and safety assurance. However, such language models are prone to hallucinations, posing a potential threat to the safety of automotive systems that may incorporate them. Within the automotive domain, VLMs could not only hallucinate traffic objects, but could also fail to identify traffic objects that are actually present, which may potentially lead to dangerous situations. Though we have observed a growing body of literature that proposes verification and validation techniques for safe and trustworthy AI, these methods are often studied in isolation, focusing either on run-time or design-time phases. Such isolated techniques could be insufficient in safety-critical, realistic contexts such as automotive perception systems. In this paper, we analyze design-time and run-time verification and validation techniques based on a taxonomy presented by Huang et al. We present an automotive study in which a design-time qualification workflow is proposed to complement run-time monitoring. This workflow combines a fixed safety-relevant ontology-based structured annotation system together with a synonym-based evaluation process to statistically evaluate three state-of-the-art VLMs against data from the nuScenes dataset. We observed that the proposed technique enables deterministic and repeatable quantification of the hallucinations VLMs generate in automotive perception-related tasks. The proposed workflow supports model comparison and deployment-oriented engineering decisions within the design-time verification and validation process and will contribute to a holistic verification strategy that strives towards trustworthy automotive perception systems |
| 2026-09-22 | [Hydrozoan: Latency-Adaptive DAG Consensus under Mixed Byzantine and Crash Faults](http://arxiv.org/abs/2609.25918v1) | Qianyu Yu, Lefteris Kokoris-Kogias et al. | DAG-based consensus protocols can achieve great throughput and the optimal three-message-delay limit for n = 3f+1 consensus. While two-delay protocols exist, they pay with reduced resilience (requiring 5f+1-style committees) or rely on fallbacks that sacrifice the DAG's high throughput. This paper introduces Hydrozoan, the first DAG protocol with a dual commit path under a hybrid fault model of f Byzantine and c crashed validators, on n = 3f+c+2p+1 validators. Leaders commit in two message delays whenever at most p validators are faulty, and in three otherwise, with no extra messages, no view changes, and multiple leaders per round. Both paths are evaluated on the same DAG, using a novel graded indirect rule to reconcile them so that every honest validator reaches the same decision. We show that under geo-distributed conditions, which path is faster is a property of geography rather than the protocol, as rounds reaching a remote region cost far more than those that do not. The (f, c, p) knobs place the fast quorum where the deployment requires it, allowing a commit in two message delays. If misconfigured, Hydrozoan can still commit in three message delays: Hydrozoan commits on whichever path fires first. We also present Optimal-Hydrozoan, a variant that tolerates one more fault on the fast path, the first construction to match the known lower bound. The safety and liveness of both protocols are machine-checked in Lean 4. Our geo-distributed evaluation shows that Hydrozoan matches Mysticeti's throughput, commits ~25% faster when the fast quorum fits fast regions, and falls back to three message delays when it does not or past p faults, where existing two-delay protocols stall. |

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



