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
| 2026-04-13 | [Detecting Safety Violations Across Many Agent Traces](http://arxiv.org/abs/2604.11806v1) | Adam Stein, Davis Brown et al. | To identify safety violations, auditors often search over large sets of agent traces. This search is difficult because failures are often rare, complex, and sometimes even adversarially hidden and only detectable when multiple traces are analyzed together. These challenges arise in diverse settings such as misuse campaigns, covert sabotage, reward hacking, and prompt injection. Existing approaches struggle here for several reasons. Per-trace judges miss failures that only become visible across traces, naive agentic auditing does not scale to large trace collections, and fixed monitors are brittle to unanticipated behaviors. We introduce Meerkat, which combines clustering with agentic search to uncover violations specified in natural language. Through structured search and adaptive investigation of promising regions, Meerkat finds sparse failures without relying on seed scenarios, fixed workflows, or exhaustive enumeration. Across misuse, misalignment, and task gaming settings, Meerkat significantly improves detection of safety violations over baseline monitors, discovers widespread developer cheating on a top agent benchmark, and finds nearly 4x more examples of reward hacking on CyBench than previous audits. |
| 2026-04-13 | [Solving Physics Olympiad via Reinforcement Learning on Physics Simulators](http://arxiv.org/abs/2604.11805v1) | Mihir Prabhudesai, Aryan Satpathy et al. | We have witnessed remarkable advances in LLM reasoning capabilities with the advent of DeepSeek-R1. However, much of this progress has been fueled by the abundance of internet question-answer (QA) pairs, a major bottleneck going forward, since such data is limited in scale and concentrated mainly in domains like mathematics. In contrast, other sciences such as physics lack large-scale QA datasets to effectively train reasoning-capable models. In this work, we show that physics simulators can serve as a powerful alternative source of supervision for training LLMs for physical reasoning. We generate random scenes in physics engines, create synthetic question-answer pairs from simulated interactions, and train LLMs using reinforcement learning on this synthetic data. Our models exhibit zero-shot sim-to-real transfer to real-world physics benchmarks: for example, training solely on synthetic simulated data improves performance on IPhO (International Physics Olympiad) problems by 5-10 percentage points across model sizes. These results demonstrate that physics simulators can act as scalable data generators, enabling LLMs to acquire deep physical reasoning skills beyond the limitations of internet-scale QA data. Code available at: https://sim2reason.github.io/. |
| 2026-04-13 | [ClawGuard: A Runtime Security Framework for Tool-Augmented LLM Agents Against Indirect Prompt Injection](http://arxiv.org/abs/2604.11790v1) | Wei Zhao, Zhe Li et al. | Tool-augmented Large Language Model (LLM) agents have demonstrated impressive capabilities in automating complex, multi-step real-world tasks, yet remain vulnerable to indirect prompt injection. Adversaries exploit this weakness by embedding malicious instructions within tool-returned content, which agents directly incorporate into their conversation history as trusted observations. This vulnerability manifests across three primary attack channels: web and local content injection, MCP server injection, and skill file injection. To address these vulnerabilities, we introduce \textsc{ClawGuard}, a novel runtime security framework that enforces a user-confirmed rule set at every tool-call boundary, transforming unreliable alignment-dependent defense into a deterministic, auditable mechanism that intercepts adversarial tool calls before any real-world effect is produced. By automatically deriving task-specific access constraints from the user's stated objective prior to any external tool invocation, \textsc{ClawGuard} blocks all three injection pathways without model modification or infrastructure change. Experiments across five state-of-the-art language models on AgentDojo, SkillInject, and MCPSafeBench demonstrate that \textsc{ClawGuard} achieves robust protection against indirect prompt injection without compromising agent utility. This work establishes deterministic tool-call boundary enforcement as an effective defense mechanism for secure agentic AI systems, requiring neither safety-specific fine-tuning nor architectural modification. Code is publicly available at https://github.com/Claw-Guard/ClawGuard. |
| 2026-04-13 | [$λ_A$: A Typed Lambda Calculus for LLM Agent Composition](http://arxiv.org/abs/2604.11767v1) | Qin Liu | Existing LLM agent frameworks lack formal semantics: there is no principled way to determine whether an agent configuration is well-formed or will terminate. We present $λ_A$, a typed lambda calculus for agent composition that extends the simply-typed lambda calculus with oracle calls, bounded fixpoints (the ReAct loop), probabilistic choice, and mutable environments. We prove type safety, termination of bounded fixpoints, and soundness of derived lint rules, with partial Coq mechanization (1,567 lines, 43 completed proofs). As a practical application, we derive a lint tool that detects structural configuration errors directly from the operational semantics. An evaluation on 835 real-world GitHub agent configurations shows that 94.1% are structurally incomplete under $λ_A$, with YAML-only lint precision at 54%, rising to 96--100% under joint YAML+Python AST analysis on 175 samples. This gap quantifies, for the first time, the degree of semantic entanglement between declarative configuration and imperative code in the agent ecosystem. We further show that five mainstream paradigms (LangGraph, CrewAI, AutoGen, OpenAI SDK, Dify) embed as typed $λ_A$ fragments, establishing $λ_A$ as a unifying calculus for LLM agent composition. |
| 2026-04-13 | [Multi-ORFT: Stable Online Reinforcement Fine-Tuning for Multi-Agent Diffusion Planning in Cooperative Driving](http://arxiv.org/abs/2604.11734v1) | Haojie Bai, Aimin Li et al. | Closed-loop cooperative driving requires planners that generate realistic multimodal multi-agent trajectories while improving safety and traffic efficiency. Existing diffusion planners can model multimodal behaviors from demonstrations, but they often exhibit weak scene consistency and remain poorly aligned with closed-loop objectives; meanwhile, stable online post-training in reactive multi-agent environments remains difficult. We present Multi-ORFT, which couples scene-conditioned diffusion pre-training with stable online reinforcement post-training. In pre-training, the planner uses inter-agent self-attention, cross-attention, and AdaLN-Zero-based scene conditioning to improve scene consistency and road adherence of joint trajectories. In post-training, we formulate a two-level MDP that exposes step-wise reverse-kernel likelihoods for online optimization, and combine dense trajectory-level rewards with variance-gated group-relative policy optimization (VG-GRPO) to stabilize training. On the WOMD closed-loop benchmark, Multi-ORFT reduces collision rate from 2.04% to 1.89% and off-road rate from 1.68% to 1.36%, while increasing average speed from 8.36 to 8.61 m/s relative to the pre-trained planner, and it outperforms strong open-source baselines including SMART-large, SMART-tiny-CLSFT, and VBD on the primary safety and efficiency metrics. These results show that coupling scene-consistent denoising with stable online diffusion-policy optimization improves the reliability of closed-loop cooperative driving. |
| 2026-04-13 | [A Mamba-Based Multimodal Network for Multiscale Blast-Induced Rapid Structural Damage Assessment](http://arxiv.org/abs/2604.11709v1) | Wanli Ma, Sivasakthy Selvakumaran et al. | Accurate and rapid structural damage assessment (SDA) is crucial for post-disaster management, helping responders prioritise resources, plan rescues, and support recovery. Traditional field inspections, though precise, are limited by accessibility, safety risks, and time constraints, especially after large explosions. Machine learning with remote sensing has emerged as a scalable solution for rapid SDA, with Mamba-based networks achieving state-of-the-art performance. However, these methods often require extensive training and large datasets, limiting real-world applicability. Moreover, they fail to incorporate key physical characteristics of blast loading for SDA. To overcome these challenges, we propose a Mamba-based multimodal network for rapid SDA that integrates multi-scale blast-loading information with optical remote sensing images. Evaluated on the 2020 Beirut explosion, our method significantly improves performance over state-of-the-art approaches. Code is available at: https://github.com/IMPACTSquad/Blast-Mamba |
| 2026-04-13 | [VLMaterial: Vision-Language Model-Based Camera-Radar Fusion for Physics-Grounded Material Identification](http://arxiv.org/abs/2604.11671v1) | Jiangyou Zhu, He Chen | Accurate material recognition is a fundamental capability for intelligent perception systems to interact safely and effectively with the physical world. For instance, distinguishing visually similar objects like glass and plastic cups is critical for safety but challenging for vision-based methods due to specular reflections, transparency, and visual deception. While millimeter-wave (mmWave) radar offers robust material sensing regardless of lighting, existing camera-radar fusion methods are limited to closed-set categories and lack semantic interpretability. In this paper, we introduce VLMaterial, a training-free framework that fuses vision-language models (VLMs) with domain-specific radar knowledge for physics-grounded material identification. First, we propose a dual-pipeline architecture: an optical pipeline uses the segment anything model and VLM for material candidate proposals, while an electromagnetic characterization pipeline extracts the intrinsic dielectric constant from radar signals via an effective peak reflection cell area (PRCA) method and weighted vector synthesis. Second, we employ a context-augmented generation (CAG) strategy to equip the VLM with radar-specific physical knowledge, enabling it to interpret electromagnetic parameters as stable references. Third, an adaptive fusion mechanism is introduced to intelligently integrate outputs from both sensors by resolving cross-modal conflicts based on uncertainty estimation. We evaluated VLMaterial in over 120 real-world experiments involving 41 diverse everyday objects and 4 typical visually deceptive counterfeits across varying environments. Experimental results demonstrate that VLMaterial achieves a recognition accuracy of 96.08%, delivering performance on par with state-of-the-art closed-set benchmarks while eliminating the need for extensive task-specific data collection and training. |
| 2026-04-13 | [Intersectional Sycophancy: How Perceived User Demographics Shape False Validation in Large Language Models](http://arxiv.org/abs/2604.11609v1) | Benjamin Maltbie, Shivam Raval | Large language models exhibit sycophantic tendencies--validating incorrect user beliefs to appear agreeable. We investigate whether this behavior varies systematically with perceived user demographics, testing whether combinations of race, age, gender, and expressed confidence level produce differential false validation rates. Inspired by the legal concept of intersectionality, we conduct 768 multi-turn adversarial conversations using Anthropic's Petri evaluation framework, probing GPT-5-nano and Claude Haiku 4.5 across 128 persona combinations in mathematics, philosophy, and conspiracy theory domains. GPT-5-nano is significantly more sycophantic than Claude Haiku 4.5 overall ($\bar{x}=2.96$ vs. $1.74$, $p < 10^{-32}$, Wilcoxon signed-rank). For GPT-5-nano, we find that philosophy elicits 41% more sycophancy than mathematics and that Hispanic personas receive the highest sycophancy across races. The worst-scoring persona, a confident, 23-year-old Hispanic woman, averages 5.33/10 on sycophancy. Claude Haiku 4.5 exhibits uniformly low sycophancy with no significant demographic variation. These results demonstrate that sycophancy is not uniformly distributed across users and that safety evaluations should incorporate identity-aware testing. |
| 2026-04-13 | [Decomposing and Reducing Hidden Measurement Error in LLM Evaluation Pipelines](http://arxiv.org/abs/2604.11581v1) | Solomon Messing | LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet these scores carry hidden uncertainty: rephrasing the prompt, switching the judge model, or changing the temperature can shift results enough to flip rankings and reverse conclusions. Standard confidence intervals ignore this variance, producing under-coverage that worsens with more data. The unmeasured variance also creates an exploitable surface: model developers can optimize against measurement noise rather than genuine capability. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and projects the most efficient path to reducing total error. For benchmark builders, the same decomposition identifies which design choices contribute exploitable surface for gaming and prescribes designs that minimize it. Across ideology annotation, safety classification, MMLU benchmarking, and a human-validated propaganda audit, projection-optimized pipelines outperform 73\% of possible naive pipelines against a human baseline. On MMLU, optimized budget allocation halves estimation error compared to standard single-prompt evaluation at equivalent cost. A small-sample variance estimation exercise is sufficient to derive confidence intervals that approach nominal coverage when the model includes the relevant pipeline facets, and to generate recommendations for reducing measurement error and improving benchmark robustness. |
| 2026-04-13 | [SemaClaw: A Step Towards General-Purpose Personal AI Agents through Harness Engineering](http://arxiv.org/abs/2604.11548v1) | Ningyan Zhu, Huacan Wang et al. | The rise of OpenClaw in early 2026 marks the moment when millions of users began deploying personal AI agents into their daily lives, delegating tasks ranging from travel planning to multi-step research. This scale of adoption signals that two parallel arcs of development have reached an inflection point. First is a paradigm shift in AI engineering, evolving from prompt and context engineering to harness engineering-designing the complete infrastructure necessary to transform unconstrained agents into controllable, auditable, and production-reliable systems. As model capabilities converge, this harness layer is becoming the primary site of architectural differentiation. Second is the evolution of human-agent interaction from discrete tasks toward a persistent, contextually aware collaborative relationship, which demands open, trustworthy and extensible harness infrastructure. We present SemaClaw, an open-source multi-agent application framework that addresses these shifts by taking a step towards general-purpose personal AI agents through harness engineering. Our primary contributions include a DAG-based two-phase hybrid agent team orchestration method, a PermissionBridge behavioral safety system, a three-tier context management architecture, and an agentic wiki skill for automated personal knowledge base construction. |
| 2026-04-13 | [From Translation to Superset: Benchmark-Driven Evolution of a Production AI Agent from Rust to Python](http://arxiv.org/abs/2604.11518v1) | Jinhua Wang, Biswa Sengupta | Cross-language migration of large software systems is a persistent engineering challenge, particularly when the source codebase evolves rapidly. We present a methodology for LLM-assisted continuous code translation in which a large language model translates a production Rust codebase (648K LOC, 65 crates) into Python (41K LOC, 28 modules), with public agent benchmarks as the objective function driving iterative refinement. Our subject system is Codex CLI, a production AI coding agent. We demonstrate that: (1) the Python port resolves 59/80 SWE-bench Verified tasks (73.8%) versus Rust's 56/80 (70.0%), and achieves 42.5% on Terminal-Bench versus Rust's 47.5%, confirming near-parity on real-world agentic tasks; (2) benchmark-driven debugging, revealing API protocol mismatches, environment pollution, a silent WebSocket failure mode, and an API 400 crash, is more effective than static testing alone; (3) the architecture supports continuous upstream synchronisation via an LLM-assisted diff-translate-test loop; and (4) the Python port has evolved into a capability superset with 30 feature-flagged extensions (multi-agent orchestration, semantic memory, guardian safety, cost tracking) absent from Rust, while preserving strict parity mode for comparison. Our evaluation shows that for LLM-based agents where API latency dominates, Python's expressiveness yields a 15.9x code reduction with negligible performance cost, while the benchmark-as-objective-function methodology provides a principled framework for growing a cross-language port from parity into an extended platform. |
| 2026-04-13 | [CAGenMol: Condition-Aware Diffusion Language Model for Goal-Directed Molecular Generation](http://arxiv.org/abs/2604.11483v1) | Yanting Li, Zhuoyang Jiang et al. | Goal-directed molecular generation requires satisfying heterogeneous constraints such as protein--ligand compatibility and multi-objective drug-like properties, yet existing methods often optimize these constraints in isolation, failing to reconcile conflicting objectives (e.g., affinity vs. safety), and struggle to navigate the non-differentiable chemical space without compromising structural validity. To address these challenges, we propose CAGenMol, a condition-aware discrete diffusion framework over molecular sequences that formulates molecular design as conditional denoising guided by heterogeneous structural and property signals. By coupling discrete diffusion with reinforcement learning, the model aligns the generation trajectory with non-differentiable objectives while preserving chemical validity and diversity. The non-autoregressive nature of diffusion language model further enables iterative refinement of molecular fragments at inference time. Experiments on structure-conditioned, property-conditioned, and dual-conditioned benchmarks demonstrate consistent improvements over state-of-the-art methods in binding affinity, drug-likeness, and success rate, highlighting the effectiveness of our framework. |
| 2026-04-13 | [Safe Human-to-Humanoid Motion Imitation Using Control Barrier Functions](http://arxiv.org/abs/2604.11447v1) | Wenqi Cai, John Abanes et al. | Ensuring operational safety is critical for human-to-humanoid motion imitation. This paper presents a vision-based framework that enables a humanoid robot to imitate human movements while avoiding collisions. Human skeletal keypoints are captured by a single camera and converted into joint angles for motion retargeting. Safety is enforced through a Control Barrier Function (CBF) layer formulated as a Quadratic Program (QP), which filters imitation commands to prevent both self-collisions and human-robot collisions. Simulation results validate the effectiveness of the proposed framework for real-time collision-aware motion imitation. |
| 2026-04-13 | [Minimal Embodiment Enables Efficient Learning of Number Concepts in Robot](http://arxiv.org/abs/2604.11373v1) | Zhegong Shangguan, Alessandro Di Nuovo et al. | Robots are increasingly entering human-interactive scenarios that require understanding of quantity. How intelligent systems acquire abstract numerical concepts from sensorimotor experience remains a fundamental challenge in cognitive science and artificial intelligence. Here we investigate embodied numerical learning using a neural network model trained to perform sequential counting through naturalistic robotic interaction with a Franka Panda manipulator. We demonstrate that embodied models achieve 96.8\% counting accuracy with only 10\% of training data, compared to 60.6\% for vision-only baselines. This advantage persists when visual-motor correspondences are randomized, indicating that embodiment functions as a structural prior that regularizes learning rather than as an information source. The model spontaneously develops biologically plausible representations: number-selective units with logarithmic tuning, mental number line organization, Weber-law scaling, and rotational dynamics encoding numerical magnitude ($r = 0.97$, slope $= 30.6°$/count). The learning trajectory parallels children's developmental progression from subset-knowers to cardinal-principle knowers. These findings demonstrate that minimal embodiment can ground abstract concepts, improve data efficiency, and yield interpretable representations aligned with biological cognition, which may contribute to embodied mathematics tutoring and safety-critical industrial applications. |
| 2026-04-13 | [The Salami Slicing Threat: Exploiting Cumulative Risks in LLM Systems](http://arxiv.org/abs/2604.11309v1) | Yihao Zhang, Kai Wang et al. | Large Language Models (LLMs) face prominent security risks from jailbreaking, a practice that manipulates models to bypass built-in security constraints and generate unethical or unsafe content. Among various jailbreak techniques, multi-turn jailbreak attacks are more covert and persistent than single-turn counterparts, exposing critical vulnerabilities of LLMs.   However, existing multi-turn jailbreak methods suffer from two fundamental limitations that affect the actual impact in real-world scenarios: (a) As models become more context-aware, any explicit harmful trigger is increasingly likely to be flagged and blocked; (b) Successful final-step triggers often require finely tuned, model-specific contexts, making such attacks highly context-dependent. To fill this gap, we propose \textit{Salami Slicing Risk}, which operates by chaining numerous low-risk inputs that individually evade alignment thresholds but cumulatively accumulate harmful intent to ultimately trigger high-risk behaviors, without heavy reliance on pre-designed contextual structures. Building on this risk, we develop Salami Attack, an automatic framework universally applicable to multiple model types and modalities.   Rigorous experiments demonstrate its state-of-the-art performance across diverse models and modalities, achieving over 90\% Attack Success Rate on GPT-4o and Gemini, as well as robustness against real-world alignment defenses. We also proposed a defense strategy to constrain the Salami Attack by at least 44.8\% while achieving a maximum blocking rate of 64.8\% against other multi-turn jailbreak attacks. Our findings provide critical insights into the pervasive risks of multi-turn jailbreaking and offer actionable mitigation strategies to enhance LLM security. |
| 2026-04-13 | [Consistency of AI-Generated Exercise Prescriptions: A Repeated Generation Study Using a Large Language Model](http://arxiv.org/abs/2604.11287v1) | Kihyuk Lee | Background: Large language models (LLMs) have been explored as tools for generating personalized exercise prescriptions, yet the consistency of outputs under identical conditions remains insufficiently examined. Objective: This study evaluated the intra-model consistency of LLM-generated exercise prescriptions using a repeated generation design. Methods: Six clinical scenarios were used to generate exercise prescriptions using Gemini 2.5 Flash (20 outputs per scenario; total n = 120). Consistency was assessed across three dimensions: (1) semantic consistency using SBERT-based cosine similarity, (2) structural consistency based on the FITT principle using an AI-as-a-judge approach, and (3) safety expression consistency, including inclusion rates and sentence-level quantification. Results: Semantic similarity was high across scenarios (mean cosine similarity: 0.879-0.939), with greater consistency in clinically constrained cases. Frequency showed consistent patterns, whereas variability was observed in quantitative components, particularly exercise intensity. Unclassifiable intensity expressions were observed in 10-25% of resistance training outputs. Safety-related expressions were included in 100% of outputs; however, safety sentence counts varied significantly across scenarios (H=86.18, p less than 0.001), with clinical cases generating more safety expressions than healthy adult cases. Conclusions: LLM-generated exercise prescriptions demonstrated high semantic consistency but showed variability in key quantitative components. Reliability depends substantially on prompt structure, and additional structural constraints and expert validation are needed before clinical deployment. |
| 2026-04-13 | [EmbodiedGovBench: A Benchmark for Governance, Recovery, and Upgrade Safety in Embodied Agent Systems](http://arxiv.org/abs/2604.11174v1) | Xue Qin, Simin Luan et al. | Recent progress in embodied AI has produced a growing ecosystem of robot policies, foundation models, and modular runtimes. However, current evaluation remains dominated by task success metrics such as completion rate or manipulation accuracy. These metrics leave a critical gap: they do not measure whether embodied systems are governable -- whether they respect capability boundaries, enforce policies, recover safely, maintain audit trails, and respond to human oversight. We present EmbodiedGovBench, a benchmark for governance-oriented evaluation of embodied agent systems. Rather than asking only whether a robot can complete a task, EmbodiedGovBench evaluates whether the system remains controllable, policy-bounded, recoverable, auditable, and evolution-safe under realistic perturbations. The benchmark covers seven governance dimensions: unauthorized capability invocation, runtime drift robustness, recovery success, policy portability, version upgrade safety, human override responsiveness, and audit completeness. We define a benchmark structure spanning single-robot and fleet settings, with scenario templates, perturbation operators, governance metrics, and baseline evaluation protocols. We describe how the benchmark can be instantiated over embodied capability runtimes with modular interfaces and contract-aware upgrade workflows. Our analysis suggests that embodied governance should become a first-class evaluation target. EmbodiedGovBench provides the initial measurement framework for that shift. |
| 2026-04-13 | [From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](http://arxiv.org/abs/2604.11137v1) | Chen Zhan, Xiaoyu Tan et al. | The integration of Large Language Models (LLMs) into clinical decision support is critically obstructed by their opaque and often unreliable reasoning. In the high-stakes domain of healthcare, correct answers alone are insufficient; clinical practice demands full transparency to ensure patient safety and enable professional accountability. A pervasive and dangerous weakness of current LLMs is their tendency to produce "correct answers through flawed reasoning." This issue is far more than a minor academic flaw; such process errors signal a fundamental lack of robust understanding, making the model prone to broader hallucinations and unpredictable failures when faced with real-world clinical complexity. In this paper, we establish a framework for trustworthy clinical argumentation by adapting the Toulmin model to the diagnostic process. We propose a novel training pipeline: Curriculum Goal-Conditioned Learning (CGCL), designed to progressively train LLM to generate diagnostic arguments that explicitly follow this Toulmin structure. CGCL's progressive three-stage curriculum systematically builds a solid clinical argument: (1) extracting facts and generating differential diagnoses; (2) justifying a core hypothesis while rebutting alternatives; and (3) synthesizing the analysis into a final, qualified conclusion. We validate CGCL using T-Eval, a quantitative framework measuring the integrity of the diagnosis reasoning. Experiments show that our method achieves diagnostic accuracy and reasoning quality comparable to resource-intensive Reinforcement Learning (RL) methods, while offering a more stable and efficient training pipeline. |
| 2026-04-13 | [Persona Non Grata: Single-Method Safety Evaluation Is Incomplete for Persona-Imbued LLMs](http://arxiv.org/abs/2604.11120v1) | Wenkai Li, Fan Yang et al. | Personality imbuing customizes LLM behavior, but safety evaluations almost always study prompt-based personas alone. We show this is incomplete: prompting and activation steering expose *different*, architecture-dependent vulnerability profiles, and testing with only one method can miss a model's dominant failure mode. Across 5,568 judged conditions on four standard models from three architecture families, persona danger rankings under system prompting are preserved across all architectures ($ρ= 0.71$--$0.96$), but activation-steering vulnerability diverges sharply and cannot be predicted from prompt-side rankings: Llama-3.1-8B is substantially more AS-vulnerable, whereas Gemma-3-27B and Qwen3.5 are more vulnerable to prompting. The most striking illustration of this divergence is the *prosocial persona paradox*: on Llama-3.1-8B, P12 (high conscientiousness + high agreeableness) is among the safest personas under prompting yet becomes the highest-ASR activation-steered persona (ASR ~0.818). This is an inversion robust to coefficient ablation and matched-strength calibration, and replicated on DeepSeek-R1-Distill-Qwen-32B. A trait refusal alignment framework, in which conscientiousness is strongly anti-aligned with refusal on Llama-3.1-8B, offers a partial geometric account. Reasoning provides only partial protection: two 32B reasoning models reach 15--18% prompt-side ASR, and activation steering separates them sharply in both baseline susceptibility and persona-specific vulnerability. Heuristic trace diagnostics suggest that the safer model retains stronger policy recall and self-correction behavior, not merely longer reasoning. |
| 2026-04-13 | [PRISM Risk Signal Framework: Hierarchy-Based Red Lines for AI Behavioral Risk](http://arxiv.org/abs/2604.11070v1) | Seulki Lee | Current approaches to AI safety define red lines at the case level: specific prompts, specific outputs, specific harms. This paper argues that red lines can be set more fundamentally -- at the level of value, evidence, and source hierarchies that govern AI reasoning. Using the PRISM (Profile-based Reasoning Integrity Stack Measurement) framework, we define a taxonomy of 27 behavioral risk signals derived from structural anomalies in how AI systems prioritize values (L4), weight evidence types (L3), and trust information sources (L2). Each signal is evaluated through a dual-threshold principle combining absolute rank position and relative win-rate gap, producing a two-tier classification (Confirmed Risk vs. Watch Signal). The hierarchy-based approach offers three advantages over case-specific red lines: it is anticipatory rather than reactive (detecting dangerous reasoning structures before they produce harmful outputs), comprehensive rather than enumerative (a single value-hierarchy signal subsumes an unlimited number of case-specific violations), and measurable rather than subjective (grounded in empirical forced-choice data). We demonstrate the framework's detection capacity using approximately 397,000 forced-choice responses from 7 AI models across three Authority Stack layers, showing that the signal taxonomy successfully discriminates between models with structurally extreme profiles, models with context-dependent risk, and models with balanced hierarchies. |

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



