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
| 2026-03-11 | [Oxygenated False Positive Biosignatures in Mars-like Exoplanet Atmospheres](http://arxiv.org/abs/2603.11017v1) | Margaret Turcotte Seavey, Shawn Domagal-Goldman et al. | Oxygen is a well-studied biosignature. Studying potential abiotic pathways for O2 build-up in exoplanet atmospheres is essential for evaluating whether the detection of O2 would constitute a biosignature detection on other worlds. Previous modeling efforts in the literature demonstrated that detectable abiotic O2 and O3 can be produced through CO2 photolysis for rocky planets around M dwarf stars. Building on modeling approaches from previous studies, we use photochemical simulations to reassess the conditions under which O2 and O3 may accumulate through similar photochemical mechanisms. Using a Mars-like atmospheric composition and planetary parameters, we vary the hydrogen mole fraction to assess how changes in HOx chemistry can affect the resulting accumulation of abiotic O2 and O3. Across the range of hydrogen mole fractions explored, we obtain a maximum O2 abundance of ~2.7% for H = 0.0065 ppm, about an order of magnitude lower than reported in the literature. This reduction is consistent with the elevated water vapor abundance adopted in our simulations, which enhances HOx-driven recycling of CO and O and thereby suppresses the accumulation of O2 and O3. Our improved understanding of how this cycle results in atmospheric false positive biosignatures in crucial towards developing future exoplanet characterization strategies. |
| 2026-03-11 | [RCTs & Human Uplift Studies: Methodological Challenges and Practical Solutions for Frontier AI Evaluation](http://arxiv.org/abs/2603.11001v1) | Patricia Paskov, Kevin Wei et al. | Human uplift studies - or studies that measure AI effects on human performance relative to a status quo, typically using randomized controlled trial (RCT) methodology - are increasingly used to inform deployment, governance, and safety decisions for frontier AI systems. While the methods underlying these studies are well-established, their interaction with the distinctive properties of frontier AI systems remains underexamined, particularly when results are used to inform high-stakes decisions. We present findings from interviews with 16 expert practitioners with experience conducting human uplift studies in domains including biosecurity, cybersecurity, education, and labor. Across interviews, experts described a recurring tension between standard causal inference assumptions and the object of study itself. Rapidly evolving AI systems, shifting baselines, heterogeneous and changing user proficiency, and porous real-world settings strain assumptions underlying internal, external, and construct validity, complicating the interpretation and appropriate use of uplift evidence. We synthesize these challenges across key stages of the human uplift research lifecycle and map them to practitioner-reported solutions, clarifying both the limits and the appropriate uses of evidence from human uplift studies in high-stakes decision-making. |
| 2026-03-11 | [STADA: Specification-based Testing for Autonomous Driving Agents](http://arxiv.org/abs/2603.10940v1) | Joy Saha, Trey Woodlief et al. | Simulation-based testing has become a standard approach to validating autonomous driving agents prior to real-world deployment. A high-quality validation campaign will exercise an agent in diverse contexts comprised of varying static environments, e.g., lanes, intersections, signage, and dynamic elements, e.g., vehicles and pedestrians. To achieve this, existing test generation techniques rely on template-based, manually constructed, or random scenario generation. When applied to validate formally specified safety requirements, such methods either require significant human effort or run the risk of missing important behavior related to the requirement.   To address this gap, we present STADA, a Specification-based Test generation framework for Autonomous Driving Agents that systematically generates the space of scenarios defined by a formal specification expressed in temporal logic (LTLf). Given a specification, STADA constructs all distinct initial scenes, a diverse space of continuations of those scenes, and simulations that reflect the behaviors of the specification.   Evaluation of STADA on a variety of LTLf specifications formalized in SCENEFLOW using three complementary coverage criteria demonstrates that STADA yields more than 2x higher coverage than the best baseline on the finest criteria and a 75% increase for the coarsest criteria. Moreover, it matches the coverage of the best baseline with 6 times fewer simulations. While set in the context of autonomous driving, the approach is applicable to other domains with rich simulation environments. |
| 2026-03-11 | [Safe RLHF Beyond Expectation: Stochastic Dominance for Universal Spectral Risk Control](http://arxiv.org/abs/2603.10938v1) | Yaswanth Chittepu, Ativ Joshi et al. | Safe Reinforcement Learning from Human Feedback (RLHF) typically enforces safety through expected cost constraints, but the expectation captures only a single statistic of the cost distribution and fails to account for distributional uncertainty, particularly under heavy tails or rare catastrophic events. This limitation is problematic when robustness and risk sensitivity are critical. Stochastic dominance offers a principled alternative by comparing entire cost distributions rather than just their averages, enabling direct control over tail risks and potential out-of-distribution failures that expectation-based constraints may overlook. In this work, we propose Risk-sensitive Alignment via Dominance (RAD), a novel alignment framework that replaces scalar expected cost constraints with First-Order Stochastic Dominance (FSD) constraints. We operationalize this constraint by comparing the target policy's cost distribution to that of a reference policy within an Optimal Transport (OT) framework, using entropic regularization and Sinkhorn iterations to obtain a differentiable and computationally efficient objective for stable end-to-end optimization. Furthermore, we introduce quantile-weighted FSD constraints and show that weighted FSD universally controls a broad class of Spectral Risk Measures (SRMs), so that improvements under weighted dominance imply guaranteed improvements in the corresponding spectral risk. This provides a principled mechanism for tuning a model's risk profile via the quantile weighting function. Empirical results demonstrate that RAD improves harmlessness over baselines while remaining competitive in helpfulness, and exhibits greater robustness on out-of-distribution harmlessness evaluations. |
| 2026-03-11 | [LLM2Vec-Gen: Generative Embeddings from Large Language Models](http://arxiv.org/abs/2603.10913v1) | Parishad BehnamGhader, Vaibhav Adlakha et al. | LLM-based text embedders typically encode the semantic content of their input. However, embedding tasks require mapping diverse inputs to similar outputs. Typically, this input-output is addressed by training embedding models with paired data using contrastive learning. In this work, we propose a novel self-supervised approach, LLM2Vec-Gen, which adopts a different paradigm: rather than encoding the input, we learn to represent the model's potential response. Specifically, we add trainable special tokens to the LLM's vocabulary, append them to input, and optimize them to represent the LLM's response in a fixed-length sequence. Training is guided by the LLM's own completion for the query, along with an unsupervised embedding teacher that provides distillation targets. This formulation helps to bridge the input-output gap and transfers LLM capabilities such as safety alignment and reasoning to embedding tasks. Crucially, the LLM backbone remains frozen and training requires only unlabeled queries. LLM2Vec-Gen achieves state-of-the-art self-supervised performance on the Massive Text Embedding Benchmark (MTEB), improving by 9.3% over the best unsupervised embedding teacher. We also observe up to 43.2% reduction in harmful content retrieval and 29.3% improvement in reasoning capabilities for embedding tasks. Finally, the learned embeddings are interpretable and can be decoded into text to reveal their semantic content. |
| 2026-03-11 | [A Hybrid Knowledge-Grounded Framework for Safety and Traceability in Prescription Verification](http://arxiv.org/abs/2603.10891v1) | Yichi Zhu, Kan Ling et al. | Medication errors pose a significant threat to patient safety, making pharmacist verification (PV) a critical, yet heavily burdened, final safeguard. The direct application of Large Language Models (LLMs) to this zero-tolerance domain is untenable due to their inherent factual unreliability, lack of traceability, and weakness in complex reasoning. To address these challenges, we introduce PharmGraph-Auditor, a novel system designed for safe and evidence-grounded prescription auditing. The core of our system is a trustworthy Hybrid Pharmaceutical Knowledge Base (HPKB), implemented under the Virtual Knowledge Graph (VKG) paradigm. This architecture strategically unifies a relational component for set constraint satisfaction and a graph component for topological reasoning via a rigorous mapping layer. To construct this HPKB, we propose the Iterative Schema Refinement (ISR) algorithm, a framework that enables the co-evolution of both graph and relational schemas from medical texts. For auditing, we introduce the KB-grounded Chain of Verification (CoV), a new reasoning paradigm that transforms the LLM from an unreliable generator into a transparent reasoning engine. CoV decomposes the audit task into a sequence of verifiable queries against the HPKB, generating hybrid query plans to retrieve evidence from the most appropriate data store. Experimental results demonstrate robust knowledge extraction capabilities and show promises of using PharmGraph-Auditor to enable pharmacists to achieve safer and faster prescription verification. |
| 2026-03-11 | [Distributed Safety Critical Control among Uncontrollable Agents using Reconstructed Control Barrier Functions](http://arxiv.org/abs/2603.10836v1) | Yuzhang Peng, Wei Wang et al. | This paper investigates the distributed safety critical control for multi-agent systems (MASs) in the presence of uncontrollable agents with uncertain behaviors. To ensure system safety, the control barrier function (CBF) is employed in this paper. However, a key challenge is that the CBF constraints are coupled when MASs perform collaborative tasks, which depend on information from multiple agents and impede the design of a fully distributed safe control scheme. To overcome this, a novel reconstructed CBF approach is proposed. In this method, the coupled CBF is reconstructed by leveraging state estimates of other agents obtained from a distributed adaptive observer. Furthermore, a prescribed performance adaptive parameter is designed to modify this reconstruction, ensuring that satisfying the reconstructed CBF constraint is sufficient to meet the original coupled one. Based on the reconstructed CBF, we design a safety-critical quadratic programming (QP) controller and prove that the proposed distributed control scheme rigorously guarantees the safety of the MAS, even in the uncertain dynamic environments involving uncontrollable agents. The effectiveness of the proposed method is illustrated through simulations. |
| 2026-03-11 | [Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](http://arxiv.org/abs/2603.10833v1) | W. I. Chu, G. Tarroni et al. | Adverse drug events are a significant source of preventable harm, which has led to the development of automated pill recognition systems to enhance medication safety. Real-world deployment of these systems is hindered by visually complex conditions, including cluttered scenes, overlapping pills, reflections, and diverse acquisition environments. This study investigates few-shot pill recognition from a deployment-oriented perspective, prioritizing generalization under realistic cross-dataset domain shifts over architectural innovation. A two-stage object detection framework is employed, involving base training followed by few-shot fine-tuning. Models are adapted to novel pill classes using one, five, or ten labeled examples per class and are evaluated on a separate deployment dataset featuring multi-object, cluttered scenes. The evaluation focuses on classification-centric and error-based metrics to address heterogeneous annotation strategies. Findings indicate that semantic pill recognition adapts rapidly with few-shot supervision, with classification performance reaching saturation even with a single labeled example. However, stress testing under overlapping and occluded conditions demonstrates a marked decline in localization and recall, despite robust semantic classification. Models trained on visually realistic, multi-pill data consistently exhibit greater robustness in low-shot scenarios, underscoring the importance of training data realism and the diagnostic utility of few-shot fine-tuning for deployment readiness. |
| 2026-03-11 | [Kinematics of Wolf-Rayet Stars in the LMC: Clues to Subtype Origins](http://arxiv.org/abs/2603.10826v1) | Caden Burkhardt, Fiona Han et al. | We measure transverse proper motion velocities of LMC Wolf-Rayet (WR) stars using Gaia DR3 astrometry. The combined velocity distribution of WNh, O If*/WN, and WNL very massive stars ($>100\ M_\odot$; VMS) shows both slow, unejected objects ($v_\perp < 10$ $\rm km\ s^{-1}$) and stars dominated by fast, runaway velocities ($v_\perp > 24$ $\rm km\ s^{-1}$). This supports expectations that VMS ages are comparable to the dynamical ejection timescale ($\sim1.5$ Myr). These kinematics share similarities with those of lower-luminosity, classical WNh, O If*/WN, and WNL stars, as well as the SMC field OB stars, suggesting that dynamical ejections may also dominate these populations. In contrast, both single and binary WNE stars are ejected populations that show single-peaked velocity distributions, suggesting a different ejection mechanism(s). We speculate that single WNE stars might result from explosive mergers onto the shell-burning layer, thereby stripping the H envelope. Binary WC stars appear to be faster (median $v_\perp = 54$ $\rm km\ s^{-1}$) and have higher luminosities than singles (median $v_\perp = 38$ $\rm km\ s^{-1}$), suggesting that single WC stars are not descendants of the binaries. Thus, the binaries are probably stripped by mass transfer, while the WC singles likely originate from another process. The high velocities of binary WC stars are consistent with some predictions that lower mass clusters generate fast dynamical ejections. Single WC and WN3/O3 stars have ambiguous kinematics, but both show high $v_\perp$ (median $\sim 38$ $\rm km\ s^{-1}$), possibly linked to their lower masses. |
| 2026-03-11 | [A dataset of medication images with instance segmentation masks for preventing adverse drug events](http://arxiv.org/abs/2603.10825v1) | W. I. Chu, S. Hirani et al. | Medication errors and adverse drug events (ADEs) pose significant risks to patient safety, often arising from difficulties in reliably identifying pharmaceuticals in real-world settings. AI-based pill recognition models offer a promising solution, but the lack of comprehensive datasets hinders their development. Existing pill image datasets rarely capture real-world complexities such as overlapping pills, varied lighting, and occlusions. MEDISEG addresses this gap by providing instance segmentation annotations for 32 distinct pill types across 8262 images, encompassing diverse conditions from individual pill images to cluttered dosette boxes. We trained YOLOv8 and YOLOv9 on MEDISEG to demonstrate their usability, achieving mean average precision at IoU 0.5 of 99.5 percent on the 3-Pills subset and 80.1 percent on the 32-Pills subset. We further evaluate MEDISEG under a few-shot detection protocol, demonstrating that base training on MEDISEG significantly improves recognition of unseen pill classes in occluded multi-pill scenarios compared to existing datasets. These results highlight the dataset's ability not only to support robust supervised training but also to promote transferable representations under limited supervision, making it a valuable resource for developing and benchmarking AI-driven systems for medication safety. |
| 2026-03-11 | [Risk-Adjusted Harm Scoring for Automated Red Teaming for LLMs in Financial Services](http://arxiv.org/abs/2603.10807v1) | Fabrizio Dimino, Bhaskarjit Sarmah et al. | The rapid adoption of large language models (LLMs) in financial services introduces new operational, regulatory, and security risks. Yet most red-teaming benchmarks remain domain-agnostic and fail to capture failure modes specific to regulated BFSI settings, where harmful behavior can be elicited through legally or professionally plausible framing. We propose a risk-aware evaluation framework for LLM security failures in Banking, Financial Services, and Insurance (BFSI), combining a domain-specific taxonomy of financial harms, an automated multi-round red-teaming pipeline, and an ensemble-based judging protocol. We introduce the Risk-Adjusted Harm Score (RAHS), a risk-sensitive metric that goes beyond success rates by quantifying the operational severity of disclosures, accounting for mitigation signals, and leveraging inter-judge agreement. Across diverse models, we find that higher decoding stochasticity and sustained adaptive interaction not only increase jailbreak success, but also drive systematic escalation toward more severe and operationally actionable financial disclosures. These results expose limitations of single-turn, domain-agnostic security evaluation and motivate risk-sensitive assessment under prolonged adversarial pressure for real-world BFSI deployment. |
| 2026-03-11 | [A Control-Theoretic Foundation for Agentic Systems](http://arxiv.org/abs/2603.10779v1) | Ali Eslami, Jiangbo Yu | This paper develops a control-theoretic framework for analyzing agentic systems embedded within feedback control loops. In such systems, an AI agent may adapt controller parameters, select among control strategies, invoke tools, reconfigure decision architectures, or modify control objectives during operation. We formalize these capabilities by interpreting agency as hierarchical decision authority over the control architecture. A unified dynamical representation is introduced that incorporates memory, learning, tool activation, interaction signals, and goal descriptors within a single closed-loop structure. Based on this representation, we define a five-level hierarchy of agency ranging from reactive rule-based control to the synthesis of control objectives and controller architectures. The framework is presented in both nonlinear and linear settings, allowing agentic behaviors to be interpreted using standard control-theoretic constructs such as feedback gains, switching signals, parameter adaptation laws, and quadratic cost functions. The analysis shows that increasing agency introduces dynamical mechanisms including time-varying adaptation, endogenous switching, decision-induced delays, and structural reconfiguration of the control pipeline. This perspective provides a mathematical foundation for analyzing stability, safety, and performance of AI-enabled control systems. |
| 2026-03-11 | [OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency](http://arxiv.org/abs/2603.10682v1) | Guiyong Zheng, Yueting Ban et al. | Aerial vision-language navigation (AVLN) enables UAVs to follow natural-language instructions in complex 3D environments. However, existing zero-shot AVLN methods often suffer from unstable single-stream Vision-Language Model decision-making, unreliable long-horizon progress monitoring, and a trade-off between safety and efficiency. We propose OnFly, a fully onboard, real-time framework for zero-shot AVLN. OnFly adopts a shared-perception dual-agent architecture that decouples high-frequency target generation from low-frequency progress monitoring, thereby stabilizing decision-making. It further employs a hybrid keyframe-recent-frame memory to preserve global trajectory context while maintaining KV-cache prefix stability, enabling reliable long-horizon monitoring with termination and recovery signals. In addition, a semantic-geometric verifier refines VLM-predicted targets for instruction consistency and geometric safety using VLM features and depth cues, while a receding-horizon planner generates optimized collision-free trajectories under geometric safety constraints, improving both safety and efficiency. In simulation, OnFly improves task success from 26.4% to 67.8%, compared with the strongest state-of-the-art baseline, while fully onboard real-world flights validate its feasibility for real-time deployment. The code will be released at https://github.com/Robotics-STAR-Lab/OnFly |
| 2026-03-11 | [Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting](http://arxiv.org/abs/2603.10638v1) | Hansol Lim, Jongseong Brad Choi | Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift. |
| 2026-03-11 | [AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments](http://arxiv.org/abs/2603.10616v1) | Zixuan Chen, Wenquan Zhang et al. | In densely cluttered environments, physical interference, visual occlusions, and unstable contacts often cause direct dexterous grasping to fail, while aggressive singulation strategies may compromise safety. Enabling robots to adaptively decide whether to clear surrounding objects or directly grasp the target is therefore crucial for robust manipulation. We propose AdaClearGrasp, a closed-loop decision-execution framework for adaptive clearing and zero-shot dexterous grasping in densely cluttered environments. The framework formulates manipulation as a controllable high-level decision process that determines whether to directly grasp the target or first clear surrounding objects. A pretrained vision-language model (VLM) interprets visual observations and language task descriptions to reason about grasp interference and generate a high-level planning skeleton, which invokes structured atomic skills through a unified action interface. For dexterous grasping, we train a reinforcement learning policy with a relative hand-object distance representation, enabling zero-shot generalization across diverse object geometries and physical properties. During execution, visual feedback monitors outcomes and triggers replanning upon failures, forming a closed-loop correction mechanism. To evaluate language-conditioned dexterous grasping in clutter, we introduce Clutter-Bench, the first simulation benchmark with graded clutter complexity. It includes seven target objects across three clutter levels, yielding 210 task scenarios. We further perform sim-to-real experiments on three objects under three clutter levels (18 scenarios). Results demonstrate that AdaClearGrasp significantly improves grasp success rates in densely cluttered environments. For more videos and code, please visit our project website: https://chenzixuan99.github.io/adaclear-grasp.github.io/. |
| 2026-03-11 | [An Approach for Safe and Secure Software Protection Supported by Symbolic Execution](http://arxiv.org/abs/2603.10608v1) | Daniel Dorfmeister, Flavio Ferrarotti et al. | We introduce a novel copy-protection method for industrial control software. With our method, a program executes correctly only on its target hardware and behaves differently on other machines. The hardware-software binding is based on Physically Unclonable Functions (PUFs). We use symbolic execution to guarantee the preservation of safety properties if the software is executed on a different machine, or if there is a problem with the PUF response. Moreover, we show that the protection method is also secure against reverse engineering. |
| 2026-03-11 | [Safety-critical Control Under Partial Observability: Reach-Avoid POMDP meets Belief Space Control](http://arxiv.org/abs/2603.10572v1) | Matti Vahs, Joris Verhagen et al. | Partially Observable Markov Decision Processes (POMDPs) provide a principled framework for robot decision-making under uncertainty. Solving reach-avoid POMDPs, however, requires coordinating three distinct behaviors: goal reaching, safety, and active information gathering to reduce uncertainty. Existing online POMDP solvers attempt to address all three within a single belief tree search, but this unified approach struggles with the conflicting time scales inherent to these objectives. We propose a layered, certificate-based control architecture that operates directly in belief space, decoupling goal reaching, information gathering, and safety into modular components. We introduce Belief Control Lyapunov Functions (BCLFs) that formalize information gathering as a Lyapunov convergence problem in belief space, and show how they can be learned via reinforcement learning. For safety, we develop Belief Control Barrier Functions (BCBFs) that leverage conformal prediction to provide probabilistic safety guarantees over finite horizons. The resulting control synthesis reduces to lightweight quadratic programs solvable in real time, even for non-Gaussian belief representations with dimension $>10^4$. Experiments in simulation and on a space-robotics platform demonstrate real-time performance and improved safety and task success compared to state-of-the-art constrained POMDP solvers. |
| 2026-03-11 | [Towards Quantitative Reaction Dynamics of O3](http://arxiv.org/abs/2603.10546v1) | Raidel Martin-Barrios, Abhirami Vijayakumar et al. | The reaction dynamics of O(3P) + O2(3Sigma_g-) collisions in the O3(1A') electronic ground state is characterized on a high-level MRCI+Q/aug-cc-pVQZ potential energy surface represented as a reproducing kernel. For the atom exchange reactions involving the ^{16}O and ^{18}O isotopes as the atomic collision partner, associated with rates k6(T) and k8(T), respectively, a negative temperature-dependence of k(T), consistent with experiments was found. The absolute rates typically underestimate measured rates by 50 percent, depending on the experiment considered. For the ratio R(T) = k8(T)/k6(T), the measured T-dependence was found, including a cusp at lower temperatures. The differences between experiments and computations are primarily due to neglect of quantum effects, primarily zero-point effects. For the atomization reaction, leading to 3O(3P), the rates is lower by approximately one order of magnitude compared with experiments, which is a clear improvement over simulations using previous potential energy surfaces computed with smaller basis sets. Non-adiabatic effects are deemed minor for the atom exchange reactions. |
| 2026-03-11 | [IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs](http://arxiv.org/abs/2603.10521v1) | Chuan Guo, Juan Felipe Ceron Uribe et al. | Instruction hierarchy (IH) defines how LLMs prioritize system, developer, user, and tool instructions under conflict, providing a concrete, trust-ordered policy for resolving instruction conflicts. IH is key to defending against jailbreaks, system prompt extractions, and agentic prompt injections. However, robust IH behavior is difficult to train: IH failures can be confounded with instruction-following failures, conflicts can be nuanced, and models can learn shortcuts such as overrefusing. We introduce IH-Challenge, a reinforcement learning training dataset, to address these difficulties. Fine-tuning GPT-5-Mini on IH-Challenge with online adversarial example generation improves IH robustness by +10.0% on average across 16 in-distribution, out-of-distribution, and human red-teaming benchmarks (84.1% to 94.1%), reduces unsafe behavior from 6.6% to 0.7% while improving helpfulness on general safety evaluations, and saturates an internal static agentic prompt injection evaluation, with minimal capability regression. We release the IH-Challenge dataset (https://huggingface.co/datasets/openai/ih-challenge) to support future research on robust instruction hierarchy. |
| 2026-03-11 | [Naïve Exposure of Generative AI Capabilities Undermines Deepfake Detection](http://arxiv.org/abs/2603.10504v1) | Sunpill Kim, Chanwoo Hwang et al. | Generative AI systems increasingly expose powerful reasoning and image refinement capabilities through user-facing chatbot interfaces. In this work, we show that the naïve exposure of such capabilities fundamentally undermines modern deepfake detectors. Rather than proposing a new image manipulation technique, we study a realistic and already-deployed usage scenario in which an adversary uses only benign, policy-compliant prompts and commercial generative AI systems. We demonstrate that state-of-the-art deepfake detection methods fail under semantic-preserving image refinement. Specifically, we show that generative AI systems articulate explicit authenticity criteria and inadvertently externalize them through unrestricted reasoning, enabling their direct reuse as refinement objectives. As a result, refined images simultaneously evade detection, preserve identity as verified by commercial face recognition APIs, and exhibit substantially higher perceptual quality. Importantly, we find that widely accessible commercial chatbot services pose a significantly greater security risk than open-source models, as their superior realism, semantic controllability, and low-barrier interfaces enable effective evasion by non-expert users. Our findings reveal a structural mismatch between the threat models assumed by current detection frameworks and the actual capabilities of real-world generative AI. While detection baselines are largely shaped by prior benchmarks, deployed systems expose unrestricted authenticity reasoning and refinement despite stringent safety controls in other domains. |

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



