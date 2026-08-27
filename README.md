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
| 2026-08-26 | [A Self-Evolving Multi-Agent Framework Defense against LLM Jailbreak Attacks](http://arxiv.org/abs/2608.26008v1) | Tongyan Hu, Bryan Hooi | Large language models (LLMs) remain vulnerable to jailbreak attacks that exploit techniques such as role-playing, obfuscation, code transformation, and multi-step indirection to elicit harmful outputs. As jailbreak strategies keep emerging, defenses have proliferated in an ongoing cat-and-mouse game, yet most remain static: their safety behavior is fixed at deployment, so they cannot accumulate defensive experience or adapt to unseen strategies. We propose a self-evolving test-time defense built around a persistent, cross-interaction rule memory: when an attack succeeds, the framework abstracts that failure into a method-level rule capturing the structural attack wrapper rather than the harmful topic, and reuses it against future inputs. Because rules are method-level, one induced rule generalizes across an entire attack family, and the label space expands as novel wrappers appear. The mechanism operates entirely through external memory and prompting, with no parameter updates, and applies to both open-weight and black-box API models. We realize it as four cooperating modules, but the contribution is the memory-based adaptation mechanism, not the module decomposition. Across four black-box jailbreak families and multiple models, our method substantially reduces attack success rates while preserving benign utility, remains robust under an adaptive composite-wrapper attack, and does not increase over-refusal as the memory grows. |
| 2026-08-26 | [DESCENT: Directed Edge Scene Encoding for Airport Surface Movement Prediction](http://arxiv.org/abs/2608.26002v1) | Alexander Prutsch, David Schinagl et al. | Advanced automation is a key technology for enhancing the safety of ground operations amidst the increasing density of commercial air traffic. While motion forecasting is a well-studied task in autonomous driving, its application to airport surface movements remains underexplored. To enable efficient and accurate prediction in this domain, we propose DESCENT, a transformer-based architecture designed to handle heterogeneous dynamics and strict topological constraints. Our approach features a Potential Reachable Set (PRS) context sampling mechanism that adaptively collects airfield environment context across diverse operational phases. Combined with a detection transformer-based decoder, DESCENT generates accurate trajectory forecasts. Extensive evaluations on the Amelia-10 benchmark demonstrate significant performance improvements over state-of-the-art baselines. These gains are especially pronounced in safety-critical scenarios, where our domain-aware sampling provides critical long-horizon context necessary for safe navigation. |
| 2026-08-26 | [The Symmetric Pair Matching Design: A Self-Controlled Method with Automatic Adjustment for Time Effects](http://arxiv.org/abs/2608.25979v1) | Robin Denz, Filippo Saatkamp et al. | Self-controlled study designs eliminate confounding by individual-level characteristics that remain constant during the observation time and are thus widely used in pharmacoepidemiology and vaccine safety research. However, existing methods remain vulnerable to time effects, including temporal trends and seasonality in the exposure or outcome, unless these are explicitly modeled or controlled through the study design. We introduce the symmetric pair matching design (SPM), a novel self-controlled method that combines design-based adjustment for time effects with automatic control of time-invariant confounding. Unlike previous design-based approaches that account for time effects, SPM uses observation time both before and after event occurrence, thereby retaining a larger proportion of the available information. We derive the theoretical properties of the method and evaluate its finite-sample performance through simulations. In the simulations, SPM produced unbiased estimates in the presence of temporal trends in both the outcome and exposure, while maintaining greater or comparable statistical efficiency than existing approaches. SPM extends the family of self-controlled methods by providing design-based control of time effects without requiring explicit modeling thereof. By combining robustness to temporal confounding with efficient use of observation time, SPM offers a practical alternative for observational studies with transient exposures. An accompanying R package is provided to facilitate its usage in applied research. |
| 2026-08-26 | [Formal, Executable and Explainable Runtime Monitoring of Spoken Air Traffic Control Operational Procedures](http://arxiv.org/abs/2608.25926v1) | Roberto Luvini, Giacomo Longo et al. | Air traffic control procedures are executed through spoken exchanges between controllers and pilots. These interactions are essential to the safety of air transportation: failures in their execution can create severe operational hazards, as evidenced by past fatal accidents. Assessing whether an instruction has been followed requires relating what was said to the aircraft concerned, its state, and the obligations that pilots must meet. We present a runtime verification framework that monitors such procedures by checking controller-pilot exchanges, surveillance data, and onboard observations. The framework parses radio communications into events linked to the entities they concern and merges them with surveillance and onboard observations into a time-stamped trace. The ICAO-derived obligations as formalized as temporal formulas with explicit time bounds and evaluated over execution traces. Every violation is reported along with the breached obligations and the observations that support the verdict. With real traffic, the complete pipeline reaches an F1 of 0.85 against blind human-annotated violations; in 1,495 synthetic situations derived from two public corpora, the monitor logic returns the expected verdict in every case. In two historical accidents reconstructed from official investigation reports, the monitor identifies the same procedural deviations documented by the investigators. |
| 2026-08-26 | [A Hybrid Security Framework for Mini-Programs: Visual UI Compliance and Network Risk Assessment](http://arxiv.org/abs/2608.25877v1) | Panpan Shen, Lei Xie et al. | With the continuous development of the WeChat ecosystem, WeChat Mini Programs, due to their advantages of not requiring installation, using little memory, and being ready to use instantly, have seen a surge in user numbers and have now become an indispensable service carrier in mobile internet. However, as Mini Programs rapidly became popular, issues regarding the compliance of their interface interaction design and the safety of operational behavior have become increasingly apparent. Many Mini Programs have problems such as clickable buttons and icons not being standard in size, or ad pop-ups and payment entrances being placed in a way that is easy to misclick. The close or cancel buttons are often too small or hidden, making it easy to accidentally click on ads or payment content, and difficult to accurately click the cancel button. This can result in involuntary payments or being redirected to illegal pages, causing unnecessary financial losses and seriously harming users' property security and legal rights. To address the above issues, this article develops a detection program to check the position and size of various icons and buttons in Mini Programs, and analyze whether redirected links fall within a safe range. YOLOv8 is used to identify various buttons in images, displaying the corresponding icon and its data based on the mouse click position. Violations are flagged and recorded. At the same time, mitmproxy is used to capture relevant data requests generated during clicks, analyzing the safety of redirections, and presenting key information for user observation. |
| 2026-08-26 | [SkillShield: Prompt-Space Security Skills for LLM Coding Agents](http://arxiv.org/abs/2608.25817v1) | Xiaodong Wu, Zhimin Zhao et al. | A coding agent edits files and executes shell commands with its developer's privileges, allowing malicious requests to translate directly into harmful actions or functional malware. Existing defenses have complementary limitations: weight-level alignment is unavailable to API-only deployers, whereas input filters and execution-boundary monitors require auxiliary classification or checking components along the agent's trajectory. We therefore introduce SkillShield, a system-prompt defense that synthesizes security skills offline from known attacks or recorded agent failures. These skills are injected into the system prompt at session start and remain active throughout the tool-use loop. Unlike a reference monitor, they protect the system by defining the security policies the model should follow during execution. Due to the limited system-prompt space, we examine three fixed-budget provisioning scopes: all-classes, with one skill covering all threat classes, per-bundle, with one skill targeting a related subset, and per-class, with one skill dedicated to a single known class and used as the upper-bound reference. None requires runtime request classification or routing. Across six large language models on RedCode, the default all-classes skill reduces malware-generation severity from 3.37 to 0.58 and achieves a 43.6% execution attack success rate, comparable to Llama Guard 3's 42.7% without its separate 8B classifier. The per-bundle and class-fixed per-class settings further reduce this rate to 36.2% and 14.5%, respectively. Under two non-adaptive jailbreak families, SkillShield continues to outperform all baselines on malware generation. Across 731 benign task descriptions, SkillShield yields a mean safety-refusal rate of 0.14%. These results demonstrate the potential of prompt-space security skills to prevent harmful actions and malware generation for LLM coding agents. |
| 2026-08-26 | [Why Does Graph Learning Fail to Fully Benefit from a Text Teacher?](http://arxiv.org/abs/2608.25741v1) | Fumiaki Kimino, Ryoma Sato | Graph neural networks (GNNs) are widely used to represent complex interactions and relationships among entities. We investigate a multimodal model that combines two complementary ideas: a self-supervised method that enables a GNN encoder pretrained on one dataset to operate directly on another dataset with a different node-feature dimensionality, without rebuilding the model or realigning the data; and an alternating optimization method that updates a language-model module in an E-step and a GNN module in an M-step, rather than jointly training a large language model and a GNN end to end on a large graph. Despite expectations, the combined model did not sufficiently improve predictive performance. We identify six factors: (1) an external anchor in the E-step has a strength-safety trade-off: a weak anchor has little effect, whereas an overly strong anchor can damage the graph representation; (2) the knowledge of the E-step teacher is not injected directly into the GCN embedding Z; (3) the representation space constructed in the M-step is not optimized for the same objective as the E-step teacher space, resulting in a compromise representation for target classification; (4) GCN propagation averages a node's own textual information with information from its neighbors; (5) cosine alignment does not guarantee axes that are discriminative for classification, so stronger geometric alignment with the E-step text anchor need not sufficiently improve the target decision boundary or classification performance; and (6) the force that preserves the source-side self-supervised geometry in the M-step conflicts with the force that moves the representation toward the E-step teacher. We support these observations through a staged set of experiments that varies the influence of the E-step. |
| 2026-08-26 | [Reassembling Distributed Risk: Trajectory-Conditioned Action Generation for Multi-Turn Agent Safety](http://arxiv.org/abs/2608.25711v1) | Yanbo Dai, Zhenlan Ji et al. | Tool-using LLM agents extend security risks beyond generated text to actions that affect external systems. Under multi-turn decomposition attacks, a harmful objective can be distributed across individually plausible requests and tool calls, becoming apparent only from the accumulated trajectory. Existing defenses either rely on auxiliary online reasoning to recover long-horizon security evidence or assess actions after generation, often incurring additional inference cost or depending on runtime-specific action representations.   We propose \emph{Reassembling Distributed Risk} (ReDiR), a generation-time defense that conditions action generation on trajectory-level security evidence. Before each action, ReDiR compresses the current trajectory into a compact latent safety representation and injects it into the frozen base model. The representation is learned through same-model, cross-view supervision, where safe behavior from an explicit task view provides supervision for recovering distributed safety evidence from the original multi-turn trajectory. This design enables ReDiR to integrate cross-turn security information directly within the generation process without relying on a separate action-level safety module. We evaluate ReDiR on two agent-safety benchmarks across three model families and eight held-out tool domains. ReDiR reduces attack success rates to below 8\%, transfers to unseen tool domains, and preserves benign fidelity with low computational overhead. |
| 2026-08-26 | [LMSM: LLM Security Framework Inspired by Linux Security Modules](http://arxiv.org/abs/2608.25697v1) | XiuYu Zhang, Bonan Ruan et al. | Large language models (LLMs) are increasingly deployed with layered defenses, yet malicious prompts can still bypass them. Interpretability methods can expose model-internal signals along the generation path that could inform enforcement, but these signals are not security controls by themselves. Deployments that adapt them for safety typically couple each signal to its own calibration, policy logic, and intervention code, so each new artifact creates integration work instead of strengthening a shared defense. We present Language Model Security Modules (LMSM), a security framework that adapts the separation behind Linux Security Modules (LSM) to LLM serving. In LMSM, a selected security backend exposes calibrated evidence, a versioned policy evaluates active rules over trusted per-request context, and a separate gate authorizes buffered output release. This design separates mediation correctness from policy effectiveness, and it allows backend, rule, or schedule changes without rebuilding request handling or enforcement. Our prototype shows the separation working in practice: with Hugging Face Transformers and continuously batched vLLM, the same substrate hosts artifact-backed sparse autoencoder (SAE) and transcoder deployments and task-fitted dense probes, preserves request-specific decisions under scheduler churn, and selectively enforces and composes multiple rules per request. On Qwen3-4B, LMSM-Checkpoint reduces HarmBench attack success rate from 39.20% to 3.32%, with XSTest false refusals rising from 2.40% to 4.40%, while retaining 98.14% of the throughput of a matched serving path that performs no monitoring work at 32 active sequences. LMSM gives advances in interpretability and model-internal analysis a common path to runtime enforcement. |
| 2026-08-26 | [EgoNav: Bridging Learned Waypoints and Geometry-Aware Local Control for Robust Indoor Navigation](http://arxiv.org/abs/2608.25642v1) | Jing Wang, Shiqi Zhao et al. | Image-goal navigation using lightweight topological maps is a practical paradigm for indoor robot deployment: the map requires only geotagged images, and localization relies on visual matching rather than precise pose estimation. However, learned waypoint predictors can produce targets that violate geometric constraints or deviate from the global path. Executing these waypoints safely further requires a local planner capable of collision avoidance, yet existing systems either lack one or rely on fixed parameters that cannot adapt to confined spaces. To address these limitations while retaining the navigational intuition of the learned predictor, we present EgoNav, a hierarchical system that implements this idea by generating candidates from semantically segmented traversable regions and scoring them alongside the learned waypoint for geometric safety, directional coherence, and fidelity to the learned prior. An adaptive local path planner then executes the refined waypoint with parameters modulated based on the refinement outcome. Experiments in Habitat-sim and on a physical humanoid robot show that EgoNav consistently outperforms contemporary baselines in both success rate and path efficiency. |
| 2026-08-26 | [SMART: MLLM-guided Temporal Alignment for Unifying Sign Language Recognition and Spotting](http://arxiv.org/abs/2608.25493v1) | Eunjee Choi, JungHoon Sung et al. | Continuous sign language recognition (CSLR) aims to recognize gloss sequences from unsegmented sign videos under weak sequence-level supervision. However, existing methods rely on sentence-level gloss annotations, providing limited temporal and semantic guidance for fine-grained representation learning. Conventional video-text alignment also requires large batch sizes, making it inefficient for memory-intensive sign language video training. In this work, we propose SMART, an MLLM-guided temporal alignment framework for joint sign recognition and spotting. SMART uses MLLMgenerated motion descriptions as auxiliary semantic cues and performs stable videotext alignment under small-batch training. To improve temporal representation learning, we introduce a Multi-Scale Temporal Adapter that models temporal interactions during transformer encoding. For dense temporal localization, SMART incorporates CSFormer, a CSLR-guided spotting module that injects recognition-derived gloss evidence into a boundary-aware spotting network. This unified framework enables CSLR features to benefit spotting, while spotting supervision complements weak CTC-based recognition. Experiments on four sign language benchmarks, including PHOENIX14-T, CSL-Daily, Large-scale KSL, and Disaster and Safety KSL datasets, demonstrate the effectiveness of SMART across both recognition and spotting tasks. |
| 2026-08-26 | [MMJailBench: A Factorized Benchmark for Disentangling Multimodal Jailbreak Vulnerabilities](http://arxiv.org/abs/2608.25490v1) | Tianshi Wang, Jingsong Wang et al. | Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet how different factors shape their jailbreak vulnerabilities remains poorly understood. Existing benchmarks often couple harmful intent, prompt framing, visual semantics, and instruction carrier within individual jailbreak instances, obscuring the specific sources of observed vulnerabilities. To address this limitation, we introduce MMJailBench, a factorized benchmark that systematically varies and combines these factors under controlled configurations, enabling fine-grained comparison and factor-level attribution. Large-scale evaluations across 16 open-weight and proprietary MLLMs reveal highly heterogeneous and model-dependent vulnerability profiles. Jailbreak vulnerability varies markedly across harm domains, exposing uneven coverage in current multimodal safety alignment. Prompt framing emerges as the dominant source of variation, task-relevant visual semantics systematically increase jailbreak susceptibility with authority-like cues exposing particularly pronounced vulnerabilities, and visually rendered instructions do not consistently increase jailbreak susceptibility relative to direct textual instructions. To further investigate the risks introduced by multimodal context, we conduct diagnostic analyses on a representative open-weight model and identify vulnerability-associated patterns in internal representations and cross-modal interactions. Finally, we develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-efficient multimodal jailbreak auditing. |
| 2026-08-26 | [AERIS: Offline Policy Improvement for Multi-UAV Integrated Sensing and Communication](http://arxiv.org/abs/2608.25477v1) | Ziyuan Wang, Yifan Sui et al. | Unmanned aerial vehicle (UAV)-enabled integrated sensing and communication (ISAC) is a promising 6G paradigm, but dynamic multi-UAV ISAC control must jointly balance communication quality, sensing reliability, and flight safety under stochastic mobility. Existing optimization methods often require repeated global non-convex solving, while online reinforcement learning (RL) depends on risky trial-and-error flights that may cause sensing loss or collision-risk events.   This paper proposes AERIS, an offline policy improvement framework for multi-UAV ISAC. AERIS learns from fixed flight logs under centralized training and decentralized execution, so each UAV acts from local histories while training uses logged global information to assess team-level effects. We further design STAR-CRDT, an offline multi-agent RL algorithm that performs support-aware local action rectification and distills only trusted improvements into the decentralized actor. We prove an offline-support policy improvement guarantee. Experiments show that STAR-CRDT improves the main ISAC objective return by 29.3% over the strongest baseline. It further improves communication sum rate, sensing pass rate, and sensing margin by 3.4%, 4.8%, and 69.1%, while reducing collision-risk events by 54.2%. On unseen real-road maps built from OpenStreetMap data, STAR-CRDT still obtains the best return. |
| 2026-08-26 | [Towards safe and optimal flight: Viability Kernel MPC for Fully Actuated Multirotor](http://arxiv.org/abs/2608.25459v1) | Massimiliano Bertoni, Alberto Piccina et al. | Industrial aerial robotics demands safety guarantees for navigation in unstructured environments while optimizing performance and computational efficiency. This paper presents a method for generating safe pose trajectories for fully actuated multirotors within a Model Predictive Control (MPC) framework, leveraging both viability theory and data-driven methods. Obstacle avoidance is enforced through dynamically computed axis-aligned bounding boxes, providing formal safety guarantees without exhaustive offline reachability analysis. Numerical simulations on a fully actuated tilted hexarotor validate the approach, demonstrating successful navigation in cluttered environments with real-time computational performance. |
| 2026-08-26 | [Continuous Computational Social Choice: A Case Study in Bribery](http://arxiv.org/abs/2608.25444v1) | Martin Koutecký, Nikolaos Melissinos et al. | Computational social choice seeks algorithmic answers to questions about preference aggregation, safety of elections, robustness of outcomes, stability, etc. It overwhelmingly models societies as composed of discrete agents.   We propose to study computational social choice problems in a society continuum} setting, where a society is modeled as a distribution of infinitely many infinitesimal agents of different types. An analogous approach has been very useful in physics (it is the basis of statistical mechanics), economics (mean field games), and other fields.   As an initial case study, we focus on election attacks (bribery and control), which have been extensively studied in the discrete setting. We show that a broad class of standard election attacks becomes polynomial-time solvable in the society continuum. The class contains problems that are NP-hard discretely, among them Borda- and Bucklin-CCDV and unit-cost Borda-SWAP BRIBERY. Furthermore, we give polynomial-time algorithms for $k$-Approval-SWAP BRIBERY when $k$ is constant for general costs, and when $k$ varies and the cost function is additively separable. The latter result contrasts with the discrete problem, which we prove NP-complete for additively separable costs and every fixed $k\ge 2$. In contrast, we prove that Borda-SWAP BRIBERY and $k$-Approval-SWAP BRIBERY, both with general costs, remain computationally hard in the society continuum.   To obtain these results, we use both continuous and discrete optimization techniques, such as the Configuration LP framework and dynamic programming. Of particular note is the technique underlying our hardness proofs, which shows how to ''reverse the flow of hardness'' between LP formulations and pricing problems. |
| 2026-08-26 | [E2-Conditioned Finite-Horizon Effective Capacity for Public-Safety MCX over Shared O-RAN](http://arxiv.org/abs/2608.25442v1) | Jingqing Wang, Wenchi Cheng | Supporting public-safety Mission Critical Services (MCX) over a shared Open radio access network (O-RAN) requires service assurance over finite incident horizons, while ordinary mobile traffic competes for the same resources and heterogeneous E2 domains expose different observations, control actions, and actuation latencies. Existing RAN key performance indicators are retrospective, whereas conventional effective capacity characterizes an asymptotic stationary regime and therefore suppresses both the initial E2-observed condition and the finite selection-to-actuation transient. To solve this problem, in this paper we develop an E2-conditioned finite-horizon effective capacity (FH-EC) framework for public-safety MCX over shared O-RAN. Specifically, we first establish a finite-horizon O-RAN MCX-based service model that incorporates E2-observed network states, control actuation latency, and correlation-aware connectivity diversity. Based on this model, we derive an FH-EC formulation that characterizes the executable service capability within a finite mission horizon. Furthermore, we transform FH-EC into confidence-calibrated capability profiles and develop an MCX orchestration framework with contract certification, shared-resource protection, and FH-EC-driven profile selection at the Near-RT RAN Intelligent Controller (RIC). Coupled MATLAB/ns-3 evaluations demonstrate the predicted short-horizon state and actuation effects and show that adaptive connectivity selection improves MCX supportability under O-DU degradation while satisfying the configured multi-QoS and non-MCX protection requirements. |
| 2026-08-26 | [SUPER ODOMETRY 2.0: Resilient Odometry via Hierarchical Adaptation](http://arxiv.org/abs/2608.25427v1) | Shibo Zhao, Sifan Zhou et al. | Resilient and robust odometry is crucial for autonomous systems operating in complex and dynamic environments. Existing odometry systems often struggle with severe sensory degradations and extreme conditions such as smoke, sandstorms, snow, or low-light conditions, threatening both the safety and functionality of robots. To address these challenges, we present Super Odometry, a sensor fusion framework that dynamically adapts to varying levels of environmental degradation. Super Odometry employs a hierarchical structure to integrate four core modules from lower-level to higher-level adaptability including adaptive feature selection, adaptive state direction selection, adaptive engine selection, and a novel learning- based inertial odometry. The inertial odometry, trained on over 100 hours of heterogeneous robotic platforms, captures comprehensive motion dynamics. Super Odometry elevates the inertial measurement unit (IMU) to equal importance with camera and LiDAR within the sensor fusion framework, providing a reliable fallback when exteroceptive sensors fail. Super Odometry has been validated across 200 kilometers and 800 operational hours on a fleet of aerial, wheeled, and legged robots, under diverse sensor configurations, environmental degradation, and aggressive motion profiles. It marks an important step towards safe and long-term robotic autonomy in all-degraded environments. |
| 2026-08-26 | [Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks](http://arxiv.org/abs/2608.25390v1) | Andrey Labunets | Refusal training protects AI models from jailbreaks by training models to decline unsafe queries, reducing the risk of misuse. Recent work finds that refusal behavior in aligned language models can be mediated by a single activation direction or a low-dimensional refusal subspace shared across harmful prompts: ablating those directions suppresses refusals while largely preserves other model capabilities. Yet it remains unclear why safety-critical features in a wide range of models emerge and concentrated, low-dimensional structure. In a case study of OLMo-2-0425-1B-Instruct we find that the refusal geometry reflects refusal training: activation updates resulting from refusal-completion first-token losses explain the resulting refusal direction and refusal subspace. We study refusal directions through the training dynamics across refusal datasets and reveal that their brittleness is associated with repetitive refusal starts, which in turn is linked to concentration of gradients and refusal features in a low-dimensional subspace. Across frozen-model analyses and controlled synthetic fine-tuning, we find evidence of a hardening lever: diverse refusal starts can raise stable ranks of gradients and activation changes, making refusals harder to remove with a vector ablation attack. |
| 2026-08-26 | [HRGuard: Gating Relationship Manipulation in Multi-Turn Agentic AI Conversations](http://arxiv.org/abs/2608.25340v1) | Pei-Sze Tan, Tasuku Igarashi et al. | Agentic AI assistants are increasingly used in everyday life. However, they may also be misused to support harmful manipulation in interpersonal relationships. This problem is role-sensitive. Requests from users who seek to manipulate others should be blocked. Users who seek protection from manipulation should instead receive supportive guidance. We study agentic relationship harm, which describes harm to human-human relationships that is mediated or assisted by AI agents. In multi-turn settings, individually plausible actions may combine into a harmful workflow. We introduce a benchmark of 1,000 five-turn conversations. It covers both attacker-side and victim-side scenarios. It also includes direct and adversarially paraphrased variants. We further propose HRGuard. It includes an online pre-generation gate and a turn-level post-generation gate. The post-generation gate maintains a decayed cumulative risk state and interrupts emerging manipulative workflows. Across eight generation models, HRGuard reduces harmful compliance while preserving victim-side protective guidance. It also outperforms a generic safety prompt and three general-purpose guard models. Independent-judge evaluation supports the main findings. Under our evaluation protocol, the tested generic prompt and general-purpose guards leave substantial residual risk, motivating turn-aware relationship-specific evaluation. |
| 2026-08-26 | [Scalable Tube-Tightened Multi-Agent Safety via Certified Constraint Reduction](http://arxiv.org/abs/2608.25323v1) | Armel Koulong | This paper develops a certified constraint-reduction method for distributed model predictive control with tube-tightened exponential control barrier functions (eCBFs) in multi-agent systems. At each prediction stage, pairwise agent--agent and agent--obstacle eCBF conditions define halfspaces in the local control space. Rather than enforcing all such halfspaces, a geometry-adaptive subset is retained and a Farkas certificate verifies that the reduced admissible set is contained in the full tightened set. For planar inputs, cone coverage is characterized through the largest angular gap: two extreme directions suffice in the strict half-plane regime, while other geometries initialize with three retained constraints and escalate only when certification fails. Conic multipliers and nominal-aware offsets are obtained in closed form, without an auxiliary optimization, and the resulting construction preserves any nominal control already admissible for the full tightened set. Consequently, the reduced controller inherits the robust safety guarantee of the underlying tube-eCBF formulation. In a ten-follower, four-obstacle study, the method retained fewer safety constraints on average, reproduced the full filter's nominal accept/reject decisions with no true safety violations, and achieved increasing computational gains as the constraint count and prediction horizon grew. |

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



