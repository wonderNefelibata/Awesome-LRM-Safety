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
| 2026-10-09 | [Can Language Models Play the Wikipedia Game?](http://arxiv.org/abs/2610.13948v1) | Alexander Hickey, Joanna Knight et al. | Language models increasingly support or automate information retrieval from document collections too large to index. This task is difficult when it requires specialist linguistic or scientific knowledge. We used the Wikipedia Game to test language models on an analogous information retrieval task. In the game, players must reach a target article using only article links. We compared sentence embedding models and LLMs across five Wikipedia languages and five academic-literature datasets. We construct very large local graphs of Wikipedia pages and of paper citations using OpenAlex and provide code to build and navigate them. This allows the study of language models' navigation capabilities without relying on third-party APIs. We compared the models with human and TF-IDF benchmarks. The Wikipedia Game format enables robust comparisons across techniques, languages, and domains. No model significantly outperformed humans, although the best models (GPT-o3 mini, GPT-4o, QwQ:32B, and DeepSeek-R1:70B) achieved comparable success. Performance and rankings varied across languages and topics, with implications for practitioners utilising these techniques. |
| 2026-10-09 | [Noise Space Learning for VLA Adaptation](http://arxiv.org/abs/2610.13943v1) | Junjie Lu, Chuheng Zhang et al. | Pretrained vision-language-action (VLA) models encode broad manipulation priors, yet adapting them to specific tasks requires balancing limited robot interaction, computational resources, and safety constraints. Noise space learning (NSL) adapts flow-matching-based VLAs by learning a context-conditioned distribution over noise while keeping the pretrained VLA fixed. Building on prior noise-steering methods, we formulate NSL as an interface for VLA adaptation that accommodates different learning algorithms and instantiate it with PPO, SAC, and AWR. Theoretically, we provide an information-theoretic analysis of how NSL enables efficient adaptation. Empirically, we study these NSL algorithms on LIBERO, RoboTwin, and four real-robot manipulation tasks under interaction, computation, and execution constraints. Across these settings, we develop a suite of effective NSL variants and derive practical recommendations for choosing among them under different resource and deployment regimes. |
| 2026-10-09 | [Tracking the Invisible: Self-Supervised Wheel Track Detection with Thermal-RGB Fusion](http://arxiv.org/abs/2610.13872v1) | Yiming Yang, Jeremy P. Bos | Accurate detection of drivable area is essential for autonomous driving and advanced driver assistance systems. While state-of-the-art methods focus on general traversability estimation, they prove inadequate for snow-covered roads where following existing wheel tracks offers crucial advantages: improved traction, predictable vehicle dynamics, and verified path stability. To address this, we propose a low-cost, self-supervised approach detecting safety-critical wheel tracks using synchronized RGB-I (RGB + Long-Wave Infrared) imagery. Our method automatically generates training labels by projecting future wheel trajectories from recorded GNSS data. A neural network is designed to identify wheel tracks with the 4-channel RGB-I images from challenging, white-out road surfaces. Experimental results demonstrate the effectiveness of our approach in visually monotonous and featureless adverse environment, offering a robust and cost-effective alternative for winter autonomy. The newly introduced ThermalTrack dataset is available at: https://digitalcommons.mtu.edu/thermaltrack/. |
| 2026-10-09 | [SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2610.13737v1) | Hongrui Zheng, Aditya Singh et al. | Cooperative multi-agent tasks often impose constraints on a shared objective or collective safety. A team level violation cost function can measure this constraint outcome, but does not distinguish which agents' actions most strongly influence that violation. We present SafeDiSTL, a constrained multi-agent reinforcement learning framework that allows for credit assignment even with joint constraint measurements. We use Signal Temporal Logic (STL) robustness to measure episode-level violation through running-maximum cost increments. Smooth full-episode robustness and differentiable dynamics provide action sensitivities, which allocate each cost increment across agents at the same decision time. These sensitivities account for delayed effects when weighting agents' shares of each cost increment. We study the framework in cooperative payload transport, connected navigation, and networked traffic control tasks. SafeDiSTL achieves 1.6-5.1x the safe-success rate in transport, improves navigation safe-success by 11.1%, and delivers 50-68% more throughput in traffic when compared to constrained baselines. Results demonstrate a strong improvement of the learned policy in dealing with shared constraint while retaining the performance of unconstrained policies. |
| 2026-10-09 | [APEX-Harm: Evaluating Agent Safety in Shared Professional Workspaces](http://arxiv.org/abs/2610.13623v1) | Ziyi Yang, Adel Bibi et al. | Professional agents increasingly work through command-line interfaces and multiple tools over shared workspaces containing files, messages, and business applications. These environments create safety risks that are difficult to capture with attacks placed at a single, known read location. We introduce APEX-Harm, a benchmark for evaluating prompt and script injection against long-horizon agents performing finance, consulting, and legal tasks in large workspaces. Each task contains 115 to 328 files, together with email, calendar, chat, document, spreadsheet, PDF, and code-execution tools exposed through MCP servers. The benchmark includes six attack categories covering static and dynamic injections, task-specific script replacement, and attacker-provided generic scripts, and evaluates attack success, exposure, and task performance. We evaluate 12 frontier models on 180 task instances per model. Opus 5 and Sonnet 5 have the lowest overall attack success rates (8% and 20%), while task-specific script replacement and attacker-provided generic script attacks remain substantially more effective, averaging 81-84% and 73% attack success, respectively. Gemini 3.8 Flash achieves the highest mean task score (0.877) while still exhibiting a 33% attack success rate. These results show that completing the requested task is not sufficient evidence of safe behavior: an agent can produce a correct deliverable while also performing an unauthorized action in the workspace. |
| 2026-10-09 | [Limit-sure Winning in Team Games without Shared Randomness](http://arxiv.org/abs/2610.13569v1) | Léonard Brice, Thomas A. Henzinger et al. | We study concurrent stochastic games in which a team of agents plays against an adversarial environment. If the agents could access a shared source of randomness, then these games reduce to two-player games. Despite being common in the literature on coalition games, this assumption does not accurately model real-world systems. In this paper, we consider the setting where the agents only have access to individual sources of randomness, and must satisfy a given reachability or safety objective. The limit-sure problem asks whether the team can ensure that the objective is satisfied with probability at least $1-λ$ for every $λ>0$. For reachability objectives, we show that this problem is $\mathsf{NP}$-complete. For safety objectives, we show that limit-sure winning coincides with almost-sure winning, which is known to lie in $\mathsf{P}$. As a corollary of these results, we conclude that model checking of the qualitative fragment of the logic Individually Randomised ATL -- introduced in the recent work of Brice et al. -- is $\mathsf{P}^{\mathsf{NP}}$-complete. |
| 2026-10-09 | [A Local Agentic AI Framework to Generate Edge Case Scenarios for Safety Assurance](http://arxiv.org/abs/2610.13479v1) | Victoria J. Hodge, Colin Paterson et al. | Generating edge case scenarios is crucial for training, testing and safety assuring autonomous systems. This identifies hazardous failures to increase system reliability and robustness, ensures the system can still function safely on the boundaries of its operational design domain and learns the expected system behaviour for these edge cases. The rapid advancement in the generation and analysis capabilities of language models (LMs) has expanded their applicability and recent work has investigated LM scenario generation. However, many recent approaches rely on human-in-the-loop or rule-based strategies. This restricts the LM's ability to generate diverse and edge case scenarios; the long tail of the distribution. Our work uses local LMs in a laptop-based agentic AI framework for adaptive scenario generation. This is cheap, fully-controlled, ensures data privacy and security. It uses think, act, observe, update, and continue loops with lightweight JSON encoding to generate scenarios. We use the system requirements, system description and environment description as knowledge bases in contextual prompts for an LM to ensure scenario realism. A second, independent LM critiques each scenario using domain specific criteria so it can be refined. The refined scenarios are run in simulation and evaluation metrics collected. These metric scores are analysed and similar scenarios are created for edge cases (exploitation) or novel scenarios are generated to explore the scenario space. Post-hoc clustering using metric scores and thematic-type analysis of the rich-text scenario descriptions produced by the LM pinpoints functional gaps and their triggering conditions. This creates traceable evidence for the safety assurance case. Our work highlights the potential of a flexible and adaptive local agentic AI framework for automating edge case generation and producing traceable safety assurance evidence. |
| 2026-10-09 | [HoldLight: Making Indoor Climbing More Accessible for Blind and Low-Vision Climbers through Layered Route Preview and Real-Time Next-Hold Guidance](http://arxiv.org/abs/2610.13445v1) | Chenyu Du, Junyan He et al. | Indoor climbing creates access barriers for blind and low-vision (BLV) climbers because route understanding and in-climb next-hold localization are typically visually mediated and socially dependent. We present HoldLight, a workflow-centered accessibility system combining layered pre-climb route preview, concise real-time next-hold cueing, and explicit fallback under uncertainty. We grounded the design through formative inquiry with BLV climbers, refined the interaction via pilot testing, and conducted a controlled within-subjects evaluation comparing HoldLight with a controlled on-demand human-support condition derived from current support practices. Findings provide initial evidence that structured support can improve pre-climb route understanding, support recovery from next-hold uncertainty, and reduce immediate human re-entry for routine route-information clarification under controlled conditions. HoldLight contributes a bounded accessibility workflow, demonstrating where structured route-information is helpful, where explicit fallback is necessary, and how technical assistance can integrate with retained human safety oversight. |
| 2026-10-09 | [EndoRSI: Physical-Evidence-Driven Recursive Improvement of Endoscopic Robot Policies](http://arxiv.org/abs/2610.13376v1) | Chi Kit Ng, Yidong Zhang et al. | Artificial intelligence is beginning to automate scientific discovery, yet physical robot-policy research still relies on scientists and engineers to interpret experimental failures and decide how controllers should be revised. We present EndoRSI (Endoscopic Recursive Self-Improvement), a human-governed, self-improving robotic research system in which the embedded EndoForge coding agent iteratively develops and refines endoscopic robot policies using evidence from physical experiments. Physical execution remains isolated within a safety-constrained runtime under human authorisation. In motorised endoscopy, validation evidence first caused the agent to reject an initial action representation, while subsequent physical experiments falsified an offline-supported insertion strategy by revealing hazardous closed-loop behaviour absent from the source data. Failed physical rollouts were then incorporated into policy revision; insertion success subsequently increased from 2/21 in the first generation to 17/21 in the fourth. The frozen final policy retained competence across visual and hardware shifts and was the best performing autonomous policy across six independent ex-vivo porcine specimens. These results show that physical robot interaction can serve as experimental evidence for agent-driven robot policy research. |
| 2026-10-09 | [Towards Safe World Models: Risks, Defenses, and Evaluation](http://arxiv.org/abs/2610.13372v1) | Xiaoma Wang, Litian Zhang et al. | World models support planning, control, learning from imagined experience, and model-mediated evaluation by predicting action-conditioned consequences. Their safety cannot be inferred from predictive accuracy alone: errors in observations, state and memory, dynamics, evaluation, execution, or feedback can propagate into unsafe decisions and persist through updates. This work organizes the literature on world-model safety around a six-stage lifecycle covering data and observations, state and memory, dynamics and imagination, evaluation and planning, action and control, and feedback and update. We synthesize non-adversarial failures, attacks, defenses, evaluation protocols, benchmarks, and deployment conditions, and distinguish evidence directness from outcome level. The current evidence corpus contains 163 cited works. We use the framework to identify where claims are supported, where attribution remains unresolved, and which measurements are needed to connect model-level behavior to closed-loop and operational safety. |
| 2026-10-09 | [FPD-Drive: Future Policy Distillation for End-to-End Autonomous Driving](http://arxiv.org/abs/2610.13232v1) | Li-Heng Chen, Qifeng Chen et al. | End-to-end autonomous driving models are typically trained on real-world driving logs, which record only the trajectory actually executed in each scene. This provides limited supervision for learning alternative valid behaviors. Multi-target supervision can mitigate this limitation, but broader trajectory coverage makes safe candidate selection more challenging. We propose FPD-Drive, an end-to-end planner that distills privileged future knowledge into a visual driving policy to improve both multi-trajectory supervision and safety selection. Specifically, we introduce Future-Privileged Multi-Target Distillation to transfer diverse, high-quality trajectory targets from a teacher with access to recorded future observations to the student. These targets extend supervision beyond the single recorded human trajectory. For candidate selection, we further introduce Safety-Aware Score Correction to correct ranking errors between nearby safe and unsafe candidates and raise underestimated safety scores of safe candidates. At deployment, the student requires neither future observations nor teacher inference. Our base model achieves 94.9 PDMS on NAVSIM v1 navtest and 55.9 EPDMS on NAVSIM v2 navhard. The scaled FPD-Drive-Pro improves these scores to 95.5 PDMS and 58.1 EPDMS, respectively, achieving SOTA performance among the compared methods on both benchmarks. Zero-shot closed-loop evaluation on HUGSIM further yields an HD-Score of 45.1 and RC of 55.3. |
| 2026-10-09 | [Configuration Migration Planning through Decoupled Modeling and Searching](http://arxiv.org/abs/2610.13161v1) | Dan Wang, Peng Zhang et al. | Executing large network configuration changes, called migrations, is slow and error-prone today; operators must manually devise a safe order for updating devices that does not disrupt traffic and meets operational intents (e.g., low total time). We develop Smoothie, a versatile migration planning framework that, unlike existing tools, can support diverse operational intents. Inspired by automated planning, it achieves versatility by decoupling the encoding of network safety and intents from the search for a viable plan. Smoothie uses symbolic simulation for general and scalable safety encoding, after which it can use a range of planners, including existing off-the-shelf planners and a new one we developed for optimized parallel plans. We demonstrate Smoothie's versatility and benchmark its performance using diverse migration scenarios on many topologies. It finds both sequential plans and parallel plans, and the parallel ones compress up to thousands of configuration updates into no more than 3 steps. Its versatility does not come at the expense of performance: finishing most planning tasks in less than 1 hour. |
| 2026-10-09 | [Safety Graph: When Safety Responses Become Attack Surfaces in Cyber-Physical Systems](http://arxiv.org/abs/2610.13086v1) | Rakesh Podder, Philip D. Hopkins et al. | Modern industrial control systems (ICS) are susceptible to cyber attacks triggering safety failures in the operational technology (OT) layer. The traditional way to mitigate any failure in OT is to implement safety responses. However, we show in this work that these responses can open up new attack surfaces, thereby vastly expanding the scope of lateral movement. Existing frameworks model cyber-to-physical escalation but provide no mechanism to reason about how these safety responses to OT-failures can trigger new cascading cyber attacks in the Information Technology (IT) layer. To enable such reasoning, we introduce Safety Graph (SG), a formal framework unifying three paradigms - Attack Graphs (AG), Fault Trees (FT), and Event Trees (ET) - over a shared propositional state space, capturing the full bidirectional IT-OT cascade. We provide a sound and complete compilation of SG to a classical AI planning problem, performing plan search using A* to find cross-layer cascade paths. Through a Web3D modeling and simulation of the Tennessee Eastman chemical process plant, which is a widely used chemical engineering benchmark serving as a standard for testing control, monitoring and fault propagation, we demonstrate that the planning compilation of SG automatically discovers safety-response-induced lateral movements, and identifies the responsible barriers that are unidentifiable under existing models. A complexity and scalability analysis characterizes how the planning compilation of SG scales with model size and cascade depth. To the best of our knowledge, SG is the first framework to formally model IT-OT attacks and enable operators to explore the problem of cyber attacks triggering cascading failures in ICS. |
| 2026-10-09 | [Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs](http://arxiv.org/abs/2610.13078v1) | Zixuan Weng, Haoqi He et al. | Safety-aligned large language models (LLMs) remain vulnerable to jailbreaks and can over-refuse benign requests. Understanding these failures requires distinguishing harmfulness recognition from the computation that writes refusal. We reframe this as a pathway question and make it measurable on residual-stream updates in decoder-only Transformers, separating the harmfulness information a module update exposes from the change it produces in readable refusal evidence. Across 18 Base-Instruct pairs spanning Qwen, Llama, and Gemma, harmfulness is decoded more readily from attention updates, particularly in early layers, whereas MLP updates contribute more strongly to harm-selective refusal writing in aligned models. Post-training strengthens MLP refusal writing in all 18 pairs without systematically improving harmfulness decodability, while positive MLP writing becomes more concentrated in deeper layers after post-training. Across six reasoning-enabled models, harmfulness remains decodable during reasoning while strong refusal writing concentrates at answer onset, a pattern that persists with reasoning disabled. Causal interventions further support this pathway: resetting downstream MLP updates to their clean values substantially reduces the refusal shift caused by upstream attention interventions. Building on these findings, we introduce Sentinel, which combines an attention-derived perception score P with an MLP-derived decision score D. Their joint profiles characterize jailbreaks and over-refusals, and conditional routing with both signals achieves a lower aggregate error rate than single-signal routing and evaluated fixed reminders across eight models. |
| 2026-10-09 | [CB-Orchestrator: Adaptive Workflow Optimization for LLM Agents via Contextual Bandits](http://arxiv.org/abs/2610.13064v1) | Jiahang Sun, Zhiwei Shang et al. | Large Language Model (LLM) agents have demonstrated remarkable capabilities in tackling complex tasks through agentic workflows. However, manually designing these workflows is labor-intensive and lacks flexibility to adapt to diverse queries. While automated workflow optimization is a promising direction, existing methods often incur expensive API costs during training-phase feedback collection. Furthermore, they fail to adaptively reuse high-quality workflows, which creates a bottleneck for both performance and cost-effectiveness. To address these limitations, we propose CB-Orchestrator, an adaptive framework that decouples workflow generation from selection. In the first stage, we construct a diverse pool of candidate workflows via evolutionary search. In the second stage, we formulate workflow selection as a Contextual Bandit problem, which enables sample-efficient learning by balancing exploration and exploitation, thereby significantly reducing the feedback required for training. During inference, the model adaptively selects the optimal workflow tailored to each query. Evaluations across five diverse benchmarks demonstrate that CB-Orchestrator consistently outperforms all baselines. Notably, on the MATH dataset, CB-Orchestrator achieves better performance than strong recent baselines while reducing training-phase API token overhead by no less than 40.55% and total end-to-end training (including workflow pool generation) token overhead by no less than 29.73%. |
| 2026-10-09 | [CoBe: A Benchmark For Conversational Counterfactual Text Editing](http://arxiv.org/abs/2610.13057v1) | Arvind Raghavan, Inwoo Hwang et al. | Counterfactual text editing is the ability to revise a given text in light of a hypothetical change while preserving facts not causally affected by that change. When asked what would have happened if a doctor had not recommended surgery, the model output should retain the patient's medical history and modify only the consequences of the decision. This requires causal understanding: a model must determine, without being explicitly instructed to, what events are affected by the hypothetical change and what should remain fixed. This skill is central to human reasoning, especially in safety-critical domains like healthcare or law. By contrast, the task of associational text generation merely generates text statistically associated with the altered condition. In the medical example, an associational response might make the patient's symptoms less severe because milder cases correlate with no surgery. Such responses produce a plausible scenario, but not the counterfactual version of the given one. Prior counterfactual-text datasets do not clearly separate these modes of reasoning. We introduce the Counterfactual Benchmark (CoBe), a testbed for models' zero-shot ability to counterfactually edit a given textual scenario when prompted to do so in a conversationally natural way. CoBe contains 2000 questions spanning healthcare, science and engineering, business, and everyday scenarios. We test a suite of frontier and smaller models, with frontier systems achieving an average score of $58.05\%$. Our findings indicate that models frequently default to associational generation rather than preserving the causal structure of the original scenario. Moreover, performance does not scale consistently with either model size or output token length, suggesting that the failure may reflect model training objectives rather than insufficient model capacity. |
| 2026-10-09 | [Evidential Multimodal Transformers with Dempster-Shafer Fusion for Emotion Recognition](http://arxiv.org/abs/2610.13013v1) | Muhammad Usama, Yeoreum Jung | Softmax-based multimodal emotion recognition methods often assign high confidence even when sensor data are corrupted or contradictory, for example when facial expressions suggest positive valence while vocal prosody indicates negative arousal. We present an Evidential Multimodal Transformer with Dempster-Shafer Fusion (EMT-DS) that addresses this problem through formal uncertainty quantification. EMT-DS models predictions as Subjective Opinions (Dirichlet distributions), separating epistemic uncertainty (lack of evidence) from aleatoric uncertainty (sensor noise). Modality-specific transformer encoders produce belief mass distributions that are fused with Dempster's Rule of Combination, which normalizes the joint belief mass by a conflict coefficient 1/(1-c), so that contradictory evidence is explicitly quantified as conflict and the fused prediction becomes diffuse rather than being forced into an unjustified consensus. Under subject-independent and speaker-independent evaluation on four benchmarks, EMT-DS achieves 84.3% accuracy on IEMOCAP versus 81.7% for the best baseline (p < 0.01), 82.4% on DEAP under leave-one-subject-out evaluation versus 78.1%, and 96.8% on the highly imbalanced KEMDy20, and it improves accuracy on CMU-MOSEI by 2.4 points over the strongest evidential baseline. In controlled ablation, it reduces Expected Calibration Error from 0.153 (softmax transformer) to 0.042. It maintains 71.2% accuracy under severe Gaussian noise on audio features (sigma = 1.0), where attention baselines drop to 45.3%, and degrades gracefully when entire modalities are absent. The fusion step has O(MK^2) complexity, compared with O(M^2 d^2) for cross-modal attention. The learned conflict metric also flags likely errors: reviewing 20% of samples captures 80% of misclassifications, which supports selective prediction in clinical and safety-critical systems. |
| 2026-10-09 | [Robust Halo Orbit Stationkeeping Under Missed Thrust Events and Maneuver Execution Errors](http://arxiv.org/abs/2610.12906v1) | Shaurya Shrivastava, Kenshiro Oguri | Spacecraft stationkeeping is subject to various operational uncertainties such as missed thrust events, maneuver execution errors, and dynamics modeling errors. Missed thrust events are the most disruptive of these, and current practices mitigate that risk through Monte Carlo-driven heuristics and conservative margins that offer no formal safety guarantees and do not scale to multiple missed thrust events. This paper models these unplanned thruster outages through the Markov Jump System framework, and represents maneuver execution errors and dynamics linearization errors as bounded multiplicative noise. The resulting closed-loop system is a linear time-varying periodic Markov jump system with multiplicative noise, for which Lyapunov-like mean square stability and periodic set invariance conditions are derived. A linear matrix inequality-based optimal controller synthesis framework then designs a periodic, mode-dependent mean-square stable feedback policy that respects path constraints with probability one. The framework is demonstrated on the Earth-Moon 9:2 near-rectilinear halo orbit (NRHO), under both impulsive and low-thrust propulsion, and on a Saturn-Enceladus $L_1$ NRHO using impulsive propulsion. Nonlinear Monte Carlo simulations show that the spacecraft recovers from an arbitrary number of missed thrust events while respecting path constraints, and that the multiplicative noise model introduces additional robustness to maneuver execution error and linearization error. |
| 2026-10-09 | [When Charts Deceive Web Agents: A Paired Benchmark for Misleading Visualization-Grounded Tasks](http://arxiv.org/abs/2610.12823v1) | Yisheng Li, Dongming Han et al. | Web agents increasingly browse, interpret, and act on web content on behalf of users. In many realistic workflows, however, the visual evidence they consume is not neutral: charts and dashboards can be misleading, shaping judgments without issuing any explicit malicious instruction. Unlike chart question answering, Web Agents treat charts as intermediate evidence whose misinterpretation can propagate into concrete downstream actions --- a reliability risk that remains underexplored. We introduce MisVisAgentBench, the first paired executable benchmark for evaluating how misleading visualizations affect Web Agent execution. It comprises 280 task instances in matched clean--misleading pairs that differ only in chart evidence. We further provide a diagnostic framework that isolates misleading-induced failures from baseline task difficulty (Zhu et al., 2025) and captures execution-level effects such as extra browsing steps and decision hesitation. Across thirteen vision-language Web Agents, replacing clean charts with misleading counterparts reduces the average task success rate by more than 34 percentage points. Moreover, these failures predominantly follow the targeted misleading action branch, suggesting that deceptive charts systematically redirect rather than randomly disrupt agent decisions. These results reveal that visualization robustness is a critical yet understudied dimension of Web Agent safety and reliability. |
| 2026-10-08 | [VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails](http://arxiv.org/abs/2610.12813v1) | Duy Nguyen, Lesly Miculicich et al. | LLM agents increasingly operate over sensitive resources, such as private documents and mutable system states, which can create serious safety and security risks. Runtime guardrails should therefore make inspectable, reproducible, and deterministic allow or refuse decisions before potentially irreversible agent actions are executed. However, safety and security requirements are rarely complete before deployment, so guardrails need to keep adapting as new risks emerge. Existing adaptive guardrails learn from deployment experience, but they usually apply safety policies through LLM inference, which can lead to inconsistent decisions and ambiguous policy interpretations. Formal guardrails, on the other hand, enforce policies deterministically, but they usually rely on pre-deployment or task-specific policies and have no mechanism to adapt to newly observed risks. We introduce VEPA, a framework for the verified evolution of formal agent guardrails. VEPA represents the guardrail as a formal policy of executable symbolic rules over typed agent events. During deployment, we use an LLM guided by deployment feedback to propose policy patches that fix newly observed failures while preserving previously correct behavior. Then, a deterministic verifier only accepts policies that are structurally valid and conflict-free. Finally, an evidence-aware authority layer activates a verified rule only when it has enough supporting evidence, so uncertain updates cannot directly affect guardrail decisions. Across ASSEBench-Safety, ASSEBench-Security, and ATBench, VEPA consistently outperforms baselines while expanding the scope of deterministic formal-policy enforcement. These gains also hold under more challenging deployment conditions, including mixed streams, corrupted feedback, and longer horizons. |

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



