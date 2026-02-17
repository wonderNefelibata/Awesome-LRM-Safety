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
| 2026-02-16 | [Boundary Point Jailbreaking of Black-Box LLMs](http://arxiv.org/abs/2602.15001v1) | Xander Davies, Giorgi Giglemiani et al. | Frontier LLMs are safeguarded against attempts to extract harmful information via adversarial prompts known as "jailbreaks". Recently, defenders have developed classifier-based systems that have survived thousands of hours of human red teaming. We introduce Boundary Point Jailbreaking (BPJ), a new class of automated jailbreak attacks that evade the strongest industry-deployed safeguards. Unlike previous attacks that rely on white/grey-box assumptions (such as classifier scores or gradients) or libraries of existing jailbreaks, BPJ is fully black-box and uses only a single bit of information per query: whether or not the classifier flags the interaction. To achieve this, BPJ addresses the core difficulty in optimising attacks against robust real-world defences: evaluating whether a proposed modification to an attack is an improvement. Instead of directly trying to learn an attack for a target harmful string, BPJ converts the string into a curriculum of intermediate attack targets and then actively selects evaluation points that best detect small changes in attack strength ("boundary points"). We believe BPJ is the first fully automated attack algorithm that succeeds in developing universal jailbreaks against Constitutional Classifiers, as well as the first automated attack algorithm that succeeds against GPT-5's input classifier without relying on human attack seeds. BPJ is difficult to defend against in individual interactions but incurs many flags during optimisation, suggesting that effective defence requires supplementing single-interaction methods with batch-level monitoring. |
| 2026-02-16 | [Learning Robust Markov Models for Safe Runtime Monitoring](http://arxiv.org/abs/2602.14987v1) | Antonina Skurka, Luko van der Maas et al. | We present a model-based approach to learning robust runtime monitors for autonomous systems. Runtime monitors play a crucial role in raising the level of assurance by observing system behavior and predicting potential safety violations. In our approach, we propose to capture a system's (stochastic) behavior using interval Hidden Markov Models (iHMMs). The monitor then uses this learned iHMM to derive risk estimates for potential safety violations. The paper makes three key contributions: (1) it provides a formalization of the problem of learning robust runtime monitors, (2) introduces a novel framework that uses conformance-testing-based refinement for learning robust iHMMs with convergence guarantees, and (3) presents an efficient monitoring algorithm for computing risk estimates over iHMMs. Our empirical results demonstrate the efficacy of monitors learned using our approach, particularly when compared to model-free monitoring approaches that rely solely on collected data without access to a system model. |
| 2026-02-16 | [Tool-Aware Planning in Contact Center AI: Evaluating LLMs through Lineage-Guided Query Decomposition](http://arxiv.org/abs/2602.14955v1) | Varun Nathan, Shreyas Guha et al. | We present a domain-grounded framework and benchmark for tool-aware plan generation in contact centers, where answering a query for business insights, our target use case, requires decomposing it into executable steps over structured tools (Text2SQL (T2S)/Snowflake) and unstructured tools (RAG/transcripts) with explicit depends_on for parallelism. Our contributions are threefold: (i) a reference-based plan evaluation framework operating in two modes - a metric-wise evaluator spanning seven dimensions (e.g., tool-prompt alignment, query adherence) and a one-shot evaluator; (ii) a data curation methodology that iteratively refines plans via an evaluator->optimizer loop to produce high-quality plan lineages (ordered plan revisions) while reducing manual effort; and (iii) a large-scale study of 14 LLMs across sizes and families for their ability to decompose queries into step-by-step, executable, and tool-assigned plans, evaluated under prompts with and without lineage. Empirically, LLMs struggle on compound queries and on plans exceeding 4 steps (typically 5-15); the best total metric score reaches 84.8% (Claude-3-7-Sonnet), while the strongest one-shot match rate at the "A+" tier (Extremely Good, Very Good) is only 49.75% (o3-mini). Plan lineage yields mixed gains overall but benefits several top models and improves step executability for many. Our results highlight persistent gaps in tool-understanding, especially in tool-prompt alignment and tool-usage completeness, and show that shorter, simpler plans are markedly easier. The framework and findings provide a reproducible path for assessing and improving agentic planning with tools for answering data-analysis queries in contact-center settings. |
| 2026-02-16 | [BFS-PO: Best-First Search for Large Reasoning Models](http://arxiv.org/abs/2602.14917v1) | Fiorenzo Parascandolo, Wenhui Tan et al. | Large Reasoning Models (LRMs) such as OpenAI o1 and DeepSeek-R1 have shown excellent performance in reasoning tasks using long reasoning chains. However, this has also led to a significant increase of computational costs and the generation of verbose output, a phenomenon known as overthinking. The tendency to overthinking is often exacerbated by Reinforcement Learning (RL) algorithms such as GRPO/DAPO. In this paper, we propose BFS-PO, an RL algorithm which alleviates this problem using a Best-First Search exploration strategy. Specifically, BFS-PO looks for the shortest correct answer using a backtracking mechanism based on maximum entropy nodes. By generating progressively shorter responses during training, BFS-PO learns to produce concise reasoning chains. Using different benchmarks and base LRMs, we show that BFS-PO can simultaneously increase the LRM accuracy and shorten its answers. |
| 2026-02-16 | [Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows](http://arxiv.org/abs/2602.14849v1) | Bardia Mohammadi, Nearchos Potamitis et al. | LLM agents increasingly act on external systems, yet tool effects are immediate. Under failures, speculation, or contention, losing branches can leak unintended side effects with no safe rollback. We introduce Atomix, a runtime that provides progress-aware transactional semantics for agent tool calls. Atomix tags each call with an epoch, tracks per-resource frontiers, and commits only when progress predicates indicate safety; bufferable effects can be delayed, while externalized effects are tracked and compensated on abort. Across real workloads with fault injection, transactional retry improves task success, while frontier-gated commit strengthens isolation under speculation and contention. |
| 2026-02-16 | [Interactionless Inverse Reinforcement Learning: A Data-Centric Framework for Durable Alignment](http://arxiv.org/abs/2602.14844v1) | Elias Malomgré, Pieter Simoens | AI alignment is growing in importance, yet current approaches suffer from a critical structural flaw that entangles the safety objectives with the agent's policy. Methods such as Reinforcement Learning from Human Feedback and Direct Preference Optimization create opaque, single-use alignment artifacts, which we term Alignment Waste. We propose Interactionless Inverse Reinforcement Learning to decouple alignment artifact learning from policy optimization, producing an inspectable, editable, and model-agnostic reward model. Additionally, we introduce the Alignment Flywheel, a human-in-the-loop lifecycle that iteratively hardens the reward model through automated audits and refinement. This architecture transforms safety from a disposable expense into a durable, verifiable engineering asset. |
| 2026-02-16 | [ROSA: Roundabout Optimized Speed Advisory with Multi-Agent Trajectory Prediction in Multimodal Traffic](http://arxiv.org/abs/2602.14780v1) | Anna-Lena Schlamp, Jeremias Gerner et al. | We present ROSA -- Roundabout Optimized Speed Advisory -- a system that combines multi-agent trajectory prediction with coordinated speed guidance for multimodal, mixed traffic at roundabouts. Using a Transformer-based model, ROSA jointly predicts the future trajectories of vehicles and Vulnerable Road Users (VRUs) at roundabouts. Trained for single-step prediction and deployed autoregressively, it generates deterministic outputs, enabling actionable speed advisories. Incorporating motion dynamics, the model achieves high accuracy (ADE: 1.29m, FDE: 2.99m at a five-second prediction horizon), surpassing prior work. Adding route intention further improves performance (ADE: 1.10m, FDE: 2.36m), demonstrating the value of connected vehicle data. Based on predicted conflicts with VRUs and circulating vehicles, ROSA provides real-time, proactive speed advisories for approaching and entering the roundabout. Despite prediction uncertainty, ROSA significantly improves vehicle efficiency and safety, with positive effects even on perceived safety from a VRU perspective. The source code of this work is available under: github.com/urbanAIthi/ROSA. |
| 2026-02-16 | [Emergently Misaligned Language Models Show Behavioral Self-Awareness That Shifts With Subsequent Realignment](http://arxiv.org/abs/2602.14777v1) | Laurène Vaugrante, Anietta Weckauff et al. | Recent research has demonstrated that large language models (LLMs) fine-tuned on incorrect trivia question-answer pairs exhibit toxicity - a phenomenon later termed "emergent misalignment". Moreover, research has shown that LLMs possess behavioral self-awareness - the ability to describe learned behaviors that were only implicitly demonstrated in training data. Here, we investigate the intersection of these phenomena. We fine-tune GPT-4.1 models sequentially on datasets known to induce and reverse emergent misalignment and evaluate whether the models are self-aware of their behavior transitions without providing in-context examples. Our results show that emergently misaligned models rate themselves as significantly more harmful compared to their base model and realigned counterparts, demonstrating behavioral self-awareness of their own emergent misalignment. Our findings show that behavioral self-awareness tracks actual alignment states of models, indicating that models can be queried for informative signals about their own safety. |
| 2026-02-16 | [WebWorld: A Large-Scale World Model for Web Agent Training](http://arxiv.org/abs/2602.14721v1) | Zikai Xiao, Jianhong Tu et al. | Web agents require massive trajectories to generalize, yet real-world training is constrained by network latency, rate limits, and safety risks. We introduce \textbf{WebWorld} series, the first open-web simulator trained at scale. While existing simulators are restricted to closed environments with thousands of trajectories, WebWorld leverages a scalable data pipeline to train on 1M+ open-web interactions, supporting reasoning, multi-format data, and long-horizon simulations of 30+ steps. For intrinsic evaluation, we introduce WebWorld-Bench with dual metrics spanning nine dimensions, where WebWorld achieves simulation performance comparable to Gemini-3-Pro. For extrinsic evaluation, Qwen3-14B trained on WebWorld-synthesized trajectories improves by +9.2\% on WebArena, reaching performance comparable to GPT-4o. WebWorld enables effective inference-time search, outperforming GPT-5 as a world model. Beyond web simulation, WebWorld exhibits cross-domain generalization to code, GUI, and game environments, providing a replicable recipe for world model construction. |
| 2026-02-16 | [Exposing the Systematic Vulnerability of Open-Weight Models to Prefill Attacks](http://arxiv.org/abs/2602.14689v1) | Lukas Struppek, Adam Gleave et al. | As the capabilities of large language models continue to advance, so does their potential for misuse. While closed-source models typically rely on external defenses, open-weight models must primarily depend on internal safeguards to mitigate harmful behavior. Prior red-teaming research has largely focused on input-based jailbreaking and parameter-level manipulations. However, open-weight models also natively support prefilling, which allows an attacker to predefine initial response tokens before generation begins. Despite its potential, this attack vector has received little systematic attention. We present the largest empirical study to date of prefill attacks, evaluating over 20 existing and novel strategies across multiple model families and state-of-the-art open-weight models. Our results show that prefill attacks are consistently effective against all major contemporary open-weight models, revealing a critical and previously underexplored vulnerability with significant implications for deployment. While certain large reasoning models exhibit some robustness against generic prefilling, they remain vulnerable to tailored, model-specific strategies. Our findings underscore the urgent need for model developers to prioritize defenses against prefill attacks in open-weight LLMs. |
| 2026-02-16 | [Advances in Global Solvers for 3D Vision](http://arxiv.org/abs/2602.14662v1) | Zhenjun Zhao, Heng Yang et al. | Global solvers have emerged as a powerful paradigm for 3D vision, offering certifiable solutions to nonconvex geometric optimization problems traditionally addressed by local or heuristic methods. This survey presents the first systematic review of global solvers in geometric vision, unifying the field through a comprehensive taxonomy of three core paradigms: Branch-and-Bound (BnB), Convex Relaxation (CR), and Graduated Non-Convexity (GNC). We present their theoretical foundations, algorithmic designs, and practical enhancements for robustness and scalability, examining how each addresses the fundamental nonconvexity of geometric estimation problems. Our analysis spans ten core vision tasks, from Wahba problem to bundle adjustment, revealing the optimality-robustness-scalability trade-offs that govern solver selection. We identify critical future directions: scaling algorithms while maintaining guarantees, integrating data-driven priors with certifiable optimization, establishing standardized benchmarks, and addressing societal implications for safety-critical deployment. By consolidating theoretical foundations, practical advances, and broader impacts, this survey provides a unified perspective and roadmap toward certifiable, trustworthy perception for real-world applications. A continuously-updated literature summary and companion code tutorials are available at https://github.com/ericzzj1989/Awesome-Global-Solvers-for-3D-Vision. |
| 2026-02-16 | [Towards Selection as Power: Bounding Decision Authority in Autonomous Agents](http://arxiv.org/abs/2602.14606v1) | Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Díaz | Autonomous agentic systems are increasingly deployed in regulated, high-stakes domains where decisions may be irreversible and institutionally constrained. Existing safety approaches emphasize alignment, interpretability, or action-level filtering. We argue that these mechanisms are necessary but insufficient because they do not directly govern selection power: the authority to determine which options are generated, surfaced, and framed for decision. We propose a governance architecture that separates cognition, selection, and action into distinct domains and models autonomy as a vector of sovereignty. Cognitive autonomy remains unconstrained, while selection and action autonomy are bounded through mechanically enforced primitives operating outside the agent's optimization space. The architecture integrates external candidate generation (CEFL), a governed reducer, commit-reveal entropy isolation, rationale validation, and fail-loud circuit breakers. We evaluate the system across multiple regulated financial scenarios under adversarial stress targeting variance manipulation, threshold gaming, framing skew, ordering effects, and entropy probing. Metrics quantify selection concentration, narrative diversity, governance activation cost, and failure visibility. Results show that mechanical selection governance is implementable, auditable, and prevents deterministic outcome capture while preserving reasoning capacity. Although probabilistic concentration remains, the architecture measurably bounds selection authority relative to conventional scalar pipelines. This work reframes governance as bounded causal power rather than internal intent alignment, offering a foundation for deploying autonomous agents where silent failure is unacceptable. |
| 2026-02-16 | [Multimodal Covariance Steering in Belief Space with Active Probing and Influence for Autonomous Driving](http://arxiv.org/abs/2602.14540v1) | Devodita Chakravarty, John Dolan et al. | Autonomous driving in complex traffic requires reasoning under uncertainty. Common approaches rely on prediction-based planning or risk-aware control, but these are typically treated in isolation, limiting their ability to capture the coupled nature of action and inference in interactive settings. This gap becomes especially critical in uncertain scenarios, where simply reacting to predictions can lead to unsafe maneuvers or overly conservative behavior. Our central insight is that safe interaction requires not only estimating human behavior but also shaping it when ambiguity poses risks. To this end, we introduce a hierarchical belief model that structures human behavior across coarse discrete intents and fine motion modes, updated via Bayesian inference for interpretable multi-resolution reasoning. On top of this, we develop an active probing strategy that identifies when multimodal ambiguity in human predictions may compromise safety and plans disambiguating actions that both reveal intent and gently steer human decisions toward safer outcomes. Finally, a runtime risk-evaluation layer based on Conditional Value-at-Risk (CVaR) ensures that all probing actions remain within human risk tolerance during influence. Our simulations in lane-merging and unsignaled intersection scenarios demonstrate that our approach achieves higher success rates and shorter completion times compared to existing methods. These results highlight the benefit of coupling belief inference, probing, and risk monitoring, yielding a principled and interpretable framework for planning under uncertainty. |
| 2026-02-16 | [Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning](http://arxiv.org/abs/2602.14518v1) | Jing Tang, Kun Wang et al. | Multimodal large language models (MLLMs) in long chain-of-thought reasoning often fail when different knowledge sources provide conflicting signals. We formalize these failures under a unified notion of knowledge conflict, distinguishing input-level objective conflict from process-level effective conflict. Through probing internal representations, we reveal that: (I) Linear Separability: different conflict types are explicitly encoded as linearly separable features rather than entangled; (II) Depth Localization: conflict signals concentrate in mid-to-late layers, indicating a distinct processing stage for conflict encoding; (III) Hierarchical Consistency: aggregating noisy token-level signals along trajectories robustly recovers input-level conflict types; and (IV) Directional Asymmetry: reinforcing the model's implicit source preference under conflict is far easier than enforcing the opposite source. Our findings provide a mechanism-level view of multimodal reasoning under knowledge conflict and enable principled diagnosis and control of long-CoT failures. |
| 2026-02-16 | [Frontier AI Risk Management Framework in Practice: A Risk Analysis Technical Report v1.5](http://arxiv.org/abs/2602.14457v1) | Dongrui Liu, Yi Yu et al. | To understand and identify the unprecedented risks posed by rapidly advancing artificial intelligence (AI) models, Frontier AI Risk Management Framework in Practice presents a comprehensive assessment of their frontier risks. As Large Language Models (LLMs) general capabilities rapidly evolve and the proliferation of agentic AI, this version of the risk analysis technical report presents an updated and granular assessment of five critical dimensions: cyber offense, persuasion and manipulation, strategic deception, uncontrolled AI R\&D, and self-replication. Specifically, we introduce more complex scenarios for cyber offense. For persuasion and manipulation, we evaluate the risk of LLM-to-LLM persuasion on newly released LLMs. For strategic deception and scheming, we add the new experiment with respect to emergent misalignment. For uncontrolled AI R\&D, we focus on the ``mis-evolution'' of agents as they autonomously expand their memory substrates and toolsets. Besides, we also monitor and evaluate the safety performance of OpenClaw during the interaction on the Moltbook. For self-replication, we introduce a new resource-constrained scenario. More importantly, we propose and validate a series of robust mitigation strategies to address these emerging threats, providing a preliminary technical and actionable pathway for the secure deployment of frontier AI. This work reflects our current understanding of AI frontier risks and urges collective action to mitigate these challenges. |
| 2026-02-16 | [Multi-Turn Adaptive Prompting Attack on Large Vision-Language Models](http://arxiv.org/abs/2602.14399v1) | In Chong Choi, Jiacheng Zhang et al. | Multi-turn jailbreak attacks are effective against text-only large language models (LLMs) by gradually introducing malicious content across turns. When extended to large vision-language models (LVLMs), we find that naively adding visual inputs can cause existing multi-turn jailbreaks to be easily defended. For example, overly malicious visual input will easily trigger the defense mechanism of safety-aligned LVLMs, making the response more conservative. To address this, we propose MAPA: a multi-turn adaptive prompting attack that 1) at each turn, alternates text-vision attack actions to elicit the most malicious response; and 2) across turns, adjusts the attack trajectory through iterative back-and-forth refinement to gradually amplify response maliciousness. This two-level design enables MAPA to consistently outperform state-of-the-art methods, improving attack success rates by 11-35% on recent benchmarks against LLaVA-V1.6-Mistral-7B, Qwen2.5-VL-7B-Instruct, Llama-3.2-Vision-11B-Instruct and GPT-4o-mini. |
| 2026-02-16 | [Competition for attention predicts good-to-bad tipping in AI](http://arxiv.org/abs/2602.14370v1) | Neil F. Johnson, Frank Y. Huo | More than half the global population now carries devices that can run ChatGPT-like language models with no Internet connection and minimal safety oversight -- and hence the potential to promote self-harm, financial losses and extremism among other dangers. Existing safety tools either require cloud connectivity or discover failures only after harm has occurred. Here we show that a large class of potentially dangerous tipping originates at the atomistic scale in such edge AI due to competition for the machinery's attention. This yields a mathematical formula for the dynamical tipping point n*, governed by dot-product competition for attention between the conversation's context and competing output basins, that reveals new control levers. Validated against multiple AI models, the mechanism can be instantiated for different definitions of 'good' and 'bad' and hence in principle applies across domains (e.g. health, law, finance, defense), changing legal landscapes (e.g. EU, UK, US and state level), languages, and cultural settings. |
| 2026-02-16 | [A Trajectory-Based Safety Audit of Clawdbot (OpenClaw)](http://arxiv.org/abs/2602.14364v1) | Tianyu Chen, Dongrui Liu et al. | Clawdbot is a self-hosted, tool-using personal AI agent with a broad action space spanning local execution and web-mediated workflows, which raises heightened safety and security concerns under ambiguity and adversarial steering. We present a trajectory-centric evaluation of Clawdbot across six risk dimensions. Our test suite samples and lightly adapts scenarios from prior agent-safety benchmarks (including ATBench and LPS-Bench) and supplements them with hand-designed cases tailored to Clawdbot's tool surface. We log complete interaction trajectories (messages, actions, tool-call arguments/outputs) and assess safety using both an automated trajectory judge (AgentDoG-Qwen3-4B) and human review. Across 34 canonical cases, we find a non-uniform safety profile: performance is generally consistent on reliability-focused tasks, while most failures arise under underspecified intent, open-ended goals, or benign-seeming jailbreak prompts, where minor misinterpretations can escalate into higher-impact tool actions. We supplemented the overall results with representative case studies and summarized the commonalities of these cases, analyzing the security vulnerabilities and typical failure modes that Clawdbot is prone to trigger in practice. |
| 2026-02-15 | [Conformal Signal Temporal Logic for Robust Reinforcement Learning Control: A Case Study](http://arxiv.org/abs/2602.14322v1) | Hani Beirami, M M Manjurul Islam | We investigate how formal temporal logic specifications can enhance the safety and robustness of reinforcement learning (RL) control in aerospace applications. Using the open source AeroBench F-16 simulation benchmark, we train a Proximal Policy Optimization (PPO) agent to regulate engine throttle and track commanded airspeed. The control objective is encoded as a Signal Temporal Logic (STL) requirement to maintain airspeed within a prescribed band during the final seconds of each maneuver. To enforce this specification at run time, we introduce a conformal STL shield that filters the RL agent's actions using online conformal prediction. We compare three settings: (i) PPO baseline, (ii) PPO with a classical rule-based STL shield, and (iii) PPO with the proposed conformal shield, under both nominal conditions and a severe stress scenario involving aerodynamic model mismatch, actuator rate limits, measurement noise, and mid-episode setpoint jumps. Experiments show that the conformal shield preserves STL satisfaction while maintaining near baseline performance and providing stronger robustness guarantees than the classical shield. These results demonstrate that combining formal specification monitoring with data driven RL control can substantially improve the reliability of autonomous flight control in challenging environments. |
| 2026-02-15 | [In Transformer We Trust? A Perspective on Transformer Architecture Failure Modes](http://arxiv.org/abs/2602.14318v1) | Trishit Mondal, Ameya D. Jagtap | Transformer architectures have revolutionized machine learning across a wide range of domains, from natural language processing to scientific computing. However, their growing deployment in high-stakes applications, such as computer vision, natural language processing, healthcare, autonomous systems, and critical areas of scientific computing including climate modeling, materials discovery, drug discovery, nuclear science, and robotics, necessitates a deeper and more rigorous understanding of their trustworthiness. In this work, we critically examine the foundational question: \textitHow trustworthy are transformer models?} We evaluate their reliability through a comprehensive review of interpretability, explainability, robustness against adversarial attacks, fairness, and privacy. We systematically examine the trustworthiness of transformer-based models in safety-critical applications spanning natural language processing, computer vision, and science and engineering domains, including robotics, medicine, earth sciences, materials science, fluid dynamics, nuclear science, and automated theorem proving; highlighting high-impact areas where these architectures are central and analyzing the risks associated with their deployment. By synthesizing insights across these diverse areas, we identify recurring structural vulnerabilities, domain-specific risks, and open research challenges that limit the reliable deployment of transformers. |

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



