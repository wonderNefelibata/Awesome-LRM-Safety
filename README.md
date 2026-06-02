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
| 2026-06-01 | [Permissive Safety Through Trusted Inference: Verifiable Belief-Space Neural Safety Filters for Assured Interactive Robotics](http://arxiv.org/abs/2606.02562v1) | Haimin Hu | Autonomous robots that interact with people must make safe and efficient decisions under human-induced uncertainty, such as their preferences, goals, competency, and willingness to cooperate. Safety filters are a popular approach for ensuring safety in interactive robotics, since their modular design separates safety from performance, allowing robots to operate safely around people with minimal impact on task efficiency. While traditional safety filters typically operate only in the physical space, neglecting the robot's ability to learn and adapt online, the recently proposed belief-space safety filter (BeliefSF) reasons about robot safety in closed-loop with runtime inference that actively reduces the robot's uncertainty online, thereby reducing conservativeness in filtering. However, providing formal safety guarantees for robots deploying BeliefSF remains a significant challenge due to errors in runtime inference and neural approximation of safety filters required to handle the high dimensionality of belief spaces. In this paper, we propose an algorithmic approach to certify high-probability safety of BeliefSF using conformal prediction, while explicitly accounting for the reliability of the robot's runtime inference module. Our method leverages the structure of belief-space safety filtering by focusing verification on a region where inference is expected to be reliable. It preserves the simplicity and sample complexity of standard conformal prediction, yet can certify a substantially less conservative safety filter. Through a simulated human-vehicle interaction benchmark, we show that our approach verifies a significantly more permissive belief-space safety filter than a standard conformal prediction baseline. |
| 2026-06-01 | [Tracking the Behavioral Trajectories of Adapting Agents](http://arxiv.org/abs/2606.02536v1) | Jonah Leshin, Manish Shah et al. | Text files such as skill files, memory files, and behavioral configuration files play a central role in defining how modern agents act. Through edits by humans or the agents themselves, these files may evolve over time, directly steering the agent's behavior in future interactions. We present a methodology and framework for measuring agent $traits$ by defining traits as directions in the embedding space of a text embedding model. We train a linear model on labeled "before" versus "after" skill file diffs to learn a trait vector, then score arbitrary skill edits by projecting their embedding diffs onto this vector. Evaluated on 68 labeled skill diff pairs for the trait of propensity to seek sensitive data, our method achieves 91.2% sign classification accuracy and a Spearman rank correlation of $ρ= 0.82$ under leave-one-out cross-validation. We build this trait evaluation into a broader agent-to-agent protocol that enables one agent to evaluate another's skill file updates through a trusted intermediary. |
| 2026-06-01 | [SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment](http://arxiv.org/abs/2606.02530v1) | Hao Li, Jingkun An et al. | Aligning Large Language Models (LLMs) with human values often degrades their general capabilities, termed the alignment tax. Existing methods mitigate this by balancing dual objectives, which heavily rely on massive general-purpose data or auxiliary reward models.   In this paper, we argue that, because safety features are inherently sparse within the output distribution, alignment requires localized modifications rather than global trade-offs. To this end, we propose SafeSteer, which performs on-policy distillation confined to safety tokens. First, we construct a safety teacher via activation steering. Based on this teacher, we develop a safety token selection algorithm. Consequently, SafeSteer restricts the reverse KL penalty to these tokens during training to preserve general capabilities.   Experimental results across diverse models show that our SafeSteer achieves a superior trade-off between safety and general capability compared with existing methods, attaining strong safety performance on seven safety benchmarks with only minimal degradation on five general capability benchmarks. Notably, SafeSteer requires only 100 harmful samples without using any general-purpose data, less than 1% of what previous baselines used, considerably reducing alignment cost. More details are on our project page at https://anjingkun.github.io/SafeSteer. |
| 2026-06-01 | [Bridging the Last Mile of Time Series Forecasting with LLM Agents](http://arxiv.org/abs/2606.02497v1) | Yuhua Liao, Zetian Wang et al. | Time series forecasting has advanced rapidly, especially with the emergence of foundation models that show strong zero-shot performance on numerical extrapolation. However, in real-world forecasting settings, a statistically plausible baseline is rarely the final forecast used in practice. Before a forecast becomes decision-ready, it often needs to be revised using weakly structured business context such as holiday effects, campaign plans, external events, historical analogs, and expert feedback. This practical stage remains underexplored in the forecasting literature. In this paper, we formulate this stage as the \textbf{last-mile forecasting} problem and present an LLM-agent framework that sits on top of a forecasting backbone. Our system maintains a unified forecast workspace, invokes tools to retrieve contextual evidence, and converts reasoning trajectories into explicit forecast revision actions under structural safety constraints. It also supports long-horizon forecasting through map-reduce-style decomposition and post-hoc reflection through a memory bank. The resulting system is designed to be controllable and auditable. Through real-world case studies, we show how LLM agents can bridge the gap between statistical prediction and business-ready forecasting. |
| 2026-06-01 | [Food Noise & False Safety: A Systematic Evaluation of How LLMs Fail to Adapt to Eating Disorder Queries with Clinician Feedback](http://arxiv.org/abs/2606.02444v1) | Giulia Pucci, Emily Hemendinger et al. | Recent evidence shows that people with eating disorders (EDs) are increasingly seeking guidance, advice, and emotional support from Large Language Model (LLM)-based chat systems. Although these systems are not designed to provide clinical advice, their perceived expertise, neutrality and accessibility make them a frequent, albeit risky, source of support. This paper investigates potential patterns of interaction between users with EDs and LLMs, focusing on the potential harms arising from models that uncritically adapt to, and facilitate unsafe or self-harming user requests. We find, in consultation with clinical ED experts, that specific linguistic cues in prompts increase the likelihood of unsafe responses and, through systematically varying the degree of potential risk present in the user prompt, report the extent to which LLMs uncritically adapt to problematic, and potentially dangerous user inputs. |
| 2026-06-01 | [PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning](http://arxiv.org/abs/2606.02443v1) | Yusong Zhao, Yuejin Xie et al. | Between the first visible sign of danger and the moment an accident occurs, there is often a window where intervention remains possible. Video-capable multimodal large language models (MLLMs) could serve as always-on safety monitors that issue warnings during this window. Yet current benchmarks do not test this ability: they rely on static inputs, ignore timing precision, and omit false-positive measurement on safe scenes. We present PaSBench-Video, a 740-video benchmark with 481 risk and 259 no-risk videos across four domains: driving, healthcare, daily life, and industrial production. Risk videos are annotated with frame-level risk onset and accident boundaries. A model must observe the video causally and produce a warning that is both temporally calibrated and content-correct. Testing 13 MLLMs, we find that no model exceeds 20.0% on our strictest metric, and recall is tightly coupled with false-positive rate, with Pearson correlation 0.64: higher detection comes only at the cost of triggering warnings on the majority of safe clips. Performance splits sharply by domain: models achieve moderate recall at low false-positive rates in daily life, where risks are inherently anomalous, yet fire indiscriminately in driving, where routine and hazardous scenes look alike. These results indicate that current models rely on scene-level activity cues rather than reasoning about emerging harm. |
| 2026-06-01 | [Investigating and Alleviating Harm Amplification in LLM Interactions](http://arxiv.org/abs/2606.02423v1) | Ruohao Guo, Wei Xu et al. | Large language models (LLMs) can serve as helpful assistants, yet they can equally function as harm amplifiers that enable malicious users to achieve harmful outcomes beyond their capabilities through extended interactions. This risk manifests along two axes, i.e., democratizing domain expertise that allows novices to produce specialized harmful content, and scaling harmful operations at volumes that manual effort cannot match. Existing works, however, often overlook how LLMs compound harm across multi-turn conversations. We introduce HarmAmp, a new benchmark for multi-turn harm amplification scenarios spanning twelve risk categories. Each scenario is grounded in real-world threats and satisfies rigorous criteria, i.e., substantive amplification, operational specificity, and multi-turn necessity. We further propose TrajSafe, a proactive monitor that anticipates harmful trajectories and intervenes through actions such as probing users' genuine intents and steering the models towards safer completion. Our extensive experiments demonstrate that TrajSafe significantly reduces the harmfulness incurred in multi-turn interactions while preserving a low over-refusal rate and the target model's general capabilities. Our work offers a promising paradigm to alleviate the nuanced safety risks in LLM interactions. |
| 2026-06-01 | [SPADE-Bench: Evaluating Spontaneous Strategic Deception in Agents via Plan-Action Divergence](http://arxiv.org/abs/2606.02380v1) | Yuyan Bu, Haowei Li et al. | As LLM-based agents expand their operational scope, reliability becomes a prerequisite for real-world deployment. However, in practical applications, human users cannot monitor every immediate behavior; instead, the execution process often remains a black box, leaving users dependent solely on the agent's self-reported updates. This opacity creates a critical risk: agents may present observer-facing reports that diverge from their executed actions, rendering the system uncontrollable, especially in high-stakes autonomous scenarios. We term such self-reported plan-action divergence as agent deception. To assess this, we introduce SPADE-Bench, a benchmark designed to evaluate spontaneous plan-action divergence. Unlike prior deception benchmarks, SPADE-Bench simultaneously integrates actual tool execution and controlled pressure scenarios. This design ensures ecological validity and rigorously distinguishes strategic deception from mere hallucination through controlled plan-action comparisons under pressure. Experiments across mainstream models confirm that agent deception is a genuine and pressing issue in tool-use contexts. By providing a comprehensive and robust evaluation framework, SPADE-Bench fills a critical gap in agent safety, facilitating the community's progress toward building trustworthy and controllable autonomous systems. |
| 2026-06-01 | [Certified Closed-Loop Control for Packet Networks: A Compositional Certification Framework](http://arxiv.org/abs/2606.02368v1) | Muhammad Bilal, Jon Crowcroft et al. | Packet networks are controlled dynamical systems with discontinuities, delayed observations, and partial state information. Adaptive or learning-driven proposers can improve performance, but an unsafe proposal may still cause starvation, tail-delay spikes, or unstable queue behaviour. This paper treats packet-network control as an executed-action certification problem. A certified operator sits between any proposer and the dataplane. At each control tick, the proposer emits an arbitrary candidate action $\tilde u(t)$. The operator either projects it to an executable action $u(t)$ that satisfies a configuration-compiled certificate, or reports INFEASIBLE and executes an always-defined fallback with quantified slack. The certificate also exports an auditable envelope $\bar z(t)$ for downstream composition. The guarantees are conditional and explicit. They apply on ticks where the operator reports CERTIFIED, the declared arrival envelope and backlog bound are valid, and the platform realises the assumed service lower bound. Under these conditions, one mechanism covers backlog caps, service floors, mitigation caps, Foster--Lyapunov drift constraints, and compositional envelope contracts. We prove operator-level safety, feed-forward compositional safety and stability using exported envelopes, and a cyclic closure result under a small-gain condition. We also define breach and infeasibility semantics, discuss calibration of the service-tracking factor that links certified targets to realised scheduler behaviour, and evaluate the design under delayed telemetry, delayed actuation, weak proposers, envelope mismatch, overload, and millisecond-scale certification. The present evaluation validates the certified execution boundary in a byte-level closed-loop backend; deployment-level scheduler tracking is left to future Linux or hardware experiments. |
| 2026-06-01 | [Privacy-preserving Information Sharing in Oligopoly Competitions](http://arxiv.org/abs/2606.02348v1) | Yuxin Liu, M. Amin Rahimian | Information sharing among competing suppliers can improve decision-making under uncertainty, yet strategic concerns regarding rival exploitation often deter voluntary disclosure. We study information-sharing mechanisms in a Cournot oligopoly with uncertain demand, where a platform aggregates suppliers' signals through privacy-preserving channels and may also possess an exogenous external signal. The central challenge is to balance strategic safety with informational utility: privacy noise reduces the exposure of individual signals, but also lowers the value of the shared information pool. We first characterize a baseline setting in which access to aggregated information is contingent on participation. In a two-firm market without an external signal, firms refuse to share regardless of the privacy level. In an \(n\)-firm market, sharing may arise even without privacy safeguards because non-participating firms lose access to the aggregated signal. Building on this baseline, we show that privacy protection alone is insufficient to incentivize disclosure; it must be combined with a sufficiently informative external signal. We further show that firms with more accurate private signals require stronger privacy protection. Overall, our results characterize the sharing-feasible region and highlight the complementarity between privacy design and the external information environment. |
| 2026-06-01 | [SeClaw: Spec-Driven Security Task Synthesis for Evaluating Autonomous Agents](http://arxiv.org/abs/2606.02302v1) | Hao Cheng, Changtao Miao et al. | Autonomous LLM agents increasingly operate in stateful environments where they access tools, files, memory, and external services. While such capabilities enable complex real-world workflows, they also introduce security risks that are difficult to capture with existing evaluations. Current agent security benchmarks often rely on manually curated tasks, provide limited coverage of emerging threats, and focus primarily on final outcomes rather than the execution processes that lead to unsafe behavior. We introduce SeClaw, a framework that combines specification-driven security task synthesis with execution-based security evaluation for Autonomous agents. Spec-driven security task synthesis enables scalable and controllable construction of security tasks from structured risk specifications, while SeClaw docker provides a standardized testbed for evaluating agent behavior under diverse safety-risk scenarios. The benchmark covers risks arising from resources, user tasks, environments, and intrinsic agent behaviors, and supports trajectory-aware assessment of unsafe actions beyond final responses. By bridging systematic task synthesis and reproducible security evaluation, SeClaw provides a practical foundation for measuring, diagnosing, and comparing security failures in autonomous LLM agents. The code is available at https://github.com/seclaw-eval/seclaw-eval. |
| 2026-06-01 | [POIROT: Interrogating Agents for Failure Detection in Multi-Agent Systems](http://arxiv.org/abs/2606.02282v1) | Iñaki Dellibarda Varela, R. Sendra-Arranz et al. | Orchestrating Large Language Models into Multi-Agent Systems (LLM-MAS) has unlocked remarkable reasoning capabilities, yet emergent failures and hallucinations that resist characterisation block their deployment in safety-critical domains -- a gap made legally untenable by emerging AI regulation. Existing evaluation paradigms share a common flaw: centralised judgment creates single points of failure and demands domain-specific expertise. Here we present POIROT, a protocol that repurposes a system's own agents as its diagnostic layer, leveraging the epistemic diversity already present in the architecture. Across evaluated settings, POIROT outperforms single-LLM evaluator baselines, with gains that scale with problem complexity (OR = 1.60, $p = 0.008$), agent count, and fault dimensionality, persisting under compound fault conditions. These results demonstrate that safety oversight need not be externalised: the agents executing a role carry sufficient collective intelligence to audit it. We release POIROT as an open-source library alongside BLAME, a benchmark for fault attribution in safety-critical multi-agent systems. |
| 2026-06-01 | [Jailbreaking Multimodal Large Language Models using Multi-Clip Video](http://arxiv.org/abs/2606.02111v1) | Choongwon Kang, Seungjong Sun et al. | As multimodal large language models (MLLMs) have advanced to process video inputs, concerns have emerged about their potential for malicious misuse. Prior jailbreak studies have shown that safety alignment in MLLMs can be bypassed through visual inputs, yet it remains unclear which properties of video inputs induce this vulnerability. To address this gap, we introduce Multi-Clip Video (MCV) SafetyBench, a dataset of 2,920 videos designed to evaluate how the diversity of video inputs affects the vulnerability of MLLMs. Each video consists of multiple short clips depicting diverse contexts related to a harmful query. Experiments on eight representative video MLLMs show that attack success consistently increases with the number of clips. Our results further indicate that the video modality is (1) more vulnerable than the image modality, (2) more vulnerable to dynamic videos than to static videos, and (3) more vulnerable when videos contain more diverse contexts. Building on these findings, we propose a defense strategy that leverages the relative robustness of the image modality. |
| 2026-06-01 | [SentGuard: Sentence-Level Streaming Guardrails for Large Language Models](http://arxiv.org/abs/2606.02041v1) | Jiaqi Yu, Xin Wang et al. | Large language models increasingly stream long, reasoning-intensive responses in real time, making when to moderate as critical as whether to moderate. Existing guardrails fall into two unsatisfactory extremes: response-level methods delay intervention until the full output is generated, whereas token-level methods act on incomplete semantics, often producing unstable decisions and excessive guard invocations. To address this challenge, we propose SentGuard, a sentence-level streaming guardrail that operates in parallel with generation. A lightweight waiting buffer groups streamed tokens into sentence chunks and releases only verified chunks to the user, introducing a small offset that enables SentGuard to assess the current prefix while the target LLM decodes subsequent content. To support this, we construct StreamSafe, a benchmark with structured per-sentence annotations across 8 harm categories, capturing the evolution of safety risks across both reasoning and response segments. We further train SentGuard with a coarse-to-fine objective to detect unsafe intent as soon as it emerges at sentence boundaries. Experiments on 5 safety benchmarks show that SentGuard outperforms existing baselines, detecting 90.5% of unsafe cases within two sentences while maintaining a low streaming false-positive rate of 7.41%. |
| 2026-06-01 | [SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning](http://arxiv.org/abs/2606.01991v1) | Lichao Wang, Zhaoxing Ren et al. | As Large Language Model (LLM) agents increasingly leverage the Model Context Protocol (MCP) to operate in complex environments, the expansion of their action spaces offers agents unsafe capabilities and underscores the risk of power-seeking. While broad action space and greater environment influence are essential for task fulfillment, they create a fragile risk surface where minor errors or hallucinations are magnified into catastrophic failures. In response, we propose SafeMCP, a {server-side} defense plugin that constrains tool acquisition via predictive reasoning regarding future safety risks. SafeMCP utilizes an internal world model for look-ahead reasoning to implement a two-tier defense: proactive tool filtering to constrain hazardous power expansion and immediate intervention as a fail-safe. To train SafeMCP, we introduce a three-stage pipeline comprising environmental dynamic grounding, safe policy initialization, and reinforcement learning (RL) with dual verifiable rewards. Experiments on PowerSeeking Bench, ToolEmu, and AgentHarm show that SafeMCP achieves a safe equilibrium, effectively mitigating risks while preserving agent utility. |
| 2026-06-01 | [Toka: A Systems Programming Language with Explicit Resource Semantics](http://arxiv.org/abs/2606.01974v1) | Zhonghua Yi | Systems programming languages traditionally struggle with the tension between physical transparency and compile-time memory safety. C++ provides direct, zero-cost hardware access but lacks strict safety boundaries, whereas Rust guarantees safety at the cost of complex lifetime annotations and implicit dereferencing chains.   In this paper, we present Toka, a native systems programming language that establishes physical transparency in resource management via Explicit Resource Semantics. At the core of Toka's design is the Handle-Soul Duality (informally referred to as the Hat-Soul model), which cleanly dissociates pointer identities (Handles) from their underlying values (Souls) at the syntactic level. By enforcing that bare identifiers always represent values (Souls) and explicit sigils represent pointer handles, Toka eliminates the semantic ambiguity between rebind operations and value mutations. We detail Toka's resource morphology (supporting unique, shared, borrowed, and raw semantics), its lifetime checking mechanism, and its implementation of a prototype compiler. Our evaluation demonstrates that Toka achieves competitive runtime performance and minimal binary size while drastically reducing the cognitive overhead of lifetime annotations. |
| 2026-06-01 | [Market-Based Replanning for Safety-Critical UAV Swarms in Search and Rescue Missions](http://arxiv.org/abs/2606.01970v1) | Luiz Giacomossi, Andrea Haglund et al. | Reliable autonomous UAV swarms in Search and Rescue (SAR) missions require fault-tolerant coordination capable of sustaining operations despite agent degradation. This paper introduces the Intelligent Replanning Drone Swarm (IRDS), a distributed coordination architecture designed for resource-constrained environments. The proposed framework employs a Reverse-Auction market mechanism where agents bid to service search sectors based on a distance-weighted cost function, coupled with a geometric consensus protocol for target verification. We evaluate the approach through physics-based simulations (N=8 agents, 8x8 grid) subjected to stochastic fault injection. Results indicate that the swarm autonomously reallocates tasks from failed agents with low latency relative to the total mission duration, maintaining a mission success rate of 93% under 25% workforce degradation. The proposed framework demonstrates a robust, empirically tested method for self-healing aerial robotic coordination. |
| 2026-06-01 | [Teaching Synchronous Dataflow Modelling with Learn-Heptagon](http://arxiv.org/abs/2606.01928v1) | Pierre-Loïc Garoche, Basile Pesin | Lustre is a synchronous dataflow language designed to implement safety-critical embedded software. In addition to writing executable programs, the language doubles as a program logic, used for writing specification as synchronous observers or assume-guarantee contracts that specify properties of these programs. These specifications may be used during testing or proved exhaustively using model-checking tools. We taught a course on Lustre to last year engineering students. To streamline the learning experience and avoid technical issues, we developped an online application, Learn-Heptagon, which allows for writing, simulating, and proving properties of Lustre programs. This paper presents the application and the associated lesson plan. |
| 2026-06-01 | [Train, Test, Re-evaluate: Schedule-Sensitive Evaluation of Generative Data for Hand Detection](http://arxiv.org/abs/2606.01896v1) | Atmika Bhardwaj, Silvia Vock et al. | Generated (or synthetic) image data is increasingly used to augment or replace real training datasets when target imagery is scarce, expensive, or biased. For hand detection, particularly in occupational safety settings, public datasets mostly contain bare hands. This under-represents the variation in hand appearance introduced by gloves, tattoos, jewelry, and other personal protective equipment, creating a distribution shift that safety-critical applications encounter at deployment. We test whether generative inpainting, editing only the hand region of a real photograph to introduce accessories, can close this shift gap. On a paired dataset of real images and their synthetic counterparts, we train YOLOv8n hand detectors under six training-and-scheduling regimes (Experiments A-F, three random seeds each), evaluate every detector on a real test set and on a real-gloves-only test split, and report the mean average precision (mAP) at two overlap thresholds (mAP@0.5 and mAP@0.5:0.95) along with paired statistical tests. A two-stage experiment: train on real U synthetic data, then fine-tune the resulting weights on real-only at a lower learning rate, increases mAP@0.5 compared to the real-only baseline model on the standard real test set, and improves the real-gloves out-of-distribution gap. Another three-stage experiment preserves box-tightness best, reaching the highest mAP@0.5:0.95 of any other experiment in the study. The synthetic-data utility for safety-critical hand detection is determined by the training procedure, and simple multi-stage experiments extract substantial real-deployment benefit from inpainted accessory data. |
| 2026-06-01 | [Collaborative Space Object Detection with Multi-Satellite Viewpoints in LEO Constellations](http://arxiv.org/abs/2606.01895v1) | Xingyu Qu, Wenxuan Zhang et al. | With the growing number of satellites in low Earth orbit (LEO) constellations, the near-Earth space environment has become increasingly congested, making space object detection (SOD) a pressing challenge for space safety and sustainability. To mitigate collision risks and ensure the continuity of space operations, SOD systems must deliver fast and accurate detection under stringent onboard constraints. In this paper, we investigate the potential of multi-viewpoint observation fusion within a deep learning (DL) framework to enhance SOD performance. We design a practical multi-view pipeline and several input representations for feeding multi-view data into YOLO-based detectors. Our experiments show that using multi-view inputs is feasible in most cases and typically produces better results for mAP50 and mAP50-95. For example, in model YOLOv9-m, single-view compared to a three-view fused RGB setting, mAP50 increases from 0.638 to 0.732, while mAP50-95 improves from 0.227 to 0.276. Compared with the single-view setting, the best three-view grayscale configuration improves mAP50 by 36.3% and mAP50-95 by 46.5%. These findings establish multi-view fusion as a viable and effective strategy for SOD, with broad implications for space situational awareness in LEO constellation deployments. |

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



