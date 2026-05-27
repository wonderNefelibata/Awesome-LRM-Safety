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
| 2026-05-26 | [FinHarness: An Inline Lifecycle Safety Harness for Finance LLM Agents](http://arxiv.org/abs/2605.27333v1) | Haoxuan Jia, Yang Liu et al. | Finance LLM agents must simultaneously block prompt-induced unauthorized actions and approve legitimate multi-step business workflows. However, boundary filters often miss irreversible mid-trajectory tool calls, while post-hoc LLM judges perform auditing only after termination -- too late for intervention and at a computational cost that scales linearly with trace length. We present FinHarness, an inline safety harness that wraps a finance agent end-to-end with three components: a Query Monitor that fuses single-turn intent with cross-turn drift, a Tool Monitor that evaluates each prospective tool call, and a Cascade module that integrates per-step risk and adaptively routes verification between a lightweight and an advanced-tier LLM judge. Fired risk factors are re-injected into the agent input as ex-ante evidence, enabling the agent to refuse, re-plan, or approve on its own. On FinVault, routed FinHarness cuts ASR from 38.3% to 15.0% while largely preserving benign approval ($41.1\% \to 39.3\%$), and uses $4.7\times$ fewer advanced-judge calls than an always-advanced ablation. |
| 2026-05-26 | [Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs](http://arxiv.org/abs/2605.27255v1) | Wenhui Tan, Minghao Li et al. | Long chain-of-thought reasoning has made autoregressive decoding the dominant inference cost of modern large language models. Existing methods target either the input side (latent compression) or the output side (speculative decoding and multi-token prediction, MTP), but the two lines of work have been pursued independently. Moreover, output-side methods must incur an expensive verifier pass to validate the unreliable draft tokens predicted by MTP. To address these issues, we propose \textbf{Pair-In, Pair-Out (PIPO)}, which unifies both sides by viewing a latent compressor and an MTP head as mirror-image operations: the compressor folds two input tokens into one latent representation, while the MTP head unfolds one hidden state into one additional output token. To remove the verifier cost without sacrificing reliability, PIPO trains a lightweight confidence head that decides whether draft tokens should be accepted. We observe that On-Policy Distillation (OPD) naturally matches the rejection-sampling criterion of speculative decoding, so the confidence head can be trained alongside OPD with negligible extra cost. Experiments on AIME 2025, GPQA-Diamond, LiveCodeBench v6, and LongBench v2 with Qwen3.5-4B and 9B backbones show that PIPO improves pass@4 over regular decoding by up to $+7.15$ points, while delivering up to $2.64\times$ first-token-latency and $2.07\times$ per-token-latency speedups. |
| 2026-05-26 | [Detecting Is Not Resolving: The Monitoring Control Gap in Retrieval Augmented LLMs](http://arxiv.org/abs/2605.27157v1) | Zhe Yu, Wenpeng Xing et al. | Retrieval-augmented LLMs are deployed for tasks where evidence quality determines action safety, yet evaluation protocols assume that single-turn robustness predicts robustness when evidence accumulates across turns. We show this assumption is fundamentally incorrect. Models exhibit a monitoring-control gap: they readily acknowledge contradictory evidence, yet this awareness fails to constrain their final recommendations - detecting epistemic conflict does not imply resolving it safely. Through a multi-turn document accumulation protocol across four model families (1.5B-32B parameters) and over 50,000 turn-level evaluations, we demonstrate that single-turn diagnostics systematically overestimate RAG safety, that contradiction acknowledgement is uncorrelated with safe resolution, a pattern corroborated by targeted human validation, and that no universal prompt fix exists. Converging mechanism evidence - hidden-state probing, attention analysis, and response-strategy taxonomy - points to action selection as the most plausible locus of the deficit: danger-relevant information is internally represented and receives enhanced attention during unsafe generation, yet fails to constrain output behavior. The gap between what models recognize and what they do must be measured and closed before retrieval-augmented systems can be trusted in high-stakes settings. |
| 2026-05-26 | [Semantic Robustness Probing via Inpainting: An Interactive Tool for Safety-Critical Object Detection](http://arxiv.org/abs/2605.27155v1) | Nico Steckhan, Krutarth Prajapati et al. | Testing object detectors in safety-critical domains requires semantically meaningful probes beyond pixel-level corruptions. We present SemProbe, a tool for semantic robustness probing: users upload deployment images, create masks manually or automatically, select operational design domain-derived factors (or custom prompts), and run diffusion-based controlled inpainting. The system supports batch jobs, parallel seed/workflow variations, and configurable generation parameters. After each output, model inference runs automatically and displays annotated before/after comparisons with performance deltas. All probes are logged as structured artifacts, enabling traceable robustness evidence aligned with safety evaluation workflows. We demonstrate \textsc{SemProbe} on hand detection for dimension saws, targeting factors from insurance-oriented test criteria. |
| 2026-05-26 | [Position: AI Safety Requires Effective Controllability](http://arxiv.org/abs/2605.27117v1) | Yige Li, Yunhao Feng et al. | AI safety is still largely framed as alignment: training models to follow human preferences, safety policies, and normative constraints. That framing has improved the behavior of modern language models, but aligned behavior does not by itself guarantee that a deployed agent can be stopped, overridden, or constrained once it operates in open-ended, interactive, and tool-using environments. A system may be safe in expectation and still fail to yield to explicit runtime authority under conflicting instructions, long-horizon execution, adversarial inputs, or risky tool use. This position paper argues that AI safety therefore requires controllability as a first-class objective. We define \emph{controllability} as the ability of an AI system to remain reliably interruptible, overridable, redirectable, and constrainable by explicit control signals at runtime while preserving ordinary utility when such signals are absent. To study this gap, we introduce \controlbench{}, a benchmark for evaluating controllability failures in high-risk agentic scenarios. Experiments with OpenClaw-based agents show that current alignment and guardrail mechanisms reduce risk, but often fail to provide persistent, authoritative, and enforceable runtime control. We therefore propose a control-centric architectural framework that highlights explicit control planes, runtime intervention pathways, persistent control states, and auditable decision interfaces as key design principles for future controllable AI systems. |
| 2026-05-26 | [BAIT: Boundary-Guided Disclosure Escalation via Self-Conditioned Reasoning](http://arxiv.org/abs/2605.27110v1) | Xuan Luo, Yue Wang et al. | In this work, we propose BAIT (Boundary-Aware Iterative Trap), a three-step jailbreak framework that approaches malicious goals through internal disclosure. BAIT first asks the model to identify the protection boundary, then requires it to refine that boundary, and finally requests a detailed example. By expanding each step upon the model's previous responses, BAIT turns the model's own reasoning and consistency tendency into a disclosure pathway. Experiments on AdvBench, JailbreakBench, AIR-Bench, and SORRY-Bench demonstrate that BAIT consistently achieves strong attack success rates across top-tier large language models, significantly advancing conventional jailbreak baselines. Further analysis reveals that: 1) prevention-oriented framing significantly outperforms direct knowledge request; 2) the refinement step plays a critical role in disclosure escalation; and 3) the first two steps have a certain chance of eliciting harmful content while triggering little filtering. |
| 2026-05-26 | [Large Language Model-Powered Query-Driven Event Timeline Summarization in Industrial Search](http://arxiv.org/abs/2605.27066v1) | Mingyue Wang, Xingyu Xie et al. | Understanding how events evolve over time is essential for search engines handling queries about trending news. We present QDET (Query-Driven Event Timeline Summarization), a production system deployed on Baidu Search that constructs focused event timelines to explain specific query events. Unlike traditional topic-centric approaches that aim for comprehensive coverage, QDET identifies and organizes sub-events closely relevant to the query from noisy candidate sets formed by millions of documents retrieved daily. QDET incorporates two key innovations: (1) multi-task supervised fine-tuning with three auxiliary tasks-temporal ordering, causal judgment, and timeline completion-that enable compact models to match the performance of much larger general-purpose models in specialized domains; (2) reinforcement learning-based event concise summarization that enforces strict length constraints while maintaining semantic quality, achieving 88.2% length compliance and outperforming 671B-scale models by 7.7 points in constraint satisfaction. Our fine-tuned 7B parameter model achieves 76.2% F1 score on timeline summarization, slightly surpassing the zero-shot performance of DeepSeek-R1-671B (76.1% F1) while using only 1% of its parameters-demonstrating that domain-specific optimization enables production-ready models with comparable quality at drastically reduced computational costs. Online A/B tests on Baidu Search validate real-world effectiveness, showing 5.5% CTR improvement, 4.6% longer dwell time, and 4.4% deeper exploration compared to single-task baselines. We further demonstrate that timeline understanding transfers to heat prediction, confirming effective knowledge transfer to downstream tasks. |
| 2026-05-26 | [Learning to Balance Motor Thermal Safety and Quadrupedal Locomotion Performance with Residual Policy](http://arxiv.org/abs/2605.27046v1) | Yuhang Wan, Weixian Lin et al. | Motor thermal management is often overlooked in the context of electrically-actuated robots, particularly legged robots, but motor overheating is a key factor that limits long-duration locomotion especially under payload conditions. This paper integrates a whole-body thermal model of a quadruped robot into the reinforcement learning pipeline to update motor temperatures, and proposes a two-stage training framework for motor thermal management. In this framework, a nominal policy is first pre-trained as a locomotion baseline capable of traversing diverse terrains. A residual policy is then trained on top of the nominal policy to provide corrective actions based on the robot's thermal state, ensuring high performance under low-temperature conditions and preventing motor overheating under high-temperature conditions. Simulation results demonstrate that the proposed policy achieves an effective balance between motor thermal safety and locomotion performance. Real-world experiments on a Unitree A1 quadruped robot further validate the approach: under a 3 kg payload, the robot achieves stable locomotion across multiple terrains for over 13 minutes, while the nominal policy alone leads to motor overheating in about 5 minutes. |
| 2026-05-26 | [TPS-Drive: Task-Guided Representation Purification for VLM-based Autonomous Driving](http://arxiv.org/abs/2605.27038v1) | Jiaxiang Li, Yumao Liu et al. | Vision-Language Models (VLMs) provide a promising foundation for autonomous driving planning, yet bridging semantic reasoning and precise 3D spatial forecasting remains a critical challenge. Existing representation strategies generally follow two paths: text-aligned methods flatten continuous spatial states into symbols, which compromises geometric structure and induces "spatial hallucinations"; dense visual methods preserve spatial topology but overwhelm standard tokenizers with redundant background textures, leading to "representation interference". To address these limitations, we introduce TPS-Drive, a novel framework centered on Task-Guided Representation Purification that empowers VLMs to Think in Purified Space. At its core, an Agent-Centric Tokenizer utilizes a task-guided vector quantization mechanism supervised by a frozen 3D detection head, which explicitly reallocates limited codebook capacity from pervasive static backgrounds to critical dynamic agents and effectively isolates spatial redundancy. Leveraging this purified spatial vocabulary, TPS-Drive employs a decoupled reasoning pipeline that sequentially performs scene understanding, future forecasting, and action generation. The framework is optimized via a progressive three-stage training paradigm, culminating in reward-driven refinement that surpasses pure imitation learning. Extensive experiments validate our approach: TPS-Drive achieves accurate agent spatial state forecasting and reduces collision rates in open-loop nuScenes evaluations, while establishing new safety records on the rigorous closed-loop NAVSIMv1 and NAVSIMv2 benchmarks. |
| 2026-05-26 | [ReasonOps: A Unified Operational Paradigm for Trustworthy Verified LLM Reasoning](http://arxiv.org/abs/2605.27014v1) | Adnan Rashid | Large Language Models (LLMs) have transformed artificial intelligence from primarily generative systems into increasingly capable reasoning agents. Recent advances in theorem proving, autoformalization, symbolic reasoning, and tool-augmented language models demonstrate substantial progress toward machine-assisted formal reasoning. However, current reasoning systems still suffer from hidden logical inconsistencies, hallucinated symbolic transitions, unsupported theorem applications, and limited reliability guarantees. Existing approaches remain fragmented across formal verification, runtime assurance, neuro-symbolic reasoning and trustworthy Artificial Intelligence (AI) research communities.   This paper introduces ReasonOps, a unified operational paradigm for trustworthy verified reasoning systems. Inspired by operational ecosystems such as DevOps and MLOps, ReasonOps treats reasoning as a continuously monitored, verifiable, reliability-aware operational process rather than an isolated inference task. The proposed paradigm integrates semantic interpretation, autoformalization, symbolic reasoning, theorem proving, runtime assurance, probabilistic reliability estimation, and adaptive correction into a unified reasoning lifecycle. The paper further presents the ReasonOps architecture, demonstrates its workflow using an autonomous braking system analysis example, and discusses its potential role in future safety-critical autonomous AI systems. We argue that operational reasoning paradigms such as ReasonOps may become foundational infrastructure for next-generation trustworthy AI ecosystems. |
| 2026-05-26 | [Trust, Geometry, and Rules: A Credibility-Aware Reinforcement Learning Framework for Safe USV Navigation under Uncertainty](http://arxiv.org/abs/2605.26974v1) | Yuhang Zhang, Shuqi Chai et al. | Autonomous navigation of Unmanned Surface Vehicles (USVs) that is safe and compliant with the International Regulations for Preventing Collisions at Sea (COLREGs) remains a formidable challenge in dynamic maritime environments, particularly when perception systems exhibit miscalibrated uncertainty. Existing Reinforcement Learning (RL)-based methods often falter because state-estimation errors induce unreliable belief states that mislead the value function, while discrete traffic rules introduce discontinuity in the learning objective. To address these challenges, we propose a framework integrating credibility-aware learning, geometric safety shielding, and continuous rule-aware embedding. First, Credibility-Weighted Value Learning (CW-VL) introduces a dynamic trust factor derived from the discrepancy between filter-estimated covariance and empirical error statistics to modulate the critic's heteroscedastic loss, preventing policy overfitting to noisy samples. Second, the Covariance-Inflated Velocity Obstacle (CI-VO) maps position-estimation uncertainty into set-wise angular margins, forming a conservative geometric shield that overrides hazardous exploratory actions. Third, Risk-Aware COLREGs Duty Embedding relaxes binary encounter duties into continuous rule-aware signals, providing smooth sector-transition information and suppressing oscillation from sparse rule rewards. Simulated encounter studies demonstrate improved training robustness against perceptual inconsistency and superior collision avoidance and COLREGs compliance over baselines. |
| 2026-05-26 | [AlbanianLLMSafety: A Safety Evaluation Dataset for Large Language Models in Albanian](http://arxiv.org/abs/2605.26954v1) | Wajdi Zaghouani, Kholoud K. Aldous et al. | Safety evaluation of Large Language Models (LLMs) has largely focused on high-resource languages, leaving low-resource languages critically underserved. We present AlbanianLLMSafety, the first publicly available safety evaluation dataset for LLMs in Albanian, a linguistically distinct low-resource language with approximately 7.5 million speakers across Albania, Kosovo, North Macedonia, and the diaspora. The dataset contains 2,951 prompts spanning 11 safety categories, including self-harm, violence, racist content, child exploitation, and radicalization, with an average of 268 prompts per category. Each prompt is provided in Albanian with an English reference translation and a detailed category label. This resource addresses a significant gap in safety evaluation infrastruc-ture for low-resource languages and provides an essential benchmark for developing safer, more inclusive LLMs. The dataset will be provided upon request to support safety evaluation, fine-tuning, red-teaming, and guardrail development for Albanian-speaking communities. |
| 2026-05-26 | [KZ-SafetyPrompts: A Kazakh Safety Evaluation Prompt Dataset for Large Language Models](http://arxiv.org/abs/2605.26947v1) | Wajdi Zaghouani, Shimaa Amer Ibrahim et al. | Kazakh is underrepresented in resources for evaluating the safety behavior of large language models. We present KZ-SafetyPrompts, a Kazakh prompt dataset for safety evaluation across eleven categories covering common risk areas such as self-harm, violence, child exploitation, sexual content, racist content, radicalization, and regulated goods or illegal activities. The dataset contains 5,717 prompts written natively in Kazakh (Cyrillic), organized by category, with English translations for cross-lingual analysis. Prompts resemble realistic user queries, often in a teen or child style, and are phrased as intent prompts without procedural instructions. We document the writing protocol, labeling procedures (including borderline-case decision rules), and quality-control steps (schema standardization, completeness checks, and deduplication). We also align the categories with widely used safety taxonomies to support integration with existing evaluation pipelines. Baseline results with GPT-4o show an overall refusal rate of 28.2%, varying from 5.5% to 53.8% across categories, indicating that Kazakh prompts expose category-specific safety gaps not captured by English-only evaluation. |
| 2026-05-26 | [Neuro-Symbolic Verification of LLM Outputs for Data-Sensitive Domains (extended preprint)](http://arxiv.org/abs/2605.26942v1) | Paul Sigloch, Christoph Benzmüller | LLMs deployed in high-stakes domains face fundamental reliability challenges: hallucinations, inconsistencies, and privacy vulnerabilities introduce unacceptable risks where errors carry legal, financial, or safety consequences. This paper presents a hybrid verification architecture combining formal symbolic methods with neural semantic analysis to provide complementary guarantees for LLM-generated content. This architecture employs logical reasoning for input verification, leveraging completeness properties to provide decidable guarantees on structured requirements. For output validation, embedding-based semantic similarity detects contextual hallucinations where formal methods lack expressiveness. This separation is realized in a parallel, actor-based pipeline, addressing limitations of prompt-based self-verification approaches, which inherit the distributional biases that produce hallucinations. The proposed architecture and type-aware verification method are validated with HAIMEDA, a real-world medical device damage assessment reporting system developed through Action Design Research. Evaluation shows hallucination detection rates of over 83% for structured entities and 72% for semantic fabrications, with a 30% reduction in report creation time, demonstrating that neuro-symbolic architectures can provide principled safeguards for LLM deployment in data-sensitive domains. |
| 2026-05-26 | [Are Video Models Zero-Shot Learners and Reasoners in Education? EduVideoBench, A Knowledge-Skills-Attitude Benchmark for Educational Video Generation](http://arxiv.org/abs/2605.26918v1) | Unggi Lee, Hoyoung Ahn et al. | Video generation models (VGMs) are rapidly entering classrooms, yet existing benchmarks evaluate only perceptual quality, intrinsic faithfulness, generic safety, or video as a reasoning medium, and none assesses whether the outputs are educationally valid. In this work, we present EduVideoBench, the first balanced benchmark in the education domain, grounded in the Knowledge-Skills-Attitude (KSA) framework so that pedagogical adequacy and educational safety are evaluated jointly rather than as ad-hoc quality dimensions. Across five frontier VGMs, our results show substantial room for improvement across knowledge, skills, and attitude before they are classroom-ready. We complement this with a qualitative analysis of expert comments, finding that educational validity is multi-component, where a single misaligned element such as pacing, legibility, or notation can invalidate an otherwise correct video. We hope EduVideoBench will guide the development of VGMs that are pedagogically grounded and safe for the classroom. |
| 2026-05-26 | [Practical Anonymous Two-Party Gradient Boosting Decision Tree](http://arxiv.org/abs/2605.26903v1) | Huang Chenyu, Zhang Fan et al. | Structured data is well handled by gradient-boosted decision trees (GBDT), which are usually trained on vertically partitioned features across mutually distrustful parties. High speed and interpretability make GBDTs popular in finance and healthcare, where neural networks may fall short. Enabling secure computation for GBDTs poses unique challenges, requiring secure record alignment for comparison. Relying on private set intersection (PSI) is a de facto approach. Mistaking PSI for a safety measure actually exposes which record identifiers (IDs) are shared between the datasets. Although circuit-PSI could help, it is costly for generic uses. New ideas are needed to efficiently train in a "dark forest". Aiming to hide the IDs, we initiate the study of anonymous GBDT training on split data held by two parties. Dual circuit-PSI in our design lets the parties alternate as receiver to run pick-then-sum over local features. Via oblivious programmable pseudorandom functions, we propagate circuit-PSI outputs as shared state across runs. Avoiding universal alignment, we resolve the neglected dilemma that ID hiding incurs a cost that scales with domain size. Next, we halve the cost of ciphertext packing used to convert single-instruction multiple-data homomorphic encryption from (ring) learning with errors in prior secure GBDT (Usenix Security' 23) and related secure machine-learning computations. Comparative experiments show our protocol remains competitive with leaky approaches in efficiency. Enabling ID-hiding aggregation, our techniques can extend to other vertically partitioned analytics. |
| 2026-05-26 | [Persistent AI Agents in Academic Research: A Single-Investigator Implementation Case Study](http://arxiv.org/abs/2605.26870v1) | Anas H. Alzahrani | Background: Large language models are typically evaluated as models, benchmarks, or short conversational episodes. Less is known about what happens when an agent is embedded persistently in a real academic research environment with durable memory, local files, external tools, scheduled routines, delegated roles, and explicit safety protocols. Methods: A structured self-observed implementation case study was conducted from January 31 to May 25, 2026. The unit of analysis was the persistent human-agent environment: researcher, agent runtime, memory layer, tools, repositories, scheduled jobs, specialized agent roles, and governance rules. Outcomes were organized using PARE-M (Persistent Agentic Research Environment Measurement), a measurement framework covering architecture, utilization, artifact production, resource use, reproducibility, and governance. Results: Recoverable main-agent telemetry contained 75,671 de-duplicated records across 96 active days, with 8,059 user-role and 23,710 assistant-role messages. The workspace included 502 memory-related files, 17 configured agent directories, and 57 skill files. Active system time was 579.7 hours (30-minute capped-gap estimate). Memory-derived records identified 482 output-proxy events and 889 failure, verification, correction, or protocol-proxy events. A strict May 2026 trajectory subset captured 627 model-completed events and 73.95 million recorded tokens, of which 82.9% were cache reads. Conclusions: The workflow was cache-dominant, suggesting that persistent agentic environments may shift the economic unit from cost per token to cost per completed artifact. Future evaluations should use artifact-level denominators, reproducible parsing rules, correction taxonomies, and independent coding of governance events. |
| 2026-05-26 | [Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal](http://arxiv.org/abs/2605.26772v1) | Kia-Jüng Yang, Dominik Meier et al. | Large reasoning models (LRMs) generate chain-of-thought (CoT) traces before producing final outputs, introducing a dynamic internal state that may complicate control mechanisms such as refusal. Unlike instruction-tuned LLMs, where refusal is mediated by a single directional subspace, refusal in large reasoning models (LRMs) additionally depends on the CoT. In DeepSeek-R1-Distill-LLaMA-8B, activation steering reverses refusal in only 39% of cases when the CoT is kept fixed, but removing the CoT entirely increases this to 70%, indicating that the CoT actively reinforces refusal. In a two-stage intervention where the model regenerates its CoT under activation steering, refusal is reversed in 94% of cases, while the resulting CoT alone retains 48% of this effect even after steering is removed. This suggests that the CoT can carry and reconstruct the compliance signal independently. These findings indicate that refusal in LRMs is jointly encoded in residual stream activations and CoT. This joint activation makes LRM more robust against activation-level interventions alone, but exposes CoT to a possible alternative surface attack. |
| 2026-05-26 | [Measuring Prediction Uncertainty in Neural Cellular Automata](http://arxiv.org/abs/2605.26726v1) | Ario Sadafi, Michael Deutges et al. | Neural cellular automata (NCA) provide a lightweight alternative to encoder-decoder segmentation networks. However, it can be difficult to decide when a prediction should be trusted. Here, we study uncertainty estimation for NCA-based medical image segmentation without modifying the underlying architecture or retraining the model. Our approach is motivated by viewing the NCA as a dynamical system where convergent attractors correspond to confident predictions. Concretely, we propose resilience, a simple measure that leverages the intrinsic iterative structure of NCAs by probing the stability of the final prediction under small perturbations of the automaton state. Predictions that return to the same solution are deemed confident, while those that change substantially are flagged as uncertain. We evaluate uncertainty by its ability to predict segmentation quality using selective prediction metrics ($Δ$Dice@90 and AURC) and ranking metrics (AUROC and AUPRC). Across multiple medical segmentation benchmarks, resilience identifies failure cases more reliably than baselines, improving trust and safety in NCA-based models. |
| 2026-05-26 | [Look Further: Socially-Compliant Navigation System in Residential Buildings](http://arxiv.org/abs/2605.26710v1) | Akira Shiba, Marina Obata et al. | The distance at which a mobile robot reacts to a person strongly impacts various qualities of the human-robot interaction. In this paper, we focus on the navigation of a mobile delivery robot platform in a residential indoor hallway environment. Social navigation methods typically focus on avoiding uncomfortable human-robot interactions, such as when a robot encroaches on someone's personal space. Since personal space has been shown to be in the range of just a few meters, social navigation methods typically focus on deconflicting and resolving these short-range interactions. In this work, however, we demonstrate that by extending the reaction distance to over eight meters, far beyond the typical interaction distance, we can improve the human's perception of the robot's motion. We introduce the Proactive Lane-Changing (PLC) motion pattern and a navigation system that leverages it to react to people at an increased distance. This pattern consists of changing the robot's lateral position as it navigates down the hallway from the center to the side at an eight-meter distance from an oncoming person.   We conducted a user study with 42 participants to assess their impressions of the delivery robot based on three service objectives: safety, smoothness, and politeness. In the straight hallway scenario (Frontal Approach), results showed significant improvement in each of these three objectives compared to typical motion patterns found in the literature: slowing down, stopping, and reactive collision avoidance in the proximity of a person. In contrast, in the intersection (Blind Corner) scenarios, none of the approaches performed significantly better than any other, with participants having a diverse range of preferences among robot motion patterns. |

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



