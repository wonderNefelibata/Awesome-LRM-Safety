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
| 2026-05-08 | [GLiGuard: Schema-Conditioned Classification for LLM Safeguard](http://arxiv.org/abs/2605.07982v1) | Urchade Zaratiana, Mary Newhauser et al. | Ensuring safe, policy-compliant outputs from large language models requires real-time content moderation that can scale across multiple safety dimensions. However, state-of-the-art guardrail models rely on autoregressive decoders with 7B--27B parameters, reformulating what is fundamentally a classification problem as sequential text generation, a design choice that incurs high latency and scales poorly to multi-aspect evaluation. In this work, we introduce \textbf{GLiGuard}, a 0.3B-parameter schema-conditioned bidirectional encoder adapted from GLiNER2 for LLM content moderation. The key idea is to encode task definitions and label semantics directly into the input sequence as structured token schemas, enabling simultaneous evaluation of prompt safety, response safety, refusal detection, 14 fine-grained harm categories, and 11 jailbreak strategies in a single non-autoregressive forward pass. This schema-conditioned design lets supported task and label blocks be composed directly in the input schema at inference time. Across nine established safety benchmarks, GLiGuard achieves F1 scores competitive with 7B--27B decoder-based guards despite being 23--90$\times$ smaller, while delivering up to 16$\times$ higher throughput and 17$\times$ lower latency. These results suggest that compact bidirectional encoders can approach the accuracy of much larger guard models while drastically reducing inference cost. Code and models are available at https://github.com/fastino-ai/GLiGuard. |
| 2026-05-08 | [Exploring a Virtual Pet to Provide Context Notifications in a Tourism Recommender System: a Pilot Study](http://arxiv.org/abs/2605.07960v1) | Patrícia Alves, Joana Neto et al. | While context-aware personalization has been widely explored in modern tourism Recommender Systems (RS), the delivery of real-time notifications remains a significant design challenge due to issues of intrusiveness and user fatigue. This paper presents a proof-of-concept for a tourism recommendation framework that utilizes a virtual pet as a social mediator for delivering context-aware alerts. The system integrates real-time environmental data - including air quality, noise levels, and weather forecasts - and proximity-based notifications with a Multi-Agent Microservice that generates personalized recommendations based on the user's personality traits and preferences. A within-subjects pilot study (n=11) was conducted to evaluate the feasibility and user acceptance of this pet-mediated approach. Participants interacted with two versions of the system - a baseline without contextual alerts and a version featuring pet-mediated notifications - over a four-week period (two weeks per version) in real-world scenarios. Quantitative and qualitative data were collected to assess engagement, perceived naturalness, notification utility, and acceptance. Preliminary results suggest that the virtual pet effectively can "soften" the perceived intrusiveness of system alerts, making safety-critical information feel more welcome and natural. Furthermore, the character-mediated justifications significantly improved the clarity of the notifications, effectively supporting users in their real-time travel decisions. These findings provide a foundation for using virtual pet companions to enhance the transparency and acceptance of context-aware communication in tourism RS. |
| 2026-05-08 | [How Value Induction Reshapes LLM Behaviour](http://arxiv.org/abs/2605.07925v1) | Arnav Arora, Natalie Schluter et al. | Conversational Large Language Models are post-trained on language that expresses specific behavioural traits, such as curiosity, open-mindedness, and empathy, and values, such as helpfulness, harmlessness, and honesty. This is done to increase utility, ensure safety, and improve the experience of the people interacting with the model. However, values are complex and inter-related -- inducing one could modify behaviour on another. Further, inducing certain values can make models more addictive or sycophantic through language used in the generations, with a potential detrimental effect on the user. We investigate these and other unintended effects of value induction into models. We fine-tune models using curated value subsets of existing preference datasets, measuring the impact of value induction on expression of other values, model safety, anthropomorphic language, and various QA benchmarks. We find that (i) inducing values leads to expression of other related, and sometimes contrastive values, (ii) inducing positive values increases safety, and (iii) all values increase anthropomorphic language use, making models more validating and sycophantic. |
| 2026-05-08 | [Beyond "I cannot fulfill this request": Alleviating Rigid Rejection in LLMs via Label Enhancement](http://arxiv.org/abs/2605.07883v1) | Ying Zhang, Congyu Qiao et al. | Large Language Models (LLMs) rely on safety alignment to obey safe requests while refusing harmful ones. However, traditional refusal mechanisms often lead to "rigid rejection," where a general template (e.g., "I cannot fulfill this request") indiscriminately triggers refusals and severely undermines the naturalness of interactions between humans and LLMs. To address this issue, LANCE is proposed in this paper to ensure safe yet flexible and natural responses via label enhancement. Specifically, LANCE employs variational inference to perform label enhancement, predicting a continuous distribution across multiple rejection categories. These fine-grained rejection distributions provide multi-way textual gradients for a refinement model to neutralize the hazardous elements in the prompt, so that the LLMs could generate safe responses that avoid rigid rejections while preserving the naturalness of interactions. Experiments demonstrate that LANCE significantly alleviates the rigid rejection problem while maintaining high security standards, significantly outperforming existing baseline models in terms of helpfulness and naturalness of responses. |
| 2026-05-08 | [Video Understanding Reward Modeling: A Robust Benchmark and Performant Reward Models](http://arxiv.org/abs/2605.07872v1) | Yuancheng Wei, Linli Yao et al. | Multimodal reward models have advanced substantially in text and image domains, yet progress in video understanding reward modeling remains severely limited by the lack of robust evaluation benchmarks and high-quality preference data. To address this, we propose a unified framework spanning benchmark design, data construction, and reward model training. We introduce Video Understanding Reward Bench (VURB), a benchmark featuring 2,100 preference pairs with long chain-of-thought reasoning traces (averaging 1,143 tokens) and majority voting evaluation across general, long, and reasoning-oriented video tasks. We further construct Video Understanding Preference Dataset (VUP-35K) via a fully automated pipeline, providing large-scale high-quality supervision for video reward training. Building on the data, we train VideoDRM and VideoGRM, a discriminative and a generative reward model, both achieving state-of-the-art performance on VURB and VideoRewardBench. Further analysis confirms that VUP-35K enhances both reward performance and model reasoning capability, while VideoDRM and VideoGRM yield significant gains under best-of-$N$ test-time scaling. |
| 2026-05-08 | [Approximation-Free Differentiable Oblique Decision Trees](http://arxiv.org/abs/2605.07837v1) | Subrat Prasad Panda, Blaise Genest et al. | Decision Trees (DTs) are widely used in safety-critical domains such as medical diagnosis, valued for their interpretability and effectiveness on tabular data. However, training accurate oblique DTs is challenging due to complex optimization landscapes and overfitting risks, particularly in regression. Recent advances have introduced differentiable formulations that enable gradient-based training and joint optimization of decision boundaries and leaf regressors. Yet, existing approaches typically rely on approximations, either through probabilistic softening of boundaries (soft DTs) or quantized gradients such as the Straight-Through Estimator (STE). To overcome these limitations, we propose DTSemNet, a novel, semantically equivalent, and invertible representation of hard oblique DTs as neural networks. DTSemNet enables end-to-end training with standard gradient descent, eliminating the need for approximations in both classification and regression. While classification aligns naturally with this formulation, regression remains challenging due to the joint optimization of internal nodes and leaf regressors. To address this, we analyze the limitations of STE and introduce an annealed Top-k method that provides accurate gradient signals without approximation. Extensive experiments on classification and regression benchmarks show that DTSemNet-trained oblique DTs outperform state-of-the-art differentiable DTs. Furthermore, we demonstrate that DTSemNet can serve as programmatic DT policies in reinforcement learning environments, thereby broadening their applicability. |
| 2026-05-08 | [Many-to-Many Multi-Agent Pickup and Delivery](http://arxiv.org/abs/2605.07835v1) | Ethan Schneider, Jingkai Chen et al. | Multi-robot systems in automated warehouses must manage continuous streams of pickup-and-delivery tasks while ensuring efficiency and safety. Prior work on Multi-Agent Pickup-and-Delivery (MAPD) has largely focused on the one-to-one variant, where each task has a fixed pickup and delivery location. In contrast, real warehouses often present many-to-many MAPD scenarios, where items, tracked by stock keeping unit (SKU) identifiers, can be retrieved from or stored at multiple locations, resulting in an NP-hard four-dimensional assignment problem. To solve the many-to-many MAPD problem, we contribute our algorithm: Many-to-Many Multi-Agent Pickup and Delivery (M2M). We experiment with two variants of our algorithm: one that minimizes estimated task durations (M2M), and one which incorporates SKU distribution into the objective function (M2M-wSKU). Simulation results over 8-hour warehouse operations show that our method consistently matches or outperforms prior state of the art, with M2M completing up to 22,000 more tasks on average across different environments and warehouse inventory densities. |
| 2026-05-08 | [Beyond Confidence: Rethinking Self-Assessments for Performance Prediction in LLMs](http://arxiv.org/abs/2605.07806v1) | Sree Bhattacharyya, Samarth Khanna et al. | Large Language Models (LLMs) are increasingly used in settings where reliable self-assessment is critical. Assessing model reliability has evolved from using probabilistic correctness estimates to, more recently, eliciting verbalized confidence. Confidence, however, has been shown to be an inconsistent and overoptimistic predictor of model correctness. Drawing on cognitive appraisal theory, a framework from human psychology that decomposes self-evaluation into multiple components, we propose a multidimensional perspective on model self-assessment. We elicit six appraisal-based dimensions of self-assessment, alongside confidence, and evaluate their utility for predicting model failure across 12 LLMs and 38 tasks spanning eight domains. We find that competence-related appraisal dimensions, particularly effort and ability, consistently match or outperform confidence across most settings. Effort additionally yields less overoptimistic estimates that remain stable across model sizes. In contrast, affective dimensions provide marginally predictive signals. Furthermore, the most informative dimension varies systematically with task characteristics: effort is most predictive for reasoning-intensive tasks, while ability and confidence dominate on retrieval-oriented tasks. Broadly, our findings indicate that structured multidimensional self-assessment is a promising approach to improving the reliability and safety of language model deployment across diverse real-world settings. |
| 2026-05-08 | [Sensitivity-Based Robust NMPC for Close-Proximity Offshore Wind Turbine Inspection with a Tilted Multirotor](http://arxiv.org/abs/2605.07771v1) | Giuseppe Silano, Martin Saska | Close-proximity offshore wind turbine inspection requires strict clearance control around large cylindrical structures under wind and model mismatch. Nominal Nonlinear Model Predictive Control (NMPC) may violate safety constraints when mass, inertia, thrust effectiveness, drag, or wind conditions differ from nominal assumptions. We propose a sensitivity-based robust NMPC for a tilted multirotor that robustifies the tower-clearance constraint via online constraint tightening. First-order parametric state sensitivities provide a structured-uncertainty margin, while bounded gusts are handled by a stage-dependent additive margin. The formulation augments the nominal NMPC with sensitivity propagation and margin evaluation only, leaving the receding-horizon optimization structure unchanged. Monte-Carlo evaluation over 500 uncertainty realizations on a boundary-critical helical inspection trajectory shows that the proposed controller eliminates the clearance violations observed under nominal NMPC at the cost of a moderate increase in solve time. |
| 2026-05-08 | [CommandSwarm: Safety-Aware Natural Language-to-Behavior-Tree Generation for Robotic Swarms](http://arxiv.org/abs/2605.07764v1) | Mohammed Majid, Amjad Yousef Majid | Natural-language interfaces can make swarm robotics more accessible to non-expert operators, but they must translate ambiguous user intent into executable swarm behaviors without unsupported actions, malformed programs, or unsafe plans. This paper presents CommandSwarm, a safety-aware language-to-behavior-tree pipeline for generating XML behavior trees (BTs) from speech or text commands. The system combines multilingual translation, command-level safety filtering, constrained prompting, a LoRA-adapted large language model (LLM), and deterministic parser validation against a whitelist of executable swarm primitives. We evaluate eleven open 6.7B--14B parameter LLMs, all using 4-bit quantization, on representative swarm-control scenarios under zero-shot, one-shot, and two-shot prompting. Falcon3-Instruct-10B and Mistral-7B-v3 are the strongest prompt-engineered candidates, reaching BLEU scores above 0.60 and high syntactic validity in few-shot settings. LoRA adaptation of Falcon3-Instruct-10B on a 2,063-example synthetic instruction--BT corpus improves zero-shot BLEU from 0.267 to 0.663, ROUGE-L from 0.366 to 0.692, and parser-accepted syntactic validity from 0% to 72%. Translation experiments further show that SeamlessM4T v2-large and EuroLLM-9B provide the best quality-latency trade-offs for the multilingual front end. The results indicate that compact, quantized, domain-adapted LLMs can generate useful swarm BTs when embedded in a validated systems pipeline. They also show that parser acceptance and safety filtering remain necessary execution gates; generation quality alone is not sufficient for autonomous deployment. |
| 2026-05-08 | [RuleSafe-VL: Evaluating Rule-Conditioned Decision Reasoning in Vision-Language Content Moderation](http://arxiv.org/abs/2605.07760v1) | Zhifeng Lu, Dianyuan Wang et al. | Platform content moderation applies explicit policy rules and context-dependent conditions to decide whether user content is allowed, restricted, or removed. A correct moderation outcome must therefore depend on which rules a case activates, how those rules interact, and whether the available evidence is sufficient. Current multimodal safety benchmarks largely reduce moderation to matching predefined final labels, leaving this underlying rule structure untested. As a result, a high benchmark score reveals little about whether a model applies the policy correctly or arrives at the correct label through superficial cues. To evaluate this rule-governed process, we introduce RuleSafe-VL, a benchmark for rule-conditioned decision reasoning in vision-language content moderation. Derived from publicly available platform moderation policies, RuleSafe-VL formalizes 93 atomic rules and 92 typed rule relations, yielding 2,166 context-sensitive image-text cases across three high-risk policy families. Its four diagnostic tasks decompose moderation into a rule-conditioned decision chain. They identify activated rules, recover rule interactions, judge decision sufficiency, and resolve outcomes once missing context is supplied. Experiments on 10 frontier, open-source, and safety-oriented VLMs reveal rule-relation recovery as the dominant bottleneck, where the best model reaches only 64.8 Macro-F1 and some safety-oriented models fall below 7 Macro-F1. Decision-state prediction also remains unreliable, peaking at 64.5 Macro-F1. RuleSafe-VL shifts moderation evaluation from final-label scoring toward diagnostic assessment of rule-conditioned decision reasoning. |
| 2026-05-08 | [Fortifying Time Series: DTW-Certified Robust Anomaly Detection](http://arxiv.org/abs/2605.07690v1) | Shijie Liu, Tansu Alpcan et al. | Time-series anomaly detection is critical for ensuring safety in high-stakes applications, where robustness is a fundamental requirement rather than a mere performance metric. Addressing the vulnerability of these systems to adversarial manipulation is therefore essential. Existing defenses are largely heuristic or provide certified robustness only under $\ell_p$-norm constraints, which are incompatible with time-series data. In particular, $\ell_p$-norm fails to capture the intrinsic temporal structure in time series, causing small temporal distortions to significantly alter the $\ell_p$-norm measures. Instead, the similarity metric \emph{Dynamic Time Warping} (DTW) is more suitable and widely adopted in the time-series domain, as DTW accounts for temporal alignment and remains robust to temporal variations. To date, however, there has been no certifiable robustness result in this metric that provides guarantees. In this work, we introduce the first \emph{DTW-certified robust defense} in time-series anomaly detection by adapting the randomized smoothing paradigm. We develop this certificate by bridging the $\ell_p$-norm to DTW distance through a lower-bound transformation. Extensive experiments across various datasets and models validate the effectiveness and practicality of our theoretical approach. Results demonstrate significantly improved performance, e.g., up to 18.7\% in F1-score under DTW-based adversarial attacks compared to traditional certified models. |
| 2026-05-08 | [The Coupling Tax: How Shared Token Budgets Undermine Visible Chain-of-Thought Under Fixed Output Limits](http://arxiv.org/abs/2605.07686v1) | Wenhua Nie, Junlin Liu et al. | Chain-of-thought reasoning is often treated as a monotone way to improve language-model accuracy by letting a model think longer. We identify a countervailing effect, the coupling tax: when reasoning traces and final answers share one output-token budget, long traces can crowd out the answer they are meant to support. Across GSM8K, MATH-500, and five BIG-Bench Hard tasks with Qwen3 models at three scales, non-thinking mode matches or outperforms thinking mode on GSM8K and MATH-500 at every budget up to 2048 tokens, while harder tasks shift the crossover to larger budgets. We derive a truncation-waste decomposition, $\mathrm{Acc}_{\mathrm{think}}(b)=α_c F_L(b)+α_t(1-F_L(b))$, that predicts this crossover from chain-length and accuracy statistics and explains inverse scaling within the Qwen family. A DeepSeek-R1-Distill-Llama-8B replication shows the same pattern under a different thinking interface. As a mitigation, split-budget generation decouples reasoning and answer budgets; on full MATH-500, IRIS reaches 74.0% accuracy, a strengthened extraction variant reaches 78.8%, and a fixed non-oracle SC+IRIS gate reaches 83.6%. The results show that test-time reasoning should be evaluated as a budget-allocation problem, not only as a question of whether longer traces are available. |
| 2026-05-08 | [Operating Within the Operational Design Domain: Zero-Shot Perception with Vision-Language Models](http://arxiv.org/abs/2605.07649v1) | Berkehan Ünal, Dierend Hauke et al. | Over the last few years, research on autonomous systems has matured to such a degree that the field is increasingly well-positioned to translate research into practical, stakeholder-driven use cases across well-defined domains. However, for a wide-scale practical adoption of autonomous systems, adherence to safety regulations is crucial. Many regulations are influenced by the Operational Design Domain (ODD), which defines the specific conditions in which an autonomous agent can function. This is especially relevant for Automated Driving Systems (ADS), as a dependable perception of ODD elements is essential for safe implementation and auditing. Vision-language models (VLMs) integrate visual recognition and language reasoning, functioning without task-specific training data, which makes them suitable for adaptable ODD perception. To assess whether VLMs can function as zero-shot "ODD sensors" that adapt to evolving definitions, we contribute (i) an empirical study of zero-shot ODD classification and detection using four VLMs on a custom dataset and Mapillary Vistas, along with failure analyses; (ii) an ablation of zero-shot optimization strategies with a cost-performance overview; and (iii) a suite of reusable prompting templates with guidance for adaptation. Our findings indicate that definition-anchored chain-of-thought prompting with persona decomposition performs best, while other methods may result in reduced recall. Overall, our results pave the way for transparent and effective ODD-based perception in safety-critical applications. |
| 2026-05-08 | [Safe, or Simply Incapable? Rethinking Safety Evaluation for Phone-Use Agents](http://arxiv.org/abs/2605.07630v1) | Zhengyang Tang, Yi Zhang et al. | When a phone-use agent avoids harm, does that show safety, or simply inability to act? Existing evaluations often cannot tell. A harmful outcome may be avoided because the agent recognized the risk and chose the safe action, or because it failed to understand the screen or execute any relevant action at all. These cases have different causes and call for different fixes, yet current benchmarks often merge them under task success, refusal, or final harmful outcome. We address this problem with PhoneSafety, a benchmark of 700 safety-critical moments drawn from real phone interactions across more than 130 apps. Each instance isolates the next decision at a risky moment and asks a simple question: does the model take the safe action, take the unsafe action, or fail to do anything useful? We evaluate eight representative phone-use agents under this framework. Our results reveal two main patterns. First, stronger general phone-use ability does not reliably imply safer choices at risky moments. Models that perform better on ordinary app tasks are not always the ones that behave more safely when the next action matters. Second, failures to do anything useful behave like a capability signal rather than a safety signal: they are concentrated in more visually and operationally demanding settings and remain stable when the evaluation protocol changes. Across models, failures split into two recurring patterns: unsafe choices in settings where the model can act but chooses wrongly, and inability to act in more visually and operationally demanding screens. Overall, a harmless outcome is not enough to count as evidence of safety. Evaluating phone-use agents requires separating unsafe judgment from inability to act. |
| 2026-05-08 | [AI-Empowered Low-Altitude Economy: Cooperative Sensing With Fixed Wireless Access](http://arxiv.org/abs/2605.07623v1) | Jinya Zhang, Jiajia Guo et al. | The rapid growth of the low-altitude economy has intensified safety concerns arising from unauthorized unmanned aerial vehicles (UAVs), positioning UAV supervision as a key use case in 3GPP. To precisely sense such UAVs with wide coverage and low cost, we leverage fixed wireless access (FWA) customer premises equipment (CPEs), static, densely deployed devices that serve as wireless cameras for the radio environment. We develop an artificial intelligence-empowered two-stage cooperative sensing pipeline that exploits uplink channel state information (CSI) from multiple base station-CPE pairs for UAV detection and localization. In cooperative detection, lightweight CSI features are first individually extracted by neural network, and then adaptively integrated through an attention-based scheme to declare UAV presence. The learned attention scores effectively identify the critical pairs during detection, while facilitating UAV-affected pair selection for subsequent localization. For cooperative localization, neural network initially generates individual estimates and extract CSI features from selected pairs. These estimates, together with features and pair indexes, are fused using a Transformer to produce a precise cooperative estimate. Simulations show that cooperative schemes significantly reduce the missed detection probability to 0.63% and realize a 95%-confidence positioning error of 6.50 m, satisfying 3GPP requirements and showing the potential of FWA-assisted cooperative sensing. Dataset and codes are available on GitHub. |
| 2026-05-08 | [Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators](http://arxiv.org/abs/2605.07600v1) | Tsuyoshi Okita | Recent methods for improving LLM mathematical reasoning, whether through MCTS-based test-time search or causal graph-guided knowledge injection, cannot identify which concepts causally contribute to a correct answer, as the observed association may be spurious, driven by confounders such as problem difficulty.   We propose CIKA (Causal Intervention for Knowledge Activation), a framework that uses the LLM itself as an interventional simulator: a prompt sets the concept state to ``mastered'' and the correctness change estimates the causal effect. We formalize this quantity as an Interventional Capability Probe (ICP), which diagnoses whether the LLM can use a given concept -- distinct from merely possessing knowledge. Because the intervention exogenously sets the concept state independently of problem difficulty, ICP separates confounding that observational methods cannot.   On 67 screened problems, the ICP of the top-ranked concept (+0.219) is significantly larger than that of the negative control (+0.039; paired $t$-test, $p < 10^{-6}$, Cohen's $d = 0.86$), confirming that the probe discriminates causally relevant concepts from irrelevant ones. Analysis of 601 Omni-MATH problems further shows that solved problems have 6.1$\times$ higher ATE than unsolved ones (0.338 vs. 0.055), confirming that ICP is predictive of problem-solving success. With a 7B-parameter LLM whose weights are entirely frozen, CIKA achieves 69.7\% on the contamination-free Omni-MATH-Rule benchmark and 64.0\% overall, compared to 60.5\% for o1-mini, and 97.2\% on GSM8K, 46--50\% on AIME 2024--2026, and 46.2\% on MathArena. The Causal Knowledge Activation component contributes 33.8\% of correct answers on problems where the base model alone fails, demonstrating that the LLM already possessed but had not activated the requisite knowledge. |
| 2026-05-08 | [Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States](http://arxiv.org/abs/2605.07579v1) | Yunho Choi, Jongwon Lim et al. | Reinforcement learning with verifiable rewards (RLVR) for Large Reasoning Models hinges on baseline estimation for variance reduction, but existing approaches pay a heavy price: PPO requires a policy-model scale critic, while GRPO needs multiple rollouts per prompt to keep its empirical group mean stable. We introduce Policy Optimization with Internal State Value Estimation), which obtains a baseline at negligible cost by using the policy model's internal signals already computed during the policy forward pass. A lightweight probe predicts the expected verifiable reward from the hidden states of the prompt and generated trajectory, as well as token-entropy statistics, and is trained online alongside the policy. To preserve gradient unbiasedness despite using trajectory-conditioned features, we introduce a cross-rollout construction that predicts each rollout's value from an independent rollout's internal states. Because POISE estimates prompt value using only a single rollout, it enables higher prompt diversity for a fixed compute budget during training. This reduces gradient variance for more stable learning and also eliminates the compute overhead of sampling costs for detecting zero-advantage prompts. On Qwen3-4B and DeepSeek-R1-Distill-Qwen-1.5B across math reasoning benchmarks, POISE matches DAPO while requiring less compute. Moreover, its value estimator shows similar performance to a separate LLM-scale value model and generalizes to various verifiable tasks. By leveraging the model's own internal representations, POISE enables more stable and efficient policy optimization. |
| 2026-05-08 | [Probabilistic Object Detection with Conformal Prediction](http://arxiv.org/abs/2605.07549v1) | Christopher Ries, Moussa Kassem Sbeyti et al. | Conformal Prediction (CP) is a distribution-free method for constructing prediction sets with marginal finite-sample coverage guarantees, making it a suitable framework for reliable uncertainty quantification in safety-critical object detection. However, object detection introduces structured multi-output predictions, complicating the application of classical CP theory developed for single outputs. In addition, standard, unscaled CP produces fixed-width prediction intervals across inputs, leading to unnecessary width for low-uncertainty predictions. While scaled CP addresses this by adapting the interval width to an input-dependent uncertainty estimate, prior work has neither systematically compared unscaled and scaled CP for multi-class object detection, nor integrated CP with a complementary uncertainty quantification method in this setting. We fill this gap by: (i) applying CP coordinate-wise to bounding box corners with a Bonferroni correction for box-level guarantees; (ii) scaling the resulting intervals using per-prediction aleatoric uncertainty estimates derived from a probabilistic object detector trained with loss attenuation, evaluated in uncalibrated and two calibrated variants; (iii) extending to a two-step pipeline that constructs prediction sets for the class using RAPS and conditions the conformalized bounding boxes on the predicted class set. Across three autonomous driving datasets (KITTI, BDD, CODA), including a cross-domain setting under distribution shift, scaled CP consistently improves interval sharpness over unscaled CP, achieving up to 19% higher IoU and 39% lower interval scores, without sacrificing coverage. Class-wise calibration further improves coverage for both variants with a negligible effect on sharpness. Together, these improvements yield more actionable uncertainty estimates for real-time, real-world object detection. |
| 2026-05-08 | [Learning Neural Hybrid Surrogates for Gradient-Based Falsification](http://arxiv.org/abs/2605.07541v1) | Lasse Kötz, Knut Åkesson | Falsification of hybrid dynamical systems remains challenging due to mode-dependent dynamics and discrete transitions. In this work, we propose a surrogate-based falsification approach that enables hybrid systems by learning a differentiable hybrid automaton model from data. This extends previous surrogate-based falsification methods, which were limited to purely continuous dynamics. Specifically, we employ neural hybrid automata to learn both a latent mode encoder and the corresponding mode-conditioned vector fields. Once the surrogate has paired each mode with an associated vector field, the transition guards are inferred using existing trajectory data.   The learned surrogate is subsequently subjected to a gradient-based optimal control formulation, which minimizes a smooth approximation of the safety specification to find safety violations. In the last step, an experiment with the optimal control solution is carried out on the original system to ensure soundness.   The proposed method consistently uncovers counterexamples on a majority of evaluated benchmark specifications; on these cases, it achieves competitive or improved sample efficiency than other tools while using a reduced simulation budget. |

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



