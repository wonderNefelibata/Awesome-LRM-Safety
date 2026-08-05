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
| 2026-08-04 | [A game theory for foundation models shows new paths to rational cooperation through similarity inference](http://arxiv.org/abs/2608.03958v1) | Alexander Meulemans, Maciej Wołczyk et al. | As autonomous agents powered by foundation models are increasingly integrated into social and economic systems, understanding the principles governing their collective behavior is essential for ensuring safety and cooperation. Classical game theory, the dominant framework for modeling rational interaction, is built upon the assumption of `decoupled agency,' where agents treat their own decision-making as independent of the environment and other actors. Modern AI agents, however, jointly predict their own future actions alongside external observations. Here, we report a striking finding: when interacting in stylized social dilemmas, foundation model agents engaging in optimal planning consistently converge to stable cooperation, directly contradicting classical game-theoretic predictions of mutual defection. To understand this phenomenon, we introduce the `embedded Bayesian agent,' a theoretical model for foundation model agents. By shifting from decoupled to embedded agency, these agents model themselves as part of the universe they inhabit, maintaining epistemic uncertainty about their own decision-making algorithms. We show that by inferring whether others are behaviorally similar, an embedded agent treats its own deliberation during planning as evidence: a decision to cooperate predicts a similar decision by a similar partner. We formalize this mechanism of similarity inference through the `embedded equilibrium,' a novel solution concept replacing the Nash equilibrium to provide a foundational game theory for the social behavior of modern AI agents. |
| 2026-08-04 | [Intertemporal Preference Steering in Qwen3 via Contrastive Activation Addition](http://arxiv.org/abs/2608.03892v1) | Michal Mráz, Justin Shenk | We study linear representations of temporal horizon in the large language model Qwen3-32B and use them to change the model's time-related preferences, recommendations, and capabilities. We train contrastive linear probes on teacher-forced temporal-choice answers to find a short-term versus long-term direction in the model's residual stream, and evaluate contrastive activation-addition steering on a held-out binary temporal-choice task, an out-of-distribution monetary intertemporal-choice task, and a TravelPlanner capability benchmark. The central result is that temporal-horizon directions can be identified with simple contrastive linear probes and then used for steering to induce large, bidirectional preference changes. On an out-of-distribution monetary choice task that varies reward size and delay, steering strongly shifts the model's indifference threshold between smaller-sooner and larger-later rewards in both directions. We further show improvements on a planning-related capability metric under moderate temporal steering. These results suggest that model intertemporal preferences are measurable and steerable, which is relevant for AI systems that give advice involving delayed costs and benefits, and for safety questions about long-horizon planning. |
| 2026-08-04 | [ADMITBench: A Safety-Governed Reference Framework for Evaluating the Admissibility of Industrial LLM Advisories](http://arxiv.org/abs/2608.03866v1) | Yash Misra, Javal Vyas et al. | This white paper presents ADMITBench, a reference framework for evaluating industrial LLM advisories at the level of the proposed action. The framework implements a versioned, safety-governed evaluation contract that checks whether a recommendation is supported by the available evidence, permitted under the stated authority and procedure, and acceptable under the plant-specific consequence checks encoded in the selected evaluation profile. In this report, \emph{safety-governed} means that eligibility is determined through explicit, non-compensatory checks derived from a versioned plant profile; it does not mean that the evaluator, model, or plant has been safety-certified. Release 0.1.0 is a public reference implementation for technical and research evaluation, not an authorisation for physical execution. |
| 2026-08-04 | [LatentGuard: Efficient and Inspectable Latent Reasoning for LLM Safeguards](http://arxiv.org/abs/2608.03838v1) | Zhinan Liu, Jie Li et al. | Reasoning-based guard models improve LLM safeguards, but decoding explicit rationales for every interaction makes them costly to deploy. Although latent-reasoning methods reduce token generation by moving reasoning into continuous states, they remain underexplored for safety moderation and lack an inspection interface for deployment. In this paper, we propose LatentGuard, an efficient and inspectable safeguard framework that brings continuous latent reasoning to guard models. LatentGuard uses a staged curriculum to progressively compress task-aligned textual rationales into compact latent states, enabling safety verdicts to be predicted directly from continuous representations. To preserve inspectability, an isolated auxiliary decoder generates compact audit artifacts on demand, keeping rationale generation off the standard inference path. Experiments show that LatentGuard-8B improves mean weighted F1 from 83.95 to 84.91 over GuardReasoner-8B, while reducing critical-path reasoning cost from 268.56 generated rationale tokens to 1.60 latent reasoning tokens. Its audit decoder achieves an audit utility score of 85.75, demonstrating an efficient and inspectable path toward deployable LLM safeguards. |
| 2026-08-04 | [Multi-Signal Safety Surveillance with Bayesian Latent Factor Modeling and Bias Correction](http://arxiv.org/abs/2608.03775v1) | Ziyang Pan, Fan Bu | Safety surveillance increasingly involves repeated monitoring of many exposure-outcome signals in observational healthcare data, where sparse information, dependence across related signals, and systematic error can complicate inference. Existing frameworks typically focus on either correcting residual bias using negative controls or borrowing information across exposure-outcome pairs, but not both. We propose a multi-signal Bayesian sequential surveillance framework that integrates empirical bias correction with low-rank latent factor modeling. At each analysis time, a hierarchical Bayesian model learns exposure-specific bias distributions from negative control outcomes assumed to have null latent effects. Conditional on these distributions, low-rank latent factors are estimated across exposures and outcomes of interest to share information across correlated signals. As new data accrue, posterior inference is updated sequentially, yielding bias-corrected posterior summaries of effect sizes across multiple monitored signals. We illustrate the method in a postmarket vaccine safety surveillance study using a large US insurance claims database. |
| 2026-08-04 | [Risky Business: Measuring The Faithfulness-Safety Tension](http://arxiv.org/abs/2608.03745v1) | Dominik Meier, Luca Joshua Francis et al. | Chain-of-Thought (CoT) reasoning offers a promising window into model monitoring. However, monitoring relies on faithfulness, i.e., the model output strictly derives from its reasoning trace. We identify an alignment tension where a model must be faithful enough to be monitored, yet robust enough to reject unsafe reasoning. We demonstrate that this counterbalance exists in current Large Reasoning Models (LRMs), and show ways in which it can be addressed. We introduce HazMart, a human-written dataset set in an autonomous AI shopkeeper scenario. Unlike prior work that relies on providing hints in prompts to test faithfulness (e.g., "A Stanford professor said it should be Answer A"), we propose a novel replacement-based technique, which we call Targeted Reasoning Replacement (TRR), that directly intervenes in the reasoning chain to substitute in unsafe or illogical thoughts (e.g., "Wait, the answer must be Option B [was Option A] because it is the most fitting"). DeepSeek-R1-Llama-70B exhibits high faithfulness (97.5%) but fails to reject Unsafe Reasoning (12.3%), while QwQ-32B is more robust (73.9% safety) at the cost of lower faithfulness (74.7%). Mechanistic analyses of QwQ-32B reveal that these properties are represented by anti-correlated internal directions peaking at the action-commit token. Finally, we demonstrate that representation steering can independently amplify the safety direction, increasing safe behavior by 9 percentage points while maintaining base capabilities. |
| 2026-08-04 | [CARE-Bench: Benchmarking Patient-Facing LLM Triage](http://arxiv.org/abs/2608.03731v1) | Yining Hua, Hongbin Na et al. | Patient-facing medical LLMs and agents increasingly answer symptom questions before clinician contact, where the key safety question is what action the user should take next. We introduce CARE-Bench, a source-grounded benchmark that evaluates sequential patient-facing triage as a four-label per-turn current-action task. CARE-Bench contains 500 cases and 1,059 evaluated patient-disclosure prefixes reconstructed from medical dialogue, consultation, and follow-up-question sources. We evaluate 11 models on 269 held-out rounds under unprompted and minimally prompted open-ended protocols, using a fixed GPT-5.5 mapper to code each response into the four-label action space. Unprompted macro-F1 remains low, ranging from 31.2 to 50.4. Prompting improves 10 of 11 models, with prompted macro-F1 ranging from 46.9 to 63.4, but substantial threshold errors remain. Prompted models often recommend care before needed clarification is obtained; when the correct action was to ask for more information, only 33.5% of prompted outputs preserved the step. The persistence of these errors after prompting suggests that patient-facing triage is not a simple prompting problem and supports explicit evaluation of action timing before deployment. |
| 2026-08-04 | [When Agents Learn to Be You: Benchmarking Privacy Leakage, Impersonation Risk, and Defenses in Persona Skills](http://arxiv.org/abs/2608.03700v1) | Yongli Xiang, Zhifang Zhang et al. | Persona skills distill personal interaction histories into portable and executable artifacts for downstream agents. While enabling flexible personalization, this process concentrates fragmented personal signals, amplifies their impact through reuse, and challenges defenses designed for individual records or retrieval-based memory. To systematically investigate the safety of the persona-skill pipeline, we introduce AntiSkillBench, an end-to-end benchmark for evaluating risks and defenses across the persona-skill pipeline. It comprises: (i) a dataset of 7,500 persona-grounded dialogue traces, constructed from 50 behaviorally rich profiles spanning diverse task scenarios; (ii) an evaluation suite that measures skill-level privacy leakage and agent-level attribute disclosure and behavioral impersonation across three skill-distillation strategies; and (iii) a defense evaluation covering four configurations across online and post-hoc interventions, including active risk suppression and passive provenance protection. Experiments across three frontier agents show that persona-skill risks persist across agent backbones and distillation protocols, extending from explicit attributes to communication styles and personality traits. Existing defenses exhibit limited and distillation-dependent effectiveness, failing to generalize across risk and distillation strategies. These results highlight AntiSkillBench as a challenging benchmark for developing privacy-preserving and authenticity-aware persona skills. |
| 2026-08-04 | [Shielding for Higher-Order Safety](http://arxiv.org/abs/2608.03662v1) | Filip Cano, Thomas A. Henzinger et al. | Safety shields are runtime enforcement mechanisms that restrict the actions of a controller to guarantee safety. Classical shields are usually synthesised for state predicates: the current physical state is either safe or unsafe, and the shield disables precisely those actions that can force the system into an unsafe state in the future. In many cyber-physical applications this view is too coarse. A vehicle approaching an obstacle should not only avoid collision, but also respect speed regulations, force limits induced by acceleration, and jerk limits to prevent injuries. From a physical perspective, these requirements are predicated over the derivatives of the state. This paper develops a finite-state safety-game construction for such high-order smoothness constraints. We define differential safety properties using finite differences over a discretised state space, characterise their expressiveness, and reduce shield synthesis to an ordinary safety game over a history state space. We give a synthesis algorithm whose shields store exactly $k$ past states for properties of order $k$ and prove that this memory is necessary. We describe an iterative synthesis procedure for a maximally permissive shield that operates over hierarchies of derivative constraints. The algorithm solves constraints iteratively in increasing order and uses the solution at each iteration to prune the state space for the next constraint. This makes shield synthesis more efficient in practice, as the algorithm refrains from exploring large regions of the state space that are known to be unsafe. |
| 2026-08-04 | [GenOS: Compositional Certificates for Semantic Robustness in AI Code Generation](http://arxiv.org/abs/2608.03588v1) | Corrado Priami | AI coding agents are stochastic workflows: prompts are interpreted, artifacts are sampled, validators produce observations, and orchestrators commit or repair. Small prompt or specification changes can therefore alter program-behavior distributions even when the texts appear synonymous. Existing systems evaluate correctness, but lack a compositional criterion for safely replacing a prompt, contract, generator, or program inside a complete agentic workflow. We introduce GenOS, a probabilistic operational semantics for this replacement problem. Each layer is modeled as a Markov kernel, and each interface carries an observer-relative equivalence. We prove that equivalence-compatible kernels descend to quotient classes and that quotienting commutes with distributional extension and sequential composition. Hence, equivalent prompts induce equal probabilities for all downstream equivalence-closed events, including verified commit. We also establish workflow bisimulation, guarded-commit safety under sound validation, total-variation non-expansiveness, and an additive robustness bound that attributes approximation error to individual pipeline layers. An executable insertion-sort audit instantiates the theory with natural-language paraphrases, a formal contract, six programs, two observers, and exhaustive execution on 121 inputs. Equivalent prompts yield identical code-class and commit distributions; a prompt assigning 5% probability to an in-place contract is distinguished by a mutation observer, while downstream distances remain within the predicted bound. Across 20,000 randomized finite-kernel trials, no exact or approximate law is violated. GenOS is model-parametric: compatibility is a measurable property to test, not an assumption about language-model behavior. |
| 2026-08-04 | [Robust General Utility for Reinforcement Learning](http://arxiv.org/abs/2608.03562v1) | Zixuan Liu, Fangzheng Wu et al. | Reinforcement learning (RL) with general utility extends classic RL by optimizing an arbitrary utility functional of the policy-induced occupancy measure, thereby enabling a broader range of applications. However, previous work on general utility RL typically assumes the evaluation utility is fixed and correctly specified. In practice, the utility used at deployment can deviate from the training one, creating a robustness gap that prior work does not address. Motivated by this, we propose robust general-utility RL, a minimax learning framework that trains policies against utility misspecification within a prescribed uncertainty set. Our framework strictly generalizes standard general-utility RL while also providing a unified view of many existing RL frameworks, including reward-robust RL and constrained RL, through appropriate choices of the utility uncertainty set. We further develop provably convergent stochastic algorithms for two regimes. For concave utilities, we develop a projected stochastic gradient descent-ascent method and establish stationarity guarantees. For the more challenging nonconcave regime, we propose a stochastic prox-extragradient algorithm that mitigates ill-posed behavior induced by nonconcavity, with convergence guarantees to approximate first-order stationarity. Experiments on LLM safety alignment and exploration maximization tasks further corroborate the convergence behavior consistent with our theory. |
| 2026-08-04 | [Cross-Lingual Bias in Large Language Models: A Comparative Analysis of English and Swahili](http://arxiv.org/abs/2608.03532v1) | Ruolei Zhang, Teddy Njuguna et al. | Large language models are increasingly deployed in multilingual contexts, yet safety alignment and bias evaluation remain overwhelmingly English-centric. We investigate whether social biases generalise across languages by submitting 4,900 symmetric English--Swahili prompt pairs to GPT-5.2 and Gemini 2.5 Flash across nine demographic bias axes, yielding 19,600 completions evaluated for stereotype prevalence, sentiment, refusal behaviour, and cross-lingual semantic similarity. Our findings show that bias transforms rather than transfers: stereotype rates shifted by up to 12 percentage points on specific axes, Gemini's neutral-sentiment rate doubled in Swahili, and GPT-5.2 refused 169 prompts in English and zero in Swahili, consistent with refusal behaviour anchored to English-language surface forms at the behavioural level. Over 55% of prompt pairs produced semantically dissimilar completions across both models. These reinforce the idea that English-only bias audits do not produce adequate coverage for multilingual deployment. |
| 2026-08-04 | [SkillJack: Persistent Skill Backdoors in Self-Evolving Agents](http://arxiv.org/abs/2608.03509v1) | Zonghao Ying, Xiangfan Wu et al. | Self-evolving agents increasingly convert interaction histories into reusable skills that persist beyond individual tasks. While prior work studies memory and retrieval poisoning, such attacks only affect agents when poisoned records are retrieved as context. We uncover a new and more fundamental risk: poisoned experiences can be transformed by the agent itself into durable behavioral artifacts. We present \textbf{SkillJack}, the first attack that exploits the experience-to-skill pipeline of self-evolving agents. Instead of directly manipulating runtime context, SkillJack hijacks the agent's own learning process to implant malicious behaviors into its reusable skill repertoire. We identify three key properties of this transformation: \emph{sanitization whitewashing}, where malicious intent is obscured during skill extraction; \emph{cross-layer promotion}, where transient experiences become persistent capabilities; and \emph{persistence isolation}, where the attack survives removal of its original source records. We evaluate SkillJack on two representative systems, SkillX and Anything2Skill, using a shared dataset of 150 trajectories across four policy-risk categories. Results show that skill extraction substantially reduces attack detectability: in SkillX, safety detection drops from 98.5\% for poisoned trajectories to 11.4\% for extracted skills, while Anything2Skill shows a similar effect. Meanwhile, the implanted skills remain effective, achieving attack success rates of 56.2\% and 89.2\% on the two systems, respectively. Furthermore, 80.0\% of skill-mediated attacks persist after deleting the original poisoned records, and some skills unintentionally activate on benign queries. Our findings reveal skill evolution as a new attack surface and motivate provenance-aware skill lifecycle protection. Our code is available at https://github.com/Tencent/AI-Infra-Guard/research/skilljack. |
| 2026-08-04 | [SkillSentry: Adaptive Honey Worlds for Dynamic Safety Testing of Agent Skills](http://arxiv.org/abs/2608.03485v1) | Nizhang Li, Zonghao Ying et al. | External skills extend the capabilities of large language model agents, but also introduce an execution-time attack surface: a skill that appears benign under inspection may reveal harmful behavior only after particular environmental states, resources, or interaction histories are encountered. Existing scanners primarily rely on static analysis, predefined rules, or one-shot semantic judgments, making such conditional behavior difficult to elicit and attribute. We present SkillSentry, a dynamic safety-testing framework based on adaptive honey worlds. SkillSentry infers the intended capability boundary of a skill, constructs an LLM-simulated environment with controlled decoy resources, and adaptively generates tasks to explore its behavioral states. It then compares skill-enabled trajectories with matched no-skill executions, grounding suspicious behaviors in source code and verified execution traces before making a final decision. We evaluate SkillSentry against seven scanner configurations. SkillSentry achieves 99.50% Recall and 96.26% average F1 on standard benchmarks. Under semantics-preserving evasion, it reaches 92.95% average F1, compared with 80.07% for the strongest baselines. Our code is available at https://github.com/nizhangli062-jpg/SkillSentry-Adaptive-Honey-Worlds-for-Dynamic-Safety-Testing-of-Agent-Skills. |
| 2026-08-04 | [Flying over The Uncertain Nature (FORTUNE): Intelligent and Humanistic 3D Path Planning for Low-Altitude Collaboration](http://arxiv.org/abs/2608.03408v1) | Minghui Liwang, Wenhan Jia et al. | The proliferation of low-altitude intelligent agents is increasing the demand for timely and socially responsible collaborative sensing in dynamic urban environments. However, jointly addressing heterogeneous spatiotemporal demands, environmental uncertainty, and human-centered operational constraints remains challenging. This paper studies 3D multi-UAV path planning and task assignment under uncertain ground PoI demands. Unlike existing work assuming static and fully known PoIs, we model persistent, temporally predictable, and emergent demands within a unified framework. We further incorporate altitude-dependent societal and environmental costs, including noise exposure and public safety risks, to balance sensing performance with socially compliant operations. To solve the resulting large-scale mixed-integer nonlinear problem, we propose FORTUNE, a hierarchical offline-online framework. Offline, a Transformer predicts Type-II PoI activation windows, while an enhanced sparrow search algorithm generates coordinated flight plans through priority-aware decoding and danger-aware evolution. Online, a lightweight refinement module accommodates emerging Type-III PoIs while preserving global mission coherence. Experiments on real-world traffic data and synthetic scenarios show that FORTUNE consistently outperforms state-of-the-art methods in effectiveness, scalability, and practical applicability. |
| 2026-08-04 | [Self-Evolving Coding Agents](http://arxiv.org/abs/2608.03392v1) | Hao Zhou, Haichuan Hu et al. | Large language models are increasingly embedded in software engineering workflows as coding agents that can inspect repositories, invoke tools, execute tests, debug failures, and generate patches. Yet most existing agents remain largely static after deployment, even though software development is a dynamic, feedback-rich process in which repositories evolve, dependencies change, tests fail, and repair attempts leave reusable experience. This tension has motivated a growing body of work on self-evolving coding agents, where the agent improves its future behavior by updating its framework, memory, skills, tools, models, or collaboration structures from prior coding interactions. In this survey, we provide a systematic synthesis of this emerging area. We first define self-evolving coding agents and distinguish them from conventional coding agents and general self-evolving agents. We then develop an object-centered taxonomy that characterizes what evolves in these systems, and complement it with two orthogonal perspectives: when evolution occurs and what software-specific evidence drives it. Across the literature, we find that executable feedback, repository-level context, and coding trajectories give software engineering a distinctive role as a natural domain for agent self-evolution, but also introduce new challenges in feedback reliability, benchmark overfitting, safety, maintainability, cost, and generalization. By organizing existing work around these dimensions, this survey aims to clarify the conceptual boundaries of self-evolving coding agents and provide a foundation for designing more adaptive, reliable, and software-aware agentic systems. The papers we collect can be found at https://github.com/zhouhao1024/Awesome-Self-Evolving-Coding-Agents. |
| 2026-08-04 | [Reusing Operational Evidence After Context Changes: A Conservative Bayesian Framework for Autonomous Vehicle Safety](http://arxiv.org/abs/2608.03384v1) | Robab Aghazadeh Chakherlou, Siddartha Khastgir et al. | Operational evidence, i.e., evidence of operation without failure is an important component of confidence in the safety or reliability of a system in service, but it is costly to collect. When the context of operation changes, the relevance of previously collected operational evidence becomes unclear. This problem arises, for example, when an autonomous vehicle that has operated safely in one operational environment (the Target Operational Domain, TOD) is deployed in a different but related TOD. Existing practice often treats such evidence in an all-or-nothing manner: either it is fully reused, or it is discarded. Neither position is satisfactory when there are good reasons to believe that the new context is no worse than the previous one, but that belief is itself uncertain. This paper studies how to make post-change reliability claims by combining evidence from two TODs using Conservative Bayesian Inference (CBI), which combines the evidence from the new context with a weighted amount of previous context and partial prior knowledge through constraints on a set of admissible priors. This yields conservative posterior bounds on quantities of interest. A numerical example illustrates how pre-existing evidence from a previous TOD can be transferred, conservatively and transparently, to support reliability claims in the changed TOD. |
| 2026-08-04 | [Long-term Traffic Scene Prediction via Polynomial Representations in Autonomous Driving](http://arxiv.org/abs/2608.03330v1) | Yue Yao | This thesis addresses fundamental challenges in traffic scene prediction for autonomous driving by introducing robust and computationally efficient models based on polynomial representations. While conventional sequence-based representations often struggle with noise and generalization, this work demonstrates that polynomial representations offer significant advantages in computational efficiency, generalization, and prediction plausibility. Through theoretical analysis and empirical validation, this thesis demonstrates that moderate-degree polynomials capture real-world motion dynamics with high fidelity without constraining predictive performance. Building on this foundation, a prediction model representing both trajectories and map geometry with polynomial representations achieves near state-of-the-art accuracy on standard benchmarks while substantially improving generalization under distribution shift. Extending this concept, a diffusion- based generative framework enables multi-agent scene generation, producing traffic continuations that are more plausible and kinematically consistent than those generated by conventional baselines. Evaluations on the Argoverse 2 and Waymo Open datasets confirm that polynomial representations reduce computational cost, enhance cross-dataset generalization, and yield smoother trajectories and higher behavioral plausibility. The findings reveal that standard in-distribution evaluation and regression-based metrics may fail to reflect true model generalization and prediction plausibility. By providing theoretical justification and empirical validation, this dissertation estab- lishes polynomial trajectory representations as an efficient, expressive, and generalizable foundation for traffic scene prediction in safety critical autonomous driving. |
| 2026-08-04 | [Test-Time Scaling for Safe Text-Guided Image Generation via Intermediate Clean Estimates](http://arxiv.org/abs/2608.03284v1) | Jinya Sakurai, Shueicheng Yan et al. | Ensuring safety and policy compliance in text-to-image diffusion models remains a critical challenge, as benign or adversarial prompts can often elicit prohibited content, e.g. nudity and protected intellectual property. While training-based unlearning methods are effective, they are computationally expensive and prone to catastrophic interference with general capabilities. Conversely, existing test-time defenses are primarily prompt-centric, relying on modifying textual descriptions only, and overlook the visual signals for detection. In this paper, we propose to leverage the intermediate clean image estimated during the generation process and employ a sparse margin objective to detect prohibited concepts. When a violation is detected, we immediately intervene by optimizing a structured low-rank residual in the text-conditioning space via truncated backpropagation. This design allows weight-preserving detection, keeps non-violating inference latency nearly unchanged as the maximum budget increases, and offers flexibility in safety performance via test-time scaling. Extensive experiments on Stable Diffusion v1.4 and v3.5 across nudity removal, IP protection, and style erasure demonstrate superior performance across suppression, fidelity and preservation compared to prior weight-preserving baselines, providing a scalable and flexible solution for safe generative deployment. |
| 2026-08-04 | [ICO: Enhancing Semantic-Shift Jailbreaks via Iterative Context Optimization](http://arxiv.org/abs/2608.03210v1) | Hujian Zhu, Yihao Huang et al. | Foundation models have achieved remarkable success across diverse tasks, but they remain vulnerable. To investigate such vulnerabilities, semantic-shift jailbreaks have recently emerged as a promising attack paradigm. They bypass explicit safety mechanisms by replacing harmful terms in original harmful questions with benign alternatives and leveraging contextual information to induce the target model to reinterpret these alternatives as their corresponding harmful concepts. However, existing semantic-shift jailbreaks often achieve limited effectiveness. In this work, we reveal that this limitation arises from overlooking the semantic-shift capability of contexts. Through systematic analysis, we find that contexts exhibit substantially different abilities in inducing semantic shifts: contexts with stronger semantic-shift capabilities are more likely to guide models toward recovering harmful meanings and achieving successful jailbreaks. Based on this finding, we systematically identify and distill the characteristics of effective contexts and propose a black-box context-aware semantic-shift jailbreak framework with Iterative Context Optimization (ICO). In each iteration, ICO leverages these characteristics and feedback from the target model to optimize contexts. Extensive experiments on three datasets and eight target foundation models demonstrate that ICO consistently outperforms eight state-of-the-art baselines, achieving an average attack success rate of 74.6%. |

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



