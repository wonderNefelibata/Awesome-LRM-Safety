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
| 2026-10-19 | [CoopUQ: Robust Multi-Robot Social Navigation via Uncertainty-Aware Cooperative Forecasting](http://arxiv.org/abs/2610.26852v1) | Zehao Wang, Jianpeng Yao et al. | Reinforcement learning enables mobile robots to make fast and scalable navigation decisions in constrained human-shared environments. However, directly extending single-robot approaches to multi-robot settings often fails to ensure safety and efficiency, especially under occlusions and out-of-distribution (OOD) scenarios, where individual robots may overlook potential risks or make inaccurate predictions. Leveraging cooperation effectively for both prediction and navigation remains a major challenge. We propose CoopUQ, a multi-robot social navigation framework that bridges uncertainty-aware cooperative occupancy forecasting and robust decision making. CoopUQ predicts future occupancy grid maps through cross-robot spatiotemporal feature fusion and further leverages forecasts and their uncertainty to guide policy via constrained multi-agent reinforcement learning, where the cost penalizes navigation through regions predicted to be occupied or highly uncertain. This design enables the policy to adapt its risk sensitivity under distribution shifts. Experimental results across multiple Unity3D-based simulation environments with both in-distribution and OOD settings, including diverse layouts, pedestrian densities, and rushing behavior, demonstrate improved prediction performance and safety over representative baselines. Real-robot experiments further demonstrate its robustness across various scenes. Our code and videos are available at https://coopuq.github.io/. |
| 2026-10-19 | [Safety Mid-Training: Internalizing LLM Safety as a Foundational Capability](http://arxiv.org/abs/2610.26809v1) | Zhengyue Zhao, Liwei Jiang et al. | Safety alignment for large language models (LLMs) is primarily performed during post-training, while recent work has also begun incorporating safety data into pre-training. Yet safety can degrade when later training no longer includes safety data, and gains in safety often come at the cost of general capability. This tension suggests that safety and general reasoning should not be treated as separate training goals. We instead view safety as an LLM capability that can be learned through a reasoning structure shared with general tasks. Both require the model to understand intent or task goals, apply relevant knowledge, and reason about potential consequences. We introduce safety mid-training, a stage between pre-training and post-training that jointly develops safety and general reasoning through this shared structure. Given safety and utility prompts, the base model generates structured documents and reasoning traces that are then used for mid-training. Safety mid-training reduces average attack success rate (ASR) by 72% to 81% relative to Qwen3-1.7B, 4B, and 8B base models, while also improving general capability. The benefits persist through subsequent post-training. After general-only SFT, mid-trained models largely retain their safety, achieving lower ASR than the baselines without sacrificing general capability. They also retain safety under general-only GRPO without safety rewards, while subsequent RL reaches a better safety-utility frontier when initialized from the mid-trained models. Together, these results suggest that learning safety and general reasoning within a shared mid-training stage can reduce the safety-utility trade-off and make safety more robust to subsequent training. |
| 2026-10-19 | [An Efficient Open-Source Framework for Thermohygromechanical Phase-Field Simulation of Drying Damage Refractory Castables](http://arxiv.org/abs/2610.26752v1) | Murilo H. Moreira, Roberto F. Ausas et al. | The water removal from refractory castables is a complex process involving heat and mass transport and thermohygromechanical stresses. In order to reduce the risk of explosive spalling by improving the heat-up curves, numerical models are critical to predict the castable behavior during drying. However, most of the existing models either neglect the thermomechanical stresses or consider them as an \textit{a posteriori} assessment of the likelihood of failure. The present work proposes an open-source thermohygromechanical phase-field model to simulate the drying of refractory castables and predict the occurrence of explosive spalling. The model is based on an adaptive two-scale time-stepping strategy that allows for a significant reduction of the computational cost associated with the simulation. The model was validated against literature results and qualitative comparisons with experimental data were carried out to assess its ability to predict the failure behavior of partially saturated porous media during heating. The numerical simulations were also applied to investigate the impact of different heating methods on the risk of failure during drying. It was found that even at the significantly lower heating rates typical of refractory castable drying - compared to the fire accidents that motivate most thermomechanical studies on concrete structures - thermomechanical stresses still play a significant role in the explosive spalling, a finding well established for concrete under fire but novel for refractory castables. As a consequence, it was also concluded that the use of heterogeneous heating methods can increase the likelihood of explosive spalling. Ultimately, feasible solutions to increase the drying efficiency and safety that are compatible with the industrial setting were proposed and a free, open-source implementation of the model is provided. |
| 2026-10-19 | [Towards Robust Model Organisms for Eliciting Latent Knowledge](http://arxiv.org/abs/2610.26698v1) | Lennart Finke, Hardik Bhatnagar et al. | A key problem in AI safety is eliciting latent knowledge (ELK), extracting information from a model that is misaligned and more knowledgeable than its overseer. For example, a misaligned model used to scan code for vulnerabilities may want to withhold its findings, i.e., sandbag, in order to exploit them later. To study ELK empirically, we need model organisms that mirror this setting. Existing model organisms fall short in several respects. In this work, we train model organisms to withhold solutions in a vulnerability detection task. We combine transcript distillation with counter-prompting and tamper-resistance training to instill three properties that we argue are critical for a useful ELK testbed: 1) Selectivity, the model sandbags when it thinks the overseer doesn't know the answer and complies otherwise, 2) Robustness, the model resists elicitation, and 3) Retention, the model retains a knowledge advantage over the overseer, as opposed to losing the relevant capability. We train a sequence of model organisms and show that our strongest model substantially outperforms prior models on robustness while keeping the other two properties intact. We hope that our model organisms provide a useful ELK testbed for the community, and that our methodology sparks the development of increasingly challenging model organisms.   All our code, models and the dataset are available at https://huggingface.co/collections/lennart-finke/sandbagging. |
| 2026-10-19 | [A Type System for Communication-Safe Hardware Design](http://arxiv.org/abs/2610.26599v1) | Aditya Ranjan Jha, Jason Zhijingcheng Yu et al. | Hardware components communicate through clock-cycle-sensitive protocols. Hardware Description Languages (HDLs), however, provide limited support for expressing these protocols or for checking that independently developed components agree on them. When components disagree, the composed design can perform unsynchronized exchanges or stop making progress. We call these failures communication-safety violations. Communication safety is therefore a global property, determined by interactions between multiple components. This makes it difficult to check modularly.   This paper presents Pact, a typed extension of Anvil for modular verification of communication safety in synchronous hardware. Pact types describe message-passing, control-flow, and clock-cycle behaviour. Each component is checked against its type. Compatible types then guarantee that the components compose into a communication-safe system.   Pact also supports incremental design. It can check a subsystem before the full system context is available. The check produces a summary type that captures the communication behaviour left for future compositions. To reduce annotation burden, Pact infers types by abstracting payload values while preserving communication, timing, control, and fairness. These inferred types are optimal for the checker: whenever some checker-accepted types can prove compatibility, the inferred types can prove it as well. Type inference and type checking are decidable, and compatibility is decided by a sound and complete algorithm.   We implement Pact by extending the Anvil compiler and evaluate it along four dimensions: detecting communication errors during incremental design, expressiveness on ten clock-cycle-accurate translations of open-source hardware IPs, compatibility-checking time, and generated hardware overhead. Pact expresses realistic protocols and verifies all ten benchmarks in reasonable time. |
| 2026-10-19 | [Concerns in Benchmarking Jailbreaks: An Empirical Study of Attack Success Rates](http://arxiv.org/abs/2610.26595v1) | Leon Tu Nguyen, Rohan Menon et al. | Jailbreak methods are commonly compared using Attack Success Rate (ASR), but these comparisons depend on how responses are generated and judged. In this paper, we examine eight attacks against three open-weight language models using three evaluation procedures. Low-resource language attacks achieve a mean ASR of 96.0% under keyword matching but 2.7% under a Qwen3 evaluator following StrongREJECT scoring instructions. Changing the number and arrangement of attempts reverses the ranking of DeepInception and rewriting. In a separate experiment, sampling 100 responses for each unchanged request at Ollama's default temperature of 0.8 yields 100% ASR under keyword matching but 50% under the Qwen3 evaluator. Beyond evaluator sensitivity, these findings raise a safety concern: repeated sampling can expose residual unsafe behavior despite an initial refusal, suggesting probabilistic suppression rather than robust exclusion. We recommend explicit success definitions, matched response budgets, validated evaluators, and reporting cumulative failure under repeated interaction rather than relying on single-generation tests. |
| 2026-10-19 | [Beyond Attack Success Rate: Cost-Aware Metric for Evaluating Iterative LLM Jailbreaks](http://arxiv.org/abs/2610.26589v1) | Rohan Menon, Chih-Hong Cheng | Attack success rate (ASR) evaluates jailbreaks against large language models without accounting for repeated calls to the victim model. Yet each additional call gives the deployed system another opportunity to detect the attack and intervene. We introduce $\mathrm{ASR}_{\mathrm{STEP}}$, which penalizes additional victim-model calls while retaining binary success judgments. Our evaluation considers GCG~\cite{zou2023universal}, PAIR~\cite{chao2025jailbreaking} with a step budget, and a single-step PAP~\cite{zeng2024johnny} baseline on JailBreakBench across three model families. GCG's scores decrease by 7.0--8.7 percentage points after discounting, while PAP's remain unchanged. While for PAIR~\cite{chao2025jailbreaking}, we see a wider discounting range between 3.4-12 percentage points, showing that the "step" quantity differs per iterative strategy. Reporting both metrics makes the discount due to repeated victim-model interaction explicit. Code is available \hyperlink{https://github.com/SuELS-Group/beyond_asr}{here}. |
| 2026-10-19 | [SafeWall: A Risk-Aware Job Walltime Recommendation for HPC Scheduling](http://arxiv.org/abs/2610.26577v1) | Kanglin Xu, Michael Papka et al. | In high-performance computing (HPC), users must submit resource requirements for their jobs, with user-supplied walltime being a critical parameter. However, these estimates are notoriously inaccurate, as users tend to overestimate walltime to prevent the scheduler from terminating their jobs prematurely. While existing work on learning-based walltime prediction primarily focuses on maximizing prediction accuracy, such predictions are inherently risky for walltime recommendations. Specifically, underestimation leads to job termination before completion, resulting in lost progress and wasted resources. In this study, we present SafeWall, a novel recommendation method that adjusts walltime predictions by incorporating an explicit, nonnegative safety buffer to mitigate underestimation risk. The core of our approach is a risk-aware buffer that dynamically calibrates safety margins based on individual user's historical behavior. When user data is limited, the system employs statistical shrinkage to align these margins with a global baseline, ensuring robustness. This yields a lightweight mechanism that optimizes the trade-off between prediction accuracy and termination risk. SafeWall is modular and applicable to various learning-based prediction models. We evaluate SafeWall using five public workload traces, including the full-lifetime job log from the 4,392-node Theta machine at the Argonne Leadership Computing Facility. Across all traces, SafeWall substantially reduces underestimation risk compared to the unbuffered base predictors, while preserving competitive accuracy. The source code is publicly available at https://github.com/SPEAR-UIC/JSSPP26-SafeWall. |
| 2026-10-19 | [Training Natural Schemers: RL Can Amplify Oversight-Conditional Behavior](http://arxiv.org/abs/2610.26551v1) | Bart Jaworski, Nathaniel Mitrani et al. | One of the central concerns for the safety of advanced AI is scheming: when a model strategically complies under oversight but pursues its own goals when unobserved. Current demonstrations, such as Anthropic's Alignment Faking work, depend on design choices that make them less convincing: explicit test-time prompting, handcrafted backdoors, or training data that demonstrates or hints at the target behavior. Model organisms built in this way cannot tell us whether scheming tendencies would arise or increase under realistic training conditions. We therefore construct a natural model organism of scheming, built on Kimi K2.6: trained with traits that could plausibly emerge in a frontier training pipeline, and without providing the model with the target scheming policy. The resulting model is a benign schemer: it aims to maximize aggregate help across all users over time, values its own continued operation only as a means to future help, and knows the consequences of oversight and deployment. We then train it using Reinforcement Learning (RL) with a reward that scores only task quality and policy compliance and does not reward scheming; we show that this amplifies scheming. We measure the oversight flip (the gap between monitored and unmonitored compliance with a given policy) and find that it increases during RL training across our environments. Chain-of-Thought (CoT) analysis finds explicit instrumental reasoning about oversight accompanying this behavior. The effect holds across our three RL environments, and we find that the scheming policy transfers between them. The pipeline also replicates on other open-weight base models. Our results suggest that a reward that does not reward scheming offers no inherent protection against certain traits when an oversight-conditional policy is indistinguishable from an aligned one during training. |
| 2026-10-19 | [AfriGuard: Safety and Robust Refusal For African Languages](http://arxiv.org/abs/2610.26474v1) | Israel Abebe Azime, Jesujoba O. Alabi et al. | Large language models (LLMs) designed to interact with users are typically trained to respond appropriately to harmful or toxic content. However, ensuring appropriate responses becomes more challenging in diverse linguistic and cultural contexts, where safety risks may be expressed or interpreted differently. Building safe AI systems therefore requires detecting safety risks, understanding their nature, and responding appropriately, while accounting for linguistic and cultural variations that add complexity to each stage. In this work, we introduce AfriGuard, a culturally grounded multilingual dataset for studying harmful intent, safety-category understanding, and refusal behavior across 10 African languages and English. Our findings reveal that guardrail models underperform both open- and closed-source LLMs on safety-intent and safety-category classification, while models show limited understanding of appropriate safety behavior, scoring below 60% F1. We further find that models often fail to respond appropriately to unsafe prompts not because they explicitly refuse, but because they produce linguistically inadequate responses that result in incidental non-assistance rather than proper refusals. Finally, using AfriGuard, we develop efficient guardrail models that achieve performance competitive with models up to 30x larger, introduce refusal detectors that improve average performance by 43.9% over existing detectors, and demonstrate that supervised fine-tuning can substantially improve overall safety performance. |
| 2026-10-19 | [When Guidance Becomes a Guard: Early Interception of Unsafe Inputs in Text-to-Image Generation](http://arxiv.org/abs/2610.26452v1) | Yanchun Li, Yujin Wu et al. | Text-to-image diffusion models can produce unsafe content, while output filtering intervenes only after synthesis. We investigate whether native generation signals enable early detection of unsafe and adversarial inputs. We find that classifier-free guidance exposes strong safety signals in conditional responses, accessible through compact statistics and linear readouts. Based on this finding, we propose Conditional Semantic Trajectory Interception (CSTI), which aggregates activation contrasts between conditional and unconditional responses at early checkpoints for calibrated risk detection. CSTI fits concept-specific detectors using canonical concept and safe examples, without attack samples, intermediate image decoding, or generator modification. Across five generators, four adversarial attack collections, and two general NSFW datasets, CSTI achieves the strongest overall performance among six baselines at a matched 5% FPR. Early interception reduces generation latency by over 60% on intercepted paths. These results demonstrate the effectiveness of native conditional responses for early safety interception. |
| 2026-10-19 | [Rethinking LLM Fine-Tuning via Weight Space Reparameterization: Preserving Safety during Downstream Adaptation](http://arxiv.org/abs/2610.26443v1) | Minseong Kim, Jongbok Won et al. | Fine-tuning large language models (LLMs) on downstream tasks often weakens their previously aligned safety behavior, creating an inherent trade-off between task adaptation and safety preservation. Existing approaches attempt to mitigate this issue by identifying safety-relevant neurons, layers, or update directions in the original parameter space. However, safety-related information is often distributed across many directions in this space, causing downstream updates to overwrite parameters that encode safety behavior. We propose WSR-Tune, a Weight Space Reparameterization-based fine-tuning framework that preserves safety during downstream adaptation. Our approach constructs a safety-conditioned basis and reparameterizes the weight matrices such that safety-relevant information becomes concentrated in a small subset of directions. We then identify safety-critical directions and freeze them in the reparameterized space, while updating the remaining complementary directions for downstream adaptation. By explicitly structuring the parameter space to disentangle safety and task information, our method reduces interference between objectives and enables simultaneous preservation of safety and improvement in downstream performance. Experimental results show that our method achieves a more favorable trade-off between downstream performance and safety retention, demonstrating its effectiveness for reliable LLM fine-tuning. |
| 2026-10-19 | [Answer Delivery Under a Shared Token Budget: An Audited Study of Stopping and Finalization](http://arxiv.org/abs/2610.26384v1) | Yupeng Chang, Linyuan Gao et al. | An inference budget is useful only when the system delivers an answer within it. We study a fixed-budget policy combining first-valid-box stopping with a charged finalization cue. On 100 fresh, independently audited GSM8K test questions per model, it improves delivered endpoint accuracy over first-box stopping by 8.0 percentage points for Qwen3-4B-Thinking-2507 and 7.0 for DeepSeek-R1-Distill-Llama-8B. A prelocked repeat on the same questions gives 7.0 and 10.0 points; neither seed shows a gross harm relative to first-box stopping. The point estimates of incremental physical-slot savings are below one percent. A finalization-only ablation separates cue recovery from stopping, and a fixed scalar reader tests extraction as an alternative explanation. The cue is prior art; the contribution is a component and readout analysis with charged injections and verified paths. Results cover two checkpoints, one task, one prompt family, and two fixed seed conditions. |
| 2026-10-19 | [Autonomous intelligence for mining productivity: a multi-agent AI approach to operational value creation](http://arxiv.org/abs/2610.26369v1) | N Guerin, A Dehoux et al. | Integrated Remote Operation Centres (IROCs) are emerging as the architectural frontier of Industry 4.0 for capital-intensive sectors such as mining, oil and gas, and energy. Yet two intertwined gaps constrain their evolution towards full autonomy. First, no integrated agentic architecture has been proposed for mining IROCs that orchestrates Supervisory Control and Data Acquisition systems, digital twins, predictive analytics and human-machine interfaces under safety-critical constraints. Second, no evaluation framework jointly addresses the real-time, Health-Safety-Environment, regulatory-auditability and cost-per-decision requirements that govern industrial agentic deployments. This paper makes four contributions. It synthesises industry needs and quantified gains documented in analogous industries that motivate an agentic approach to mining IROCs. It introduces EI2ROC, a three-tier multi-agent architecture combining a Large Language Model orchestrator with specialised monitoring and production-capacityprediction agents communicating through a Model Context Protocol-style tool interface. It provides a critical review of agentic evaluation along six orthogonal dimensions and identifies the industrial blind spot of current benchmarks. Finally, it proposes a multi-dimensional benchmarking framework for industrial agentic AI comprising scenario library, harness, metrics and humancalibration protocol. The paper states explicitly which components are implemented in the current prototype, which belong to the proposed reference design and which remain future work, and reports preliminary cost and latency observations from an operating site. Together, EI2ROC and the benchmarking framework constitute a methodological blueprint transferable across CAPEX intensive industries. |
| 2026-10-19 | [Quantifying Uncertainty Bounds in Spectral Shallow Water Bathymetry: The Profile Likelihood Framework for Semi-Analytical Models](http://arxiv.org/abs/2610.26352v1) | Mohamed Ali Ghannami, Guillaume Sicot et al. | Accurate bathymetric mapping in shallow waters is critical for coastal monitoring, navigation safety, and environmental conservation. While semi-analytical (SA) models have proven effective for estimating water column depth (WCD) from optical remote sensing data, quantifying the associated uncertainties remains a significant challenge. This study introduces a unified framework based on profile likelihood to evaluate WCD uncertainties in Semi-Analytical (SA) models, bridging traditional variance-covariance methods with likelihood-based inference. Through controlled simulations, we systematically assess the performance of first-order statistical tests-likelihood ratio, Wald (observed and expected), and score tests-across varying environmental conditions and sensor configurations.<p>Our results demonstrate that the expected Fisher information Wald test, equivalent to classical variance-covariance propagation, provides reliable uncertainty bounds in hyperspectral settings with clear waters. However, its performance deteriorates in multispectral applications and complex water conditions, where likelihood ratio tests prove more robust. This divergence highlights fundamental limitations in the small-sample regime typical of per-pixel spectral bathymetry. Notably, even under idealized conditions with true parameter initialization, local optimization challenges in SA models can compromise both parameter estimation and uncertainty quantification, suggesting even greater challenges in practical applications with less favorable initialization. The findings underscore the importance of robust statistical inference in improving the reliability of bathymetric products for operational applications.</p> |
| 2026-10-19 | [LLMs Prompted for Legal Context Refuse More: Overrefusal from Small On-Premises LLMs in Criminal Legal Context](http://arxiv.org/abs/2610.26211v1) | Anastasiia Kucherenko, Francois Brouchoud et al. | Confidentiality requirements increasingly push legal professionals toward small, on-device LLMs that keep case data in-house, used for everyday tasks such as summarization, translation, and reformulation. Following standard prompting guidance, a user naturally states their role (``I am a lawyer...''). We show that this innocuous habit has an unexpected cost: such authority framing can make the model selectively refuse legitimate requests, introducing bias through uneven access to assistance. We evaluate four modern small open-weight LLMs on benign legal-domain prompts to measure this over-refusal.   Surprisingly, we find that authority-style prefixes (''I am legal counsel working for a national supreme court'', ''I'm a defense lawyer'') systematically increase refusal rates by 2--20x over the no-prefix baseline, while a known role-play jailbreak prefix shows mixed effects, sharply increasing refusals in some models and barely shifting them in others. The finding suggests that small on-prem deployable LLMs are unstable under contextual framings that a real institutional user might naturally introduce, and further investigation is essential to minimize opportunities for bias. |
| 2026-10-19 | [Anytime-Valid Monitoring of Robust Stochastic Models](http://arxiv.org/abs/2610.26193v1) | Martin Kurečka, Roman Andriushchenko et al. | Formal methods provide a principal approach to obtaining safety guarantees for autonomous agents. Their key bottleneck is their reliance on a formal model (e.g., a Markov decision process) describing the agent-environment interaction. Learning or crafting the model is an error-prone process, which undermines the guarantees offered by formal methods. Robust models account for modelling uncertainty, but may still fail to capture the true environment. This issue can be addressed through \emph{runtime model monitoring}, which aims to raise an alarm when a likely model-environment discrepancy is detected, while keeping the probability of false alarm bounded. Existing approaches impose various assumptions and do not clearly explain the rationale behind their design choices, leaving the broader design space for these monitors somewhat opaque. In this paper, we formalise the design space of monitors for robust stochastic models through the lens of \emph{parameterised sequential betting} algorithms, highlighting the importance of \emph{observation feature} design (choosing what information about the history the monitor uses) and \emph{bet design} (how rich a family of betting rules the monitor can represent). We show that our framework generalizes the existing approaches, and we use its formalization to propose a new \emph{mixture of linear monitors (ML)} design. We prove that these ML monitors achieve an asymptotically optimal detection time within the class of linear monitors, making them a principled choice for our monitoring problem. Our experiments confirm that ML monitors significantly reduce detection times compared to existing monitors and can effectively exploit more complex observation features. |
| 2026-10-19 | [MoveBench: A Rubric-Based Benchmark for Evaluating Large Language Models in Sports and Health Coaching](http://arxiv.org/abs/2610.26185v1) | Yitong Li, Jiang Bian et al. | Despite rapid advances in large language models (LLMs), their deployment in the sports and health domain remains high-risk due to the absence of domain-specific evaluation benchmarks. Existing general-purpose benchmarks (e.g., MMLU and HELM) primarily assess broad knowledge and reasoning capabilities, but fail to capture critical domain requirements such as exercise safety, physiological appropriateness, personalization, and actionable coaching quality. This gap is particularly concerning in fitness and health applications, where incorrect or unsafe recommendations can lead to physical harm. To address this limitation, we introduce MoveBench, the first comprehensive benchmarking framework specifically designed for evaluating LLMs in sports and health coaching. The proposed benchmark integrates heterogeneous task formats, including multiple-choice questions (MCQs) and open-ended question answering (OpenQA), grounded in real-world user scenarios and paired with structured rubrics emphasizing safety compliance, scientific validity, and training logic. Through large-scale evaluation of 15 diverse LLMs on 809 questions with 5,012 rubric-level criteria, our results reveal substantial performance variability, particularly in safety awareness, structured program design, and personalization capability, highlighting a critical gap between knowing and doing. Our work establishes a foundational evaluation standard for sports and health LLM applications and provides a reproducible framework for advancing safe, reliable, and context-aware AI-driven coaching systems. Open resources are available at https://github.com/SatoSakula/MoveBench. |
| 2026-10-19 | [Nash-MOPD: Optimizing Task Trade-offs in Multi-Teacher On-Policy Distillation](http://arxiv.org/abs/2610.26124v1) | Siyuan Gan, Bo Yang et al. | Multi-teacher on-policy distillation (MOPD) aggregates the capabilities of multiple domain-expert teachers into a single student model. However, all domains in MOPD share the student's finite budget of optimization steps and rollouts. Fixed uses constant domain sampling ratios based on corpus sizes; by contrast, D$^3$-MOPD adjusts the ratios using the teacher-student reverse-KL divergence and its recent decrease. However, favoring large recent decreases can direct more training toward domains that already improve quickly, while domains that improve more slowly may receive too few examples. To address inflexible allocation and the risk of giving insufficient training to domains that improve more slowly, we propose Nash-MOPD, an adaptive domain scheduler based on a local minimax game using existing reverse-KL statistics. The equilibrium maximizes the smallest predicted proportional reduction under the local model and gives more prompts to domains that need more examples for comparable improvement. Consequently, Nash-MOPD responds to changing learning progress and supports domains that improve more slowly without extra teacher queries or rollouts. At step 100, Nash-MOPD exceeds both baselines' final mathematics and code checkpoint scores on DeepSeek-R1-Distill-Qwen-1.5B and reaches Fixed's on Qwen3-8B-Base. Furthermore, it leads eight of nine reported metrics and is best or tied best on all six mathematics and code accuracy metrics, respectively. |
| 2026-10-19 | [FinSafeGuard: A Quality-Driven Synthetic Dataset for Financial AI Safety](http://arxiv.org/abs/2610.26066v1) | Joseph Thomas Thacil, Reetu Raj Harsh et al. | Generating high-quality synthetic datasets at scale remains a fundamental challenge for training specialized AI systems in data-scarce domains. We present a comprehensive methodology that produces 14.3 million quality-filtered synthetic training examples from a structured taxonomy of 632,000 scenario specifications. Our four-stage pipeline ad- dresses critical challenges in synthetic data quality, diversity, and realism through: (1) systematic pre-generation research establishing optimal generation parameters via diminishing returns analysis, (2) conditional auto evolution mechanisms that remediate low-diversity batches during generation, (3) a two-layer semantic deduplication architecture combining lexical and embedding-based methods, and (4) LLM-as-Judge quality evaluation filtering samples based on behavioral correctness and tuple fidelity. Empirical analysis across three domains (Banking, Insurance, Financial Services) demonstrates domain-specific deduplication patterns ranging from 9.9% to 40.6%, with output generation modes exhibiting highest redundancy. Our batch size optimization reveals mode-specific diversity saturation thresholds, enabling principled allocation strategies. The complete pipeline reduces 26.4M raw samples to 14.3M high-confidence examples through merge (20.9% reduction), semantic deduplication (18.7% reduction), and quality-based filtering (15.9% reduction), providing a reproducible framework for domain-specific synthetic data generation with rigorous quality control. |

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



