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
| 2026-05-01 | [Self-Adaptive Multi-Agent LLM-Based Security Pattern Selection for IoT Systems](http://arxiv.org/abs/2605.00741v1) | Saeid Jamshidi, Foutse Khomh et al. | The adoption of Internet of Things (IoT) systems at the network edge of smart architectures is increasing rapidly, intensifying the need for security mechanisms that are both adaptive and resource-efficient. In such environments, runtime defence mechanisms are no longer limited to detection alone but become a resource-constrained task of selecting mitigation actions. Security controls must be carefully selected, combined, and executed under latency, energy, and computational constraints, while preventing unsafe interactions between controls. Existing approaches predominantly rely on static rule sets and learned policies, which provide limited guarantees of feasibility, conflict safety, and execution correctness in resource-constrained edge settings. To address this limitation, we introduce ASPO, a self-adaptive multi-agent security pattern selection that integrates Large Language Model (LLM)-based reasoning with deterministic enforcement within a MAPE-K control loop. ASPO explicitly separates stochastic decision generation from execution: LLM agents propose candidate mitigation portfolios, while a deterministic optimisation core enforces closed-world action integrity, conflict-free composition, and resource feasibility at every decision epoch. We deploy ASPO on a distributed edge-gateway testbed and evaluate it across two workloads, each comprising 500 and 1000 runtime security decisions, using replayed IoT attack traffic. In addition, the results demonstrate invariant safety properties, including 100% conflict-free activation, consistent resource feasibility across workloads, and stable pattern dominance with perfect rank preservation. Importantly, deeper decision exploration reduces extreme-case execution costs, compressing tail latency and energy overheads by 21.9% and 23.1%, respectively, without increasing mean energy consumption. |
| 2026-05-01 | [FinSafetyBench: Evaluating LLM Safety in Real-World Financial Scenarios](http://arxiv.org/abs/2605.00706v1) | Yutao Hou, Yihan Jiang et al. | Large language models (LLMs) are increasingly applied in financial scenarios. However, they may produce harmful outputs, including facilitating illegal activities or unethical behavior, posing serious compliance risks. To systematically evaluate LLM safety in finance, we propose FinSafetyBench, a bilingual (English-Chinese) red-teaming benchmark designed to test an LLM's refusal of requests that violate financial compliance. Grounded in real-world financial crime cases and ethics standards, the benchmark comprises 14 subcategories spanning financial crimes and ethical violations. Through extensive experiments on general-purpose and finance-specialized LLMs under three representative attack settings, we identify critical vulnerabilities that allow adversarial prompts to bypass compliance safeguards. Further analysis reveals stronger susceptibility in Chinese contexts and highlights the limitations of prompt-level defenses against sophisticated or implicit manipulation strategies. |
| 2026-05-01 | [STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](http://arxiv.org/abs/2605.00699v1) | Xutao Mao, Liangjie Zhao et al. | Red-teaming Vision-Language Models is essential for identifying vulnerabilities where adversarial image-text inputs trigger toxic outputs. Existing approaches treat image generation as a black box, returning only terminal toxicity scores and leaving open the question of when and how toxic semantics emerge during multi-step synthesis. We introduce STARE, a hierarchical reinforcement learning framework that treats the denoising trajectory itself as the attack surface, under a direct white-box T2I and query-only black-box VLM setting. By coupling a high-level prompt editor with low-level T2I fine-tuning via Group Relative Policy Optimization (GRPO), STARE attains a 68\% improvement in Attack Success Rate over state-of-the-art black-box and white-box baselines. More importantly, this trajectory-level view surfaces the Optimization-Induced Phase Alignment phenomenon: vanilla models exhibit diffuse toxicity, whereas adversarial optimization concentrates conceptual harms into early semantic phases and detail-oriented harms into late refinement. Targeted perturbations of either window selectively suppress different toxicity categories, indicating that this temporal structure is a genuine causal handle rather than a side effect of the hierarchical design. The phenomenon turns toxicity formation from a chaotic process into a small set of predictable vulnerability windows, providing both a potent attack engine and a basis for phase-aware safety mechanisms. Content warning: This paper contains examples of toxic content that may be offensive or disturbing. |
| 2026-05-01 | [ML-Bench&Guard: Policy-Grounded Multilingual Safety Benchmark and Guardrail for Large Language Models](http://arxiv.org/abs/2605.00689v1) | Yunhan Zhao, Zhaorun Chen et al. | As Large Language Models (LLMs) are increasingly deployed in cross-linguistic contexts, ensuring safety in diverse regulatory and cultural environments has become a critical challenge. However, existing multilingual benchmarks largely rely on general risk taxonomies and machine translation, which confines guardrail models to these predefined categories and hinders their ability to align with region-specific regulations and cultural nuances. To bridge these gaps, we introduce ML-Bench, a policy-grounded multilingual safety benchmark covering 14 languages. ML-Bench is constructed directly from regional regulations, where risk categories and fine-grained rules derived from jurisdiction-specific legal texts are directly used to guide the generation of multilingual safety data, enabling culturally and legally aligned evaluation across languages. Building on ML-Bench, we develop ML-Guard, a Diffusion Large Language Model (dLLM)-based guardrail model that supports multilingual safety judgment and policy-conditioned compliance assessment. ML-Guard has two variants, one 1.5B lightweight model for fast `safe/unsafe' checking and a more capable 7B model for customized compliance checking with detailed explanations. We conduct extensive experiments against 11 strong guardrail baselines across 6 existing multilingual safety benchmarks and our ML-Bench, and show that ML-Guard consistently outperforms prior methods. We hope that ML-Bench and ML-Guard can help advance the development of regulation-aware and culturally aligned multilingual guardrail systems. |
| 2026-05-01 | [Evaluating the Architectural Reasoning Capabilities of LLM Provers via the Obfuscated Natural Number Game](http://arxiv.org/abs/2605.00677v1) | Lixing Li | While Large Language Models have achieved notable success on formal mathematics benchmarks such as MiniF2F, it remains unclear whether these results stem from genuine logical reasoning or semantic pattern matching against pre-training data. This paper identifies Architectural Reasoning: the ability to synthesize formal proofs using exclusively local axioms and definitions within an alien math domain, as the necessary ability for future automated theorem discovery AI. We use the Obfuscated Natural Number Game, a benchmark to evaluate Architectural Reasoning. By renaming identifiers in the Natural Number Game in Lean 4, we created a zero-knowledge, closed environment. We evaluate state-of-the-art models, finding a universal latency tax where obfuscation increases inference time. The results also reveal a divergence in robustness: while general models (Claude-Sonnet-4.5, GPT-4o) suffer performance degradation, reasoning models (DeepSeek-R1, GPT-5, DeepSeek-Prover-V2) maintain the same accuracy despite the absence of semantic cues. These findings provide a quantitative metric for assessing the true capacity for mathematical reasoning. |
| 2026-05-01 | [Augmented Lagrangian Multiplier Network for State-wise Safety in Reinforcement Learning](http://arxiv.org/abs/2605.00667v1) | Jiaming Zhang, Yujie Yang et al. | Safety is a primary challenge in real-world reinforcement learning (RL). Formulating safety requirements as state-wise constraints has become a prominent paradigm. Handling state-wise constraints with the Lagrangian method requires a distinct multiplier for every state, necessitating neural networks to approximate them as a multiplier network. However, applying standard dual gradient ascent to multiplier networks induces severe training oscillations. This is because the inherent instability of dual ascent is exacerbated by network generalization -- local overshoots and delayed updates propagate to adjacent states, further amplifying policy fluctuations. Existing stabilization techniques are designed for scalar multipliers, which are inadequate for state-dependent multiplier networks. To address this challenge, we propose an augmented Lagrangian multiplier network (ALaM) framework for stable learning of state-wise multipliers. ALaM consists of two key components. First, a quadratic penalty is introduced into the augmented Lagrangian to compensate for delayed multiplier updates and establish the local convexity near the optimum, thereby mitigating policy oscillations. Second, the multiplier network is trained via supervised regression toward a dual target, which stabilizes training and promotes convergence. Theoretically, we show that ALaM guarantees multiplier convergence and thus recovers the optimal policy of the constrained problem. Building on this framework, we integrate soft actor-critic (SAC) with ALaM to develop the SAC-ALaM algorithm. Experiments demonstrate that SAC-ALaM outperforms state-of-the-art safe RL baselines in both safety and return, while also stabilizing training dynamics and learning well-calibrated multipliers for risk identification. |
| 2026-05-01 | [From Prediction to Practice: A Task-Aware Evaluation Framework for Blood Glucose Forecasting](http://arxiv.org/abs/2605.00645v1) | Alireza Namazi, Heman Shakeri | Clinical time-series forecasting is increasingly studied for decision support, yet standard aggregate metrics can obscure whether a model is actually useful for the task it is meant to serve. In safety-critical settings, low average error can coexist with dangerous failures in exactly the high-risk regimes that matter most. We present a task-aware evaluation framework for blood glucose forecasting built around two downstream uses: hypoglycemia early warning and insulin dosing decision support. For early warning, we evaluate on real data from three clinical cohorts using event-level recall and false alarms per patient-day, metrics that reflect operational alarm burden rather than aggregate accuracy. We show that models appearing acceptable overall, with recall above 0.9 on the full test set, can fail badly in the post-bolus slice, where insulin-on-board is elevated and missed warnings carry the greatest clinical consequences. Standard forecasting evaluation, however, does not test whether a model can reason about the effects of actions, a requirement for supporting insulin dosing decisions. We therefore add a second, interventional arm using the FDA-accepted UVA/Padova simulator, where we evaluate whether forecasters can predict glucose responses to altered insulin plans in paired factual/counterfactual scenarios. We show that models that look strong on real-data forecasting often fail to predict the direction, magnitude, or ranking of intervention effects, and choose poor insulin doses when evaluated under a clinically motivated cost. Taken together, the two arms reveal a consistent gap between forecasting accuracy and task-relevant usefulness. We release the benchmark, the standardized preprocessing pipeline for public cohorts, and the simulator-based interventional dataset as a reproducible toolkit. |
| 2026-05-01 | [Jailbreaking Vision-Language Models Through the Visual Modality](http://arxiv.org/abs/2605.00583v1) | Aharon Azulay, Jan Dubiński et al. | The visual modality of vision-language models (VLMs) is an underexplored attack surface for bypassing safety alignment. We introduce four jailbreak attacks exploiting the vision component: (1) encoding harmful instructions as visual symbol sequences with a decoding legend, (2) replacing harmful objects with benign substitutes (e.g., bomb -> banana) then prompting for harmful actions using the substitute term, (3) replacing harmful text in images (e.g., on book covers) with benign words while visual context preserves the original meaning, and (4) visual analogy puzzles whose solution requires inferring a prohibited concept. Evaluating across six frontier VLMs, our visual attacks bypass safety alignment and expose a cross-modality alignment gap: text-based safety training does not automatically generalize to harmful intent conveyed visually. For example, our visual cipher achieves 40.9% attack success on Claude-Haiku-4.5 versus 10.7% for an equivalent textual cipher. To further our insight into the attack mechanism, we present preliminary interpretability and mitigation results. These findings highlight that robust VLM alignment requires treating vision as a first-class target for safety post-training. |
| 2026-05-01 | [Linking Behaviour and Perception to Evaluate Meaningful Human Control over Partially Automated Driving](http://arxiv.org/abs/2605.00556v1) | Ashwin George, Lucas Elbert Suryana et al. | Partial driving automation creates a tension: drivers remain legally responsible for vehicle behaviour, yet their active control is significantly reduced. This reduction undermines the engagement and sense of agency needed to intervene safely. Meaningful human control (MHC) has been proposed as a normative framework to address this tension. However, empirical methods for evaluating whether existing systems actually provide MHC remain underdeveloped. In this study, we investigated the extent to which drivers experience MHC when interacting with partially automated driving systems. Twenty-four drivers completed a simulator study involving silent automation failures under two modes - haptic shared control (HSC) and traded control (TC). We derived behavioural metrics from telemetry data, subjective perception scores from post-trial surveys and used them to test hypothesised relations between them derived from the properties of systems under MHC. The confirmatory analysis showed a significant negative correlation between the perception of the automated vehicle (AV) understanding the driver and conflict in steering torques. An exploratory analysis also revealed a surprising positive correlation between reaction times and the perception of sufficient control. Qualitative feedback from open-ended post-experiment questionnaires revealed that mismatches in intentions between the driver and automation, lack of safety, and resistance to driver inputs contribute to the reduction of perceived MHC, while subtle haptic guidance aligned with driver intent had a positive effect. These findings suggest that future designs should prioritise effortless driver interventions, transparent communication of automation intent, and context-sensitive authority allocation to strengthen meaningful human control in partially automated driving. |
| 2026-05-01 | [Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance](http://arxiv.org/abs/2605.00553v1) | Minchan Kwon, Sunghyun Baek et al. | Large Language Model (LLM) Red-Teaming, which proactively identifies vulnerabilities of LLMs, is an essential process for ensuring safety. Finding effective and diverse attacks in red-teaming is important, but achieving both is challenging. Generative Flow Networks (GFNs) that perform distribution matching are a promising methods, but they are notorious for training instability and mode collapse. In particular, unstable rewards in red-teaming accelerate mode collapse. We propose Stable-GFN (S-GFN), which eliminates partition function $Z$ estimation in GFN and reduces training instability. S-GFN avoids Z-estimation through pairwise comparisons and employs a robust masking methodology against noisy rewards. Additionally, we propose a fluency stabilizer to prevent the model from getting stuck in local optima that produce gibberish. S-GFN provides more stable training while maintaining the optimal policy of GFN. We demonstrate the overwhelming attack performance and diversity of S-GFN across various settings. |
| 2026-05-01 | [Zero-Knowledge Model Checking](http://arxiv.org/abs/2605.00487v1) | Pascal Berrang, Mirco Giacobbe et al. | We introduce a technology to formally verify that a software system satisfies a temporal specification of functional correctness, without revealing the system itself. Our method combines a deductive approach to model checking to obtain a formal certificate of correctness for the system, with zero-knowledge proofs to convince an external verifier that the system -- kept secret -- complies with its specification of correctness -- made public. We consider proof certificates represented as ranking functions, and introduce both an explicit-state and a symbolic scheme for model checking in zero knowledge. Our explicit-state scheme assumes systems represented as transition graphs. We use polynomial commitments to convince the verifier that the public proof certificates correspond to the secret transition relation. Our symbolic scheme assumes systems specified as linear guarded commands and uses piecewise-linear ranking functions. We apply Farkas' lemma to obtain a witness for the validity of the ranking function with public and secret components, and employ sigma protocols for matrix multiplication and range proofs to convince the verifier of the witness's existence. We built a prototype to demonstrate the practical efficacy of our two schemes on linear temporal logic verification examples. Our technology enables formal verification in domains where both the safety and the confidentiality of the system under analysis are critical. |
| 2026-05-01 | [ReLay: Personalized LLM-Generated Plain-Language Summaries for Better Understanding, but at What Cost?](http://arxiv.org/abs/2605.00468v1) | Joey Chan, Yikun Han et al. | Plain Language Summaries (PLS) aim to make research accessible to lay readers, but they are typically written in a one-size-fits-all style that ignores differences in readers' information needs and comprehension. In health contexts, this limitation is particularly important because misunderstanding scientific information can affect real-world decisions. Large language models (LLMs) offer new opportunities for personalizing PLS, but it remains unclear whether personalization helps, which strategies are most effective, and how to balance personalization with safety. We introduce ReLay, a dataset of 300 participant--PLS pairs from 50 lay participants in both static (expert-written) and interactive (LLM-personalized) settings. ReLay includes user characteristics, health information needs, information-seeking behavior, comprehension outcomes, interaction logs, and quality ratings. We use ReLay to evaluate five LLMs across two personalization methods. Personalization improves comprehension and perceived quality, but it also raises the risk of reinforcing user biases and introducing hallucinations, revealing a trade-off between personalization and safety. These findings highlight the need for personalization methods that are both effective and trustworthy for diverse lay audiences. |
| 2026-05-01 | [Impact of Task Phrasing on Presumptions in Large Language Models](http://arxiv.org/abs/2605.00436v1) | Kenneth J. K. Ong | Concerns with the safety and reliability of applying large-language models (LLMs) in unpredictable real-world applications motivate this study, which examines how task phrasing can lead to presumptions in LLMs, making it difficult for them to adapt when the task deviates from these assumptions. We investigated the impact of these presumptions on the performance of LLMs using the iterated prisoner's dilemma as a case study. Our experiments reveal that LLMs are susceptible to presumptions when making decisions even with reasoning steps. However, when the task phrasing was neutral, the models demonstrated logical reasoning without much presumptions. These findings highlight the importance of proper task phrasing to reduce the risk of presumptions in LLMs. |
| 2026-05-01 | [ClozeMaster: Fuzzing Rust Compiler by Harnessing LLMs for Infilling Masked Real Programs](http://arxiv.org/abs/2605.00413v1) | Hongyan Gao, Yibiao Yang et al. | Ensuring the reliability of the Rust compiler is of paramount importance, given increasing adoption of Rust for critical systems development, due to its emphasis on memory and thread safety. However, generating valid test programs for the Rust compiler poses significant challenges, given Rust's complex syntax and strict requirements. With the growing popularity of large language models (LLMs), much research in software testing has explored using LLMs to generate test cases. Still, directly using LLMs to generate Rust programs often results in a large number of invalid test cases. Existing studies have indicated that test cases triggering historical compiler bugs can assist in software testing. Our investigation into Rust compiler bug issues supports this observation. Inspired by existing work and our empirical research, we introduce a bracket-based masking and filling strategy called clozeMask. The clozeMask strategy involves extracting test code from historical issue reports, identifying and masking code snippets with specific structures, and using an LLM to fill in the masked portions for synthesizing new test programs. This approach harnesses the generative capabilities of LLMs while retaining the ability to trigger Rust compiler bugs. It enables comprehensive testing of the compiler's behavior, particularly exploring edge cases. We implemented our approach as a prototype CLOZEMASTER. CLOZEMASTER has identified 27 confirmed bugs for rustc and mrustc, of which 10 have been fixed by developers. Furthermore, our experimental results indicate that CLOZEMASTER outperforms existing fuzzers in terms of code coverage and effectiveness. |
| 2026-05-01 | [An eHMI Presenting Request-to-Intervene and Takeover Status of Level 3 Automated Vehicles to Support Surrounding Traffic Safety](http://arxiv.org/abs/2605.00377v1) | Hailong Liu, Masaki Kuge et al. | Level 3 automated vehicles (AVs) issue a request to intervene (RtI) when the automated driving system approaches its system limitations. Although this takeover transition is safety-critical, it is usually invisible to surrounding manually driven vehicle (MV) drivers. This study proposes an external human-machine interface (eHMI) called eHMI C+O that externalizes the RtI-related takeover status of a Level~3 AV using cyan and orange light bars. A driving-simulator experiment with 40 participants examined whether the proposed eHMI supports surrounding MV drivers during AV takeover scenarios. The results showed that, compared with the ADS-status-only eHMI condition, which is similar to ``Automated Driving Marker Lights,'' and the no-eHMI condition, the proposed eHMI C+O significantly improved participants' understanding of the AV's driving intention, their prediction of its behavior, and their perceived sufficiency of the information presented by the AV. It also reduced hesitation, increased confidence, and promoted earlier and larger increases in time headway after the RtI was issued. In the AV accident scenario, eHMI C+O significantly reduced the odds of accident involvement for the following MV compared with the no-eHMI condition, corresponding to a 76.8% reduction in accident odds. Exploratory path analysis suggested that the safety benefit of the proposed eHMI C+O may be associated with improved situation awareness and earlier defensive driving responses. These findings indicate that externalizing RtI-related takeover status can help surrounding drivers better understand Level 3 AVs and respond more safely during safety-critical takeover transitions. |
| 2026-05-01 | [AlphaInventory: Evolving White-Box Inventory Policies via Large Language Models with Deployment Guarantees](http://arxiv.org/abs/2605.00369v1) | Chenyu Huang, Jianghao Lin et al. | We study how large language models can be used to evolve inventory policies in online, non-stationary environments. Our work is motivated by recent advances in LLM-based evolutionary search, such as AlphaEvolve, which demonstrates strong performance for static and highly structured problems such as mathematical discovery, but is not directly suited to online dynamic inventory settings. To this end, we propose AlphaInventory, an end-to-end inventory-policy evolution and inference framework grounded in confidence-interval-based certification. The framework trains a large language model using reinforcement learning, incorporates demand data as well as numerical and textual features beyond demand, and generates white-box inventory policy with statistical safety guarantees for deployment in future periods. We further introduce a unified theoretical interface that connects training, inference, and deployment. This allows us to characterize the probability that the AlphaInventory evolves a statistically safe and improved policy, and to quantify the deployment gap relative to the oracle-safe benchmark. Tested on both synthetic data and real-world retail data, AlphaInventory outperforms classical inventory policies and deep learning based methods. In canonical inventory settings, it evolves new policies that improve upon existing benchmarks. |
| 2026-05-01 | [Unlearning What Matters: Token-Level Attribution for Precise Language Model Unlearning](http://arxiv.org/abs/2605.00364v1) | Jiawei Wu, DouDou Zhou | Machine unlearning has emerged as a critical capability for addressing privacy, safety, and regulatory concerns in large language models (LLMs). Existing methods operate at the sequence level, applying uniform updates across all tokens despite only a subset encoding the knowledge targeted for removal. This introduces gradient noise, degrades utility, and leads to suboptimal forgetting. We propose TokenUnlearn, a token-level attribution framework that identifies and selectively targets critical tokens. Our approach combines knowledge-aware signals via masking, and entropy-aware signals to yield importance scores for precise token selection. We develop two complementary strategies: hard selection, applying unlearning only to high-importance tokens, and soft weighting, modulating gradient contributions based on importance scores. Both extend existing methods to token-level variants. Theoretical analysis shows token-level selection improves gradient signal-to-noise ratio. Experiments on TOFU and WMDP benchmarks across three model architectures demonstrate consistent improvements over sequence-level baselines in both forgetting effectiveness and utility preservation. |
| 2026-05-01 | [Conformalized Quantum DeepONet Ensembles for Scalable Operator Learning with Distribution-Free Uncertainty](http://arxiv.org/abs/2605.00330v1) | Purav Matlia, Christian Moya et al. | Operator learning enables fast surrogate modeling of high-dimensional dynamical systems, but existing approaches face two fundamental limitations: quadratic inference complexity and unreliable uncertainty quantification in safety-critical settings. We propose Conformalized Quantum DeepONet Ensembles, a framework that addresses both challenges simultaneously. By leveraging Quantum Orthogonal Neural Networks (QOrthoNNs), we reduce operator inference complexity from O(n^2) to O(n), enabling scalable evaluation over fine discretizations. To provide rigorous uncertainty quantification, we combine ensemble-based epistemic modeling with adaptive conformal prediction, yielding distribution-free coverage guarantees. A key challenge in ensembling is that naive parallelism scales hardware resources linearly with the number of models. We resolve this by using Superposed Parameterized Quantum Circuits (SPQCs), which compress multiple ensemble members into a single circuit and enable simultaneous multi-model execution. Experiments on synthetic partial differential equations and real-world power system dynamics demonstrate that our approach achieves accurate predictions while maintaining calibrated uncertainty under realistic quantum noise. These results establish a practical pathway toward scalable, uncertainty-aware operator learning in quantum machine learning. |
| 2026-05-01 | [Prompt-Induced Score Variance in Zero-Shot Binary Vision-Language Safety Classification](http://arxiv.org/abs/2605.00326v1) | Charles Weng, Dingwen Li et al. | Single-prompt first-token probabilities from zero-shot vision-language model (VLM) safety classifiers are treated as decision scores, but we show they are unreliable under semantically equivalent prompt reformulation: even when the binary label is constrained to a fixed output position, equivalent prompts can induce materially different unsafe probabilities for the same sample. Across multimodal safety benchmarks and multiple VLM families, cross-prompt variance is strongly associated with prompt-level disagreement and higher error, making it a useful fragility diagnostic. A training-free mean ensemble improves NLL on all 14 dataset-model evaluation pairs and ECE on 12/14 relative to a train-selected single-prompt baseline, and wins more head-to-head NLL comparisons than labeled temperature scaling, Platt scaling, and isotonic regression applied to the same prompt. Ranking gains are consistent against the train-selected baseline on both AUROC and AUPRC, and against the full 15-prompt distribution remain consistent on AUPRC while softening on AUROC. Labeled calibration on top of the mean provides further gains when labels are available, identifying prompt averaging as a strong label-free first stage rather than a replacement for calibration. We frame this as a reliability stress test for zero-shot VLM first-token safety scores and recommend prompt-family evaluation with mean aggregation as a standard label-free reliability baseline. |
| 2026-05-01 | [Intelligent Elastic Feature Fading: Enabling Model Retrain-Free Feature Efficiency Rollouts at Scale](http://arxiv.org/abs/2605.00324v1) | Jieming Di, Xiaoyu Chen et al. | Large-scale ranking systems depend on thousands of features derived from user behavior across multiple time horizons. Typically requires model retraining -- resulting in long iteration cycles (3--6 months), substantial GPU resource consumption, and limited rollout throughput.   We introduce Intelligent Elastic Feature Fading (IEFF), a production infrastructure system that enables retrain-free feature efficiency rollouts by elastically controlling feature coverage and distribution at serving time. IEFF supports incremental feature coverage adjustments while models adapt through recurring training, eliminating dependencies on explicit retraining cycles. The system incorporates strict safety guardrails, reversibility mechanisms, and comprehensive monitoring to ensure stability at scale.   Across multiple production use cases, IEFF accelerates efficiency-related rollouts by 5$\times$, eliminates retraining-related GPU overhead, and enables faster capacity recycling. Extensive offline and online experiments demonstrate that gradual feature fading prevents 50--55\% of online performance degradation compared to abrupt feature removal, while maintaining stable model behavior. These results establish elastic, system-level feature fading as a practical and scalable approach for managing feature efficiency in modern industrial ranking systems. |

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



