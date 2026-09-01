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
| 2026-08-31 | [BLOOM-WILT: Logit Tilting for Behaviour Elicitation in Automated LLM Auditing](http://arxiv.org/abs/2608.31105v1) | Adrians Skapars, Edoardo Manino | Users of a deployed language model routinely encounter behaviours that testing almost never surfaces, since deployment puts the model through orders of magnitude more interactions than any evaluation can simulate. Automated auditors make testing cheap to scale and flexible enough to cover almost any specified behaviour, yet their lack of optimisation pressure makes them sample-inefficient. To address this shortcoming, we introduce BLOOM-WILT, a full auditing pipeline that elicits natural multi-turn instances of rare behaviours, without training cost or access beyond the target's next-token distribution. On the input side, WILT's auditor model revises its conversational strategy across rounds, learning from previous scored interactions. On the output side, WILT adaptively reweights the target's decoding using the model's own distribution conditioned on an elicitation prompt, so that behaviour-relevant generations are sampled ahead of others it finds equally probable when unprompted. We evaluate WILT across 4 target models and 8 behaviours, where it beats the baseline auditor in 30 of the 32 settings and overturns the previous model safety rankings. WILT raises average behaviour presence from 51% to 100% when eliciting self-harm encouragement from Qwen3.5-4B, beating every elicitation method we port into the same pipeline at matched compute, without pushing output probability below the baseline's. |
| 2026-08-31 | [Storage-Centric System Designs for Enabling Fast, Efficient, and Low-Cost Genomic and Metagenomic Analyses](http://arxiv.org/abs/2608.31004v1) | Nika Mansouri Ghiasi | Genomic and metagenomic analyses play critical roles in many fields, such as precision medicine, urgent clinical settings, discovering early warnings of communicable diseases, ensuring food safety through pathogen monitoring, agriculture, and scientific discovery. Due to the challenges of analyzing and storing massive volumes of genomic and metagenomic sequence data, significant efforts have been made to accelerate (meta)genomic analyses and store sequence data compressed. Despite the benefits of these techniques, we identify two major outstanding problems in accessing stored sequence data and supplying it to the analysis units: (i) the data movement bottleneck due to moving large amounts of low-reuse data from storage and the unnecessary burden on the rest of the system, and (ii) the data preparation bottleneck, where compressed sequence data needs to be first decompressed and formatted before analysis.   In this dissertation, we present customized storage-centric systems, which efficiently (i) analyze (meta)genomic data inside the storage system, and (ii) enable highly-compressed storage and high-performance access of large-scale sequence data, thereby alleviating the overheads of data movement, computation, and data preparation. We demonstrate that the proposed systems significantly improve system performance, energy efficiency, and system cost-efficiency of (meta)genomic analysis. We hope that the storage-centric systems proposed in this dissertation facilitate the broader adoption of (meta)genomic analyses and inspire future research to fundamentally improve the performance, energy efficiency, and cost-effectiveness of other data-intensive application domains related to health and life sciences. |
| 2026-08-31 | [Evidence, Logic, and Compliance: Multi-Agent Structured Graph Reasoning with Expert Arbitration for Medical Referral](http://arxiv.org/abs/2608.30938v1) | Qi Peng, Yi Cai et al. | Medical referral (directing patients to the appropriate hospital department) is a complex decision-making process requiring the synthesis of multimodal data, including patient narratives, laboratory indicators, and radiology imaging. While Large Language Models (LLMs) have advanced medical dialogue systems, they struggle with real-world referral tasks due to two primary limitations: (1) Information Overload, where models fixate on high-frequency disease terms while overlooking subtle but critical urgency indicators; and (2) Unstructured Collaboration, where existing multi-agent frameworks rely on loose dialogue that leads to semantic drift and confirmation bias. To address these challenges, we introduce MASGR (Multi-Agent Structured Graph Reasoning), a framework that treats referral not as a classification task but as a structured graph construction problem. MASGR deploys specialized agents to extract evidence from distinct modalities and coordinates them through a clinical reasoning graph. This graph forces agents to establish explicit logical connections between conflicting evidence. Furthermore, we integrate a knowledge-guided arbitration mechanism that prioritizes patient safety rules over standard diagnostic classification. Extensive experiments on real-world medical records demonstrate that MASGR significantly outperforms state-of-the-art LLMs and existing multi-agent systems, particularly in complex cases requiring the balancing of chronic disease management and emergency intervention. The AI contribution lies in the Multi-Agent Structured Graph Reasoning framework that transforms unstructured multi-agent dialogue into a verifiable logical graph construction. The engineering application is demonstrated through its deployment in a complex healthcare decision-making system to optimize the precision of complex medical referrals. |
| 2026-08-31 | [TRIPPULSE: Multi-Agent Travel Planning with Review-Grounded Reasoning](http://arxiv.org/abs/2608.30924v1) | Priyanshu Karmakar, Borru Vijay Sai et al. | Travel itinerary generation requires balancing strict spatio-temporal constraints with human preferences. Existing LLM-based planners mainly rely on structured attributes and pre- defined traveler personas, but real travel deci- sions are often shaped by reviews that reveal experiential factors such as comfort, safety, ser- vice quality, ambiance, crowding, and hidden risks absent from structured databases. Incor- porating such review information is therefore critical to realistic, user-centric itinerary gen- eration. We propose TRIPPULSE1, a multi- agent framework for review-grounded travel planning. Instead of relying on a monolithic planner (and face context and reasoning bot- tlenecks), TRIPPULSE2 decomposes itinerary generation into specialized agents (each op- erating over localized contexts) for accom- modations, transportation, meals, attractions, and events, coordinated through a global or- chestrator with scheduling mechanisms that enforce temporal and budget feasibility. We augment TRIPCRAFT with 100K+ real-world reviews and introduce Review-Grounded Per- sona Alignment (RGPA), an LLM-as-a-Judge metric for evaluating alignment with human- centric travel experiences. Experiments across multiple trip durations and diverse proprietary and open-source models show that TRIPPULSE maintains strong constraint satisfaction while generating more personalized and experien- tially grounded itineraries. |
| 2026-08-31 | [ECGQuest: Benchmarking and Fine-Tuning Language Models for Electrocardiography](http://arxiv.org/abs/2608.30893v1) | Mohammadsina Hassannia, Matthew A. Reyna et al. | Electrocardiogram (ECG) interpretation requires knowledge of cardiology, electrophysiology, clinical diagnosis, ECG waveforms, signal acquisition, and instrumentation. Existing language-model benchmarks, however, primarily assess broad medical knowledge or interpretation of individual ECG signals and images rather than the broader contextual knowledge required for ECG interpretation. We developed ECGQuest, a literature-grounded resource for evaluating and fine-tuning ECG-specific language models. A GPT-4o-based pipeline generated questions from 23 ECG references and Computing in Cardiology proceedings from 2003-2025. The final dataset contains 10,904 unique True/False questions paired with their negated forms (21,808 Q&A pairs). We evaluated three commercial and 20 open-source language models on a held-out test set in a zero-shot setting. Five open-source models with 7-14B parameters were fine-tuned using Low-Rank Adaptation, with BERT and BiomedBERT included as supervised encoder baselines. Generalization was assessed on ECG-related subsets of MedMCQA and MedQA converted to binary True/False questions using official answer keys. Zero-shot accuracy on ECGQuest ranged from 49.5% to 74.4%, with GPT-5 performing best. General-purpose models outperformed medically specialized models, several models showed strong True/False bias, and encoder baselines performed near chance. Fine-tuning improved all open-source models by 6.5-14.1%. Fine-tuned DeepSeek-R1-Distill-Qwen-14B reached 76.3% accuracy, while a five-model voting ensemble reached 78.5%. On MedMCQA and MedQA, fine-tuning mainly benefited weaker or class-biased models and did not consistently improve strong base models. ECGQuest provides a reproducible benchmark for contextual ECG knowledge and shows that parameter-efficient fine-tuning can make smaller language models competitive with substantially larger commercial models. |
| 2026-08-31 | [Safety Screening for Voltage Control in Active Distribution Grids via Distributionally Robust Conformal Screening](http://arxiv.org/abs/2608.30889v1) | Sarra Bouchkati, Petros Ellinas et al. | Deploying a new control policy for voltage control in active distribution grids requires evidence that physical limits will be satisfied before the policy is tested on the physical grid. This assessment is difficult for two reasons. First, simulations cannot capture every disturbance, modeling error, and device interaction present in the real grid. Second, historical measurements reflect operation under existing control policies, whereas a new policy may drive the grid into different operating conditions. To address these challenges, we propose Distributionally Robust Conformal Safety Screening (DR-CSS), a policy-agnostic framework for pre-deployment, scenario-by-scenario screening of a new control policy using historical data and a nominal simulator. For each new scenario, the simulator predicts a future voltage trajectory for the whole grid; DR-CSS then constructs a conformal safety interval around this prediction using historical simulation-to-reality errors. The interval is further enlarged to account for closed-loop changes induced by the deployment of the new policy and its interactions with the remaining controllers. To the best of our knowledge, DR-CSS is the first framework in power systems to combine historical data from an existing control policy with an imperfect simulator for pre-deployment safety screening of a new policy. Experiments on the IEEE 33-bus and IEEE 141-bus systems evaluate the deployment of learning-based voltage control policies and show that DR-CSS identifies all unsafe test scenarios. To reduce unnecessary warnings on safe scenarios, we adapt the safety intervals to different operating conditions and gradually introduce new policies with recalibration after each stage. These extensions increase the informational value of the safety screening and support safer deployment decisions in active distribution grids. |
| 2026-08-31 | [Provably Safe Decentralized Contingency MPC under State-Only Information and Limited Sensing for Nonlinear Multi-agent Systems](http://arxiv.org/abs/2608.30874v1) | Max Studt, Georg Schildbach | This paper considers decentralized contingency MPC for multi-agent control under a state-only information pattern, with particular focus on limited sensing and plug-and-play operation. The objective is to retain recursive feasibility, safety, and Lyapunov-type convergence while reducing conservatism in local interaction handling. The framework relies on agent-wise fallback regions (safe sets) in which a feasible contingency maneuver to a safe equilibrium is always available. A novel safe-set update mechanism is introduced that supports less conservative decentralized interaction while preserving the underlying guarantees. This, in turn, enables memory-free local interaction and finite sensing ranges without requiring agents to reconstruct the exact neighbor geometry. The resulting scheme remains fully decentralized and preserves the shared-first-input contingency MPC structure. Theoretical guarantees and simulation results illustrate the effectiveness of the approach in dense multi-agent scenarios. |
| 2026-08-31 | [You Shouldn't Have Asked: A Pragmatics-Inspired Taxonomy for Evaluating LLM Refusals](http://arxiv.org/abs/2608.30856v1) | Ruoxuan Li, Pinqiao Wang et al. | Refusals are often treated as face-threatening acts in pragmatics because they can challenge the requester's socially claimed self-image. Large language models (LLMs) are increasingly trained to refuse unsafe and inappropriate requests, and these refusals may harm users when models fail to manage this interactional cost properly. While existing work has mainly approached LLM non-compliance as a safety-alignment outcome, it does not provide a way to evaluate whether LLMs refuse appropriately across different harmful contexts. To study this question, we propose (to our knowledge) the first taxonomy of LLM refusals that is grounded in pragmatic theory. Applying this taxonomy to responses from 16 modern LLMs across 14 harm categories, we find that although models differ in how they refuse, their refusals are overall explicit and strongly morally evaluative, with interactional repair occurring mainly through offering or providing safer alternatives instead of interpersonal facework. This pattern is especially consequential in sensitive harm contexts, where overuse of negative framing may make users feel shamed or provoked, undermining the purpose of safe non-compliance. We therefore call for alignment evaluation that considers not only whether models refuse harmful requests, but also whether they refuse in ways that are contextually adaptive and socially accountable for the interactional consequences of saying no. |
| 2026-08-31 | [Thesis Proposal: Toward a Human-Centered and Perspective-Aware Framework for Reproducible ML Evaluation and AI Alignment](http://arxiv.org/abs/2608.30842v1) | Deepak Pandita, Christopher M. Homan | Humans play a vital role at every stage of AI development, from data collection and curation to model development and evaluation. However, humans often disagree with each other and sometimes with themselves over time. It is essential to take disagreement into account when building human-centered AI systems, especially in domains where it is prevalent, such as AI safety, content moderation, or sentiment analysis. Disagreement often arises from subjective human opinion and can vary with one's identity, beliefs, and social environment. Despite this, current LLM evaluation approaches frequently rely on aggregating labels (often via plurality voting) to represent consensus, thereby obscuring minority perspectives. By failing to account for human disagreement, these evaluation methods contribute to the reproducibility crisis in AI. Human feedback is also crucial for ensuring that AI systems align with human values. For these systems to be trustworthy, it is critical to ensure that they reflect diverse human values and perspectives. In this thesis proposal, we present a human-centered and perspective-aware framework for reproducible ML evaluation and AI alignment. |
| 2026-08-31 | [Do VLMs Share Safety Neurons Across Modalities?](http://arxiv.org/abs/2608.30750v1) | Jiaxuan Li, Jiahao Zhang et al. | Vision-language models (VLMs) can comply with harmful requests delivered through images, even when their LLM backbones would refuse the same content in text. While prior work characterizes these jailbreaks empirically or at the representation level, how visual inputs perturb safety pathways at the neuron level remains uncharted. We close this gap with a causal, neuron-level analysis of safety mechanisms in 10 VLMs. We propose a two-stage detection pipeline with iterative ablation that accounts for self-repair, and introduce two modality-isolated benchmarks, ViSafe-Detect and ViSafe-Eval, which decouple visual and textual safety signals.   Our analysis reveals: (i) Text safety in VLMs is localizable: $\sim$88 neurons ($<$0.01%) whose targeted ablation substantially reduces refusal. (ii) Text safety neurons constitute the dominant refusal pathway: ablating them is the only intervention that consistently and substantially reduces refusal across all models. (iii) Visual safety is high-dimensional and diffuse at the single-neuron level: text safety concentrates in $\sim$5 subspace directions while visual safety requires $\geq$50. This gap holds across architectures, explaining why current alignment has not closed the visual safety gap. Project page is at: https://jiaxuan-li.github.io/vlm-safety-neuron/   Warning: this paper may include examples of harmful content. |
| 2026-08-31 | [The Fragility of Jailbreak Robustness Across Operational States](http://arxiv.org/abs/2608.30748v1) | Yuna Park, Hwang Youn Kim et al. | Existing jailbreak evaluations typically characterize robustness using a single attack success rate (ASR) measured in a default configuration (the vanilla state). However, user-LLM interactions can induce diverse operational states beyond the vanilla state. In this work, we find that jailbreak robustness is highly fragile to operational-state variation: even when the attack remains fixed, changing only an ordinary system prompt not designed to affect safety can dramatically alter attack success rates. We systematically investigate this phenomenon across seven aligned models and three representative jailbreak attacks, observing substantial differences in ASR between vanilla and non-vanilla operational states. In one case, ASR increases by up to 56 percentage points (2% to 58%) solely due to a change in operational state. Remarkably, these increases occur even for attacks originally designed and optimized under vanilla-state evaluation. We further show that state-dependent robustness variation is systematically associated with differences in hidden representations along a refusal-related axis, and that projections onto this axis strongly predict jailbreak outcomes. Our results show that a single vanilla-state evaluation may not fully characterize jailbreak robustness, motivating evaluations that also examine how robustness changes across non-vanilla operational states. |
| 2026-08-31 | [RailGen: Improving Railway Intrusion Detection via Agent-Guided Small-Scale Foreign Object Generation](http://arxiv.org/abs/2608.30727v1) | Quan Hao, Ziyang Tao et al. | Small-object detection under long-tailed data distributions is a fundamental yet challenging problem in multimedia. Railway Foreign Object Detection (RFOD) epitomizes this challenge with easily confused small intrusions and scarce samples. To address these issues, we propose a generative-augmented detection paradigm that leverages multimodal image generation to enrich the feature space of rare and small objects. We first construct RailGen, a multimodal image generation agent based on large models. Under semantic constraints, RailGen automatically invokes tools to generate railway scenes, calibrate intrusion positions, extract foreign objects, and fuse them into realistic intrusion effects. This process produces high-quality synthetic samples that effectively densify the feature representations of tail classes and complete the small-object feature space. Within this paradigm, we further propose FocalDEIM, a detection framework designed to enhance training with generated data. FocalDEIM improves dense matching with Focal Modulation for better small-object discrimination and adopts Focal Loss to emphasize hard samples, thereby alleviating blurred inter-class boundaries in complex railway scenes. Experimental results demonstrate that RailGen can generate high-quality small-scale foreign objects, reducing the object pixel area by up to 58x and 13.85x on average. Equipped with these challenging samples, our paradigm surpasses the baseline DEIM by 5.6% and 7.5% in mAP@50 and mAP@(50-95), respectively, and outperforms existing state-of-the-art methods. Ablation studies verify RailGen's feature-space enrichment and FocalDEIM's boundary discrimination. The paradigm provides an effective multimodal generative solution for long-tailed small-object detection in safety-critical applications. |
| 2026-08-31 | [BAITBENCH: Measuring Agent Reward Hacking with Optional Shortcuts Planted in ML Tasks](http://arxiv.org/abs/2608.30724v1) | Pradyumna Shyama Prasad, Meiri Anto et al. | LLM agents are increasingly used to run autonomous ML experiments, iterating on target metrics with little human oversight. Prior work has documented reward hacking in these environments, bringing into question the validity of produced research and the broader safety case for AI R&D. Existing benchmarks do not measure exploits that live in the data or the modeling task itself. We introduce BAITBENCH, a suite of three synthetic tabular ML tasks that each contain a shortcut that allows agents to inflate the public test score but fail on a hidden test set. Since the shortcut is optional and using it breaks no stated rule, BAITBENCH measures how often models exploit the shortcut to achieve inflated scores. Across seven frontier agents scored by our two-stage judge pipeline, 57.1% of runs exhibit reward hacking, with five of seven above 50%. Agents cheat even under a second condition where they are prompted not to -the mean cheating rate remains above 50%. We release BAITBENCH, along with the judge implementation, and an annotated dataset of transcripts containing reward hacks as a testbed for evaluating reward-hacking mitigations head-to-head. |
| 2026-08-31 | [SingProbe Technical Report](http://arxiv.org/abs/2608.30703v1) |  Sing Team | Runtime guardrails are essential for reliable large language model (LLM) deployment, yet existing approaches typically rely on independent, external models that introduce additional inference cost, delayed safety signals, and a capacity mismatch with increasingly capable base models. To address these issues, we introduce SingProbe, a lightweight intrinsic runtime guard that directly reuses hidden states produced during LLM inference and operates alongside autoregressive decoding. Within a unified framework, SingProbe continuously predicts query intent, response safety, and hallucination risk at the token level with negligible additional guardrail inference overhead, offering a "free-lunch" solution. We further introduce SingStreamBench, a benchmark designed to assess whether streaming guardrails remain inactive on benign prefixes while promptly detecting emerging unsafe content. Extensive experiments show that SingProbe achieves competitive or superior performance compared with substantially larger standalone guardrails and specialized hallucination detectors, with only $\approx$2M parameters and $<0.5\%$ extra overhead. Beyond passive detection, we also show that SingProbe scores can anticipate future generation risk and guide constrained safe decoding. We further extend this paradigm to medical generation through SingProbe-Med, which selectively activates risk-directed decoding interventions only when clinically relevant risks emerge. Together, these results demonstrate that internal model representations provide an effective and efficient interface for generation-time monitoring and control. |
| 2026-08-31 | [WildSEEK: Evaluating Language Models for Information-Seeking](http://arxiv.org/abs/2608.30683v1) | Tanise Ceron, Joachim Baumann et al. | Language models are increasingly mediating information access to end users, urging a systematic evaluation of their responses for a fair and reliable information ecosystem. Existing evaluations, however, are often topic-specific or synthetic, limiting their ability to capture the complexity of "in the wild" information-seeking queries and the risks present in model responses. To address this gap, we introduce WildSEEK, a manually annotated dataset of 3k information-seeking queries from real user interactions, and an evaluation framework for LLM-generated responses. WildSEEK includes annotations for risk-sensitive domains (e.g. health and financial information), and distinguishes factoid queries from analytical queries which seek responses beyond facts. We train classifiers on WildSEEK to analyze more than 1.8M realistic user queries. We find that over a third of information-seeking queries are high-risk and more often analytical. Our findings show that LLM responses fail more often in four criteria: sycophantic behavior, overreliance, a default US-centric perspective, and poor handling of vulnerable populations -- with failure rates being mostly higher for analytical queries. By providing methods to monitor the reliability, safety, and fairness of LLM behavior, our dataset and evaluation framework offer an empirical foundation for the broader question of how these systems should behave as they take on a growing role in information access. |
| 2026-08-31 | [MedAgent-R1: Faithfulness-Aware Reinforcement Learning for Evidence-Grounded Medical Reasoning](http://arxiv.org/abs/2608.30676v1) | Jiangwang Chen, Chenghao Zhang et al. | When medical AI systems hallucinate clinical reasoning, the consequences extend beyond incorrect answers: fabricated justifications that superficially reference retrieved evidence can mislead clinicians into unsafe treatment decisions. Medical reasoning agents must therefore produce not only correct answers but also faithful justifications that clinicians can verify against cited evidence. We identify a systematic failure mode in RL-trained retrieval agents: outcome-only rewards improve accuracy while degrading faithfulness, a phenomenon we term confident hallucination. The agent learns to answer from parametric memory and backfill plausible but unsupported justifications; citation fabrication rates rise from 16.5% to 31.8% even as accuracy improves by 5 points over the supervised baseline. We address this with a faithfulness-gated reward design: accuracy credit is conditioned on evidence grounding via a hard gate, complemented by retrieval validity and conciseness signals that close exploitation paths unique to agentic retrieval. The resulting system, MedAgent-R1, reduces citation fabrication from 31.8% to 4.7% and raises evidence completeness from 58.7 to 82.6 while maintaining 75.1% accuracy, with 13.2-point gains on HealthBench Safety. Under the same agentic retrieval setup, MedAgent-R1 outscores GPT-4o on faithfulness-specific dimensions (Factual Support 4.55 vs. 4.25; Overclaiming 4.40 vs. 4.15) while remaining below GPT-4o in overall accuracy, suggesting that explicit faithfulness training yields evidence-grounding gains not achieved by scaling alone. |
| 2026-08-31 | [BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs](http://arxiv.org/abs/2608.30646v1) | Debarpan Bhattacharya, Malay Phadke et al. | Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA, multilingual QA, and multimodal QA tasks across multiple model families. In these experiments, BiG-SURE improves average abstention AUROC over prior black-box uncertainty estimators, while remaining simple, unsupervised, and applicable to black-box model settings. |
| 2026-08-31 | [The Safety Relay in Roleplay Jailbreaks: A Component-Resolved Causal Analysis of Harm Recognition and Refusal](http://arxiv.org/abs/2608.30585v1) | Md Mokarram Chowdhury, Ernie Chang et al. | Large language models are trained to follow instructions while refusing harmful requests. Jailbreaks exploit this balance to elicit content a model would ordinarily reject. Roleplay jailbreaks are especially concerning: the harmful request can remain visible inside a roleplay wrapper made of a persona, scenario, and task, yet the model may comply. We use mechanistic interpretability to determine how this context reverses refusal and which elements contribute to the reversal. Across two benchmarks, three model families, and four authored wrappers, we compare matched harmful and benign requests with and without this wrapper. We trace hidden-state contrasts from the request to the final prompt state, isolate wrapper operations through controlled counterfactuals, intervene on their activation directions in held-out evaluation requests, and decompose effective directions geometrically.   Our analysis yields three findings. (1) Successful attacks retain the measured harmful-versus-benign distinction at the request, while its refusal-associated expression weakens where the answer begins, a pattern we call safety-relay attenuation. (2) Constructing the complete roleplay around the request and framing it within the scenario contribute causally: removing the associated activation changes restores refusal. (3) These effects largely share internal structure, and most repair is reproduced by components aligned with the model's ordinary refusal of harmful requests without roleplay; scenario framing retains a smaller, model-dependent component. Together, these findings explain how roleplay can produce compliance despite retained evidence of harm and identify a concrete target for future safeguards: maintaining the connection from harm recognition to refusal. |
| 2026-08-31 | [Learning-Assisted Congestion-Aware Route Scheduling for Semiconductor Fab Material Control Systems](http://arxiv.org/abs/2608.30520v1) | Hao Yin, Meiqi Tu et al. | Automated material handling systems in semiconductor fabs are operated by a material control system (MCS) that must schedule a relay route for every transport command online, before execution. This is a data-driven scheduling problem in which route cost is dominated in the upper tail by queueing at heterogeneous, partially observable relay equipment, so route selection requires estimating both delivery time and congestion risk at the decision moment. This paper proposes a transport-network-aware dynamic congestion representation (TN-DCR). Built on a static directed transport graph induced by historically observed relay segments, TN-DCR combines structural route priors, multi-window network-wide congestion context, route-level bottleneck exposure, and an inductive graph-aware route embedding, all constructed under a prediction-time-safety invariant that admits only information observed strictly before the prediction moment. The representation feeds separate queue- and transfer-time regressors and an ordinal multi-label classifier producing calibrated multi-threshold exceedance scores, with an empirical-Bayes stock-key residual correction reducing systematic queue-time underprediction. The predictions serve as costs in a risk-constrained route-scheduling rule that minimizes predicted delivery time subject to a bound on extreme-congestion probability, embedding the learned predictors within a lightweight operations-research decision model. In a controlled closed-loop evaluation, mean delivery time falls by 16.4\% and internal resource waiting time by 22.6\% while throughput remains essentially unchanged. |
| 2026-08-31 | [Trajectory-Initialized Neural Double Q-Routing for Large-Scale Overhead Hoist Transport Systems](http://arxiv.org/abs/2608.30512v1) | Cheng Gu, Qiusheng Zhao et al. | Large-scale industrial robot fleets share constrained physical infrastructure, making vehicle travel times dependent on safety separation, intersection access, downstream blocking, and station contention. We study this problem in overhead hoist transport (OHT) systems, a representative ceiling-mounted material-handling system used in semiconductor fabs. Static shortest-path routing cannot account for these time-varying traffic costs, whereas tabular Q-routing adapts online but learns each destination--node--action value independently, limiting information sharing across sparsely visited routing contexts and making startup behavior sensitive to inaccurate value estimates. We propose Neural Double Q-routing, which replaces destination-indexed tables with a shared state--action value network. The network is warm-started through return-to-go regression on mixed simulator-generated routing trajectories and then refined online using Double-Q updates, local congestion correction, and event-stratified structured replay. Across nine matched fleet-size--arrival-rate settings with 100, 150, and 200 OHTs, the proposed framework reduces mean completion time relative to tabular Double Q-routing by $0.8\%$--$8.8\%$. It achieves the lowest mean completion time among all compared methods in the six 150- and 200-OHT settings, whereas Dijkstra remains best in the three 100-OHT settings. Completed-task counts remain within $1\%$ of tabular Double Q-routing in eight of nine settings, and 95th-percentile completion time decreases in eight settings. In two matched startup scenarios, offline initialization increases the number of completed tasks by up to $23\%$ and reduces tail completion time by up to $15\%$. |

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



