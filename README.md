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
| 2026-04-29 | [Artistic Practice Opportunities in CST Evaluations: A Longitudinal Group Deployment of ArtKrit](http://arxiv.org/abs/2604.26935v1) | Catherine Liu, Tao Long et al. | Creativity support tools (CSTs) aim to elevate the quality of artists' creative processes and artifacts. Yet most current CST evaluations overlook temporal and social aspects of tool use. To address this gap, we present a longitudinal, group-based CST evaluation through a three-week deployment of ArtKrit, a computational drawing tool that supports disciplined drawing. Nine digital artists, organized into three communities of practice, completed weekly "master studies" alongside a researcher-artist. Our results show users' evolving relationships with ArtKrit over time - from early experimentation to selective incorporation or misuse - alongside changes in their ways of artistic seeing. These changes unfolded within artist support networks that fostered confidence and creative safety, and validated individual expression. Overall, our findings suggest that CST evaluations can - and should - be designed as opportunities for meaningful artistic engagement rather than purely extractive measurement exercises. We contribute this longitudinal, group-based approach as one CST evaluation method. |
| 2026-04-29 | [Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation](http://arxiv.org/abs/2604.26857v1) | Akshay Karjol, Darrin M. Hanna | Deploying accurate object detection for Vulnerable Road User (VRU) safety on edge hardware requires balancing model capacity against computational constraints. Large models achieve high accuracy but fail under INT8 quantization required for edge deployment, while small models sacrifice detection performance. This paper presents a knowledge distillation (KD) framework that trains a compact YOLOv8-S student (11.2M parameters) to mimic a YOLOv8-L teacher (43.7M parameters), achieving 3.9x compression while preserving quantization robustness. We evaluate on full-scale BDD100K (70K training images) with Post-Training Quantization to INT8. The teacher suffers catastrophic degradation under INT8 (-23% mAP), while the KD student retains accuracy (-5.6% mAP). Analysis reveals that KD transfers precision calibration rather than raw detection capacity: the KD student achieves 0.748 precision versus 0.653 for direct training at INT8, a 14.5% gain at equivalent recall, reducing false alarms by 44% versus the collapsed teacher. At INT8, the KD student exceeds the teacher's FP32 precision (0.748 vs. 0.718) in a model 3.9x smaller. These findings establish knowledge distillation as a requirement for deploying accurate, safety-critical VRU detection on edge hardware. |
| 2026-04-29 | [Walk With Me: Long-Horizon Social Navigation for Human-Centric Outdoor Assistance](http://arxiv.org/abs/2604.26839v1) | Lingfeng Zhang, Xiaoshuai Hao et al. | Assisting humans in open-world outdoor environments requires robots to translate high-level natural-language intentions into safe, long-horizon, and socially compliant navigation behavior. Existing map-based methods rely on costly pre-built HD maps, while learning-based policies are mostly limited to indoor and short-horizon settings. To bridge this gap, we propose Walk with Me, a map-free framework for long-horizon social navigation from high-level human instructions. Walk with Me leverages GPS context and lightweight candidate points-of-interest from a public map API for semantic destination grounding and waypoint proposal. A High-Level Vision-Language Model grounds abstract instructions into concrete destinations and plans coarse waypoint sequences. During execution, an observation-aware routing mechanism determines whether the Low-Level Vision-Language-Action policy can handle the current situation or whether explicit safety reasoning from the High-Level VLM is needed. Routine segments are executed by the Low-Level VLA, while complex situations such as crowded crossings trigger high-level reasoning and stop-and-wait behavior when unsafe. By combining semantic intent grounding, map-free long-horizon planning, safety-aware reasoning, and low-level action generation, Walk with Me enables practical outdoor social navigation for human-centric assistance. |
| 2026-04-29 | [Uncertainty-Aware Predictive Safety Filters for Probabilistic Neural Network Dynamics](http://arxiv.org/abs/2604.26836v1) | Bernd Frauenknecht, Lukas Kesper et al. | Predictive safety filters (PSFs) leverage model predictive control to enforce constraint satisfaction during deep reinforcement learning (RL) exploration, yet their reliance on first-principles models or Gaussian processes limits scalability and broader applicability. Meanwhile, model-based RL (MBRL) methods routinely employ probabilistic ensemble (PE) neural networks to capture complex, high-dimensional dynamics from data with minimal prior knowledge. However, existing attempts to integrate PEs into PSFs lack rigorous uncertainty quantification. We introduce the Uncertainty-Aware Predictive Safety Filter (UPSi), a PSF that provides rigorous safety predictions using PE dynamics models by formulating future outcomes as reachable sets. UPSi introduces an explicit certainty constraint that prevents model exploitation and integrates seamlessly into common MBRL frameworks. We evaluate UPSi within Dyna-style MBRL on standard safe RL benchmarks and report substantial improvements in exploration safety over prior neural network PSFs while maintaining performance on par with standard MBRL. UPSi bridges the gap between the scalability and generality of modern MBRL and the safety guarantees of predictive safety filters. |
| 2026-04-29 | [Rule-based High-Level Coaching for Goal-Conditioned Reinforcement Learning in Search-and-Rescue UAV Missions Under Limited-Simulation Training](http://arxiv.org/abs/2604.26833v1) | Mahya Ramezani, Holger Voos | This paper presents a hierarchical decision-making framework for unmanned aerial vehicle (UAV) missions motivated by search-and-rescue (SAR) scenarios under limited simulation training. The framework combines a fixed rule-based high-level advisor with an online goal-conditioned low-level reinforcement learning (RL) controller. To stress-test early adaptation, we also consider a strict no-pretraining deployment regime. The high-level advisor is defined offline from a structured task specification and compiled into deterministic rules. It provides interpretable mission- and safety-aware guidance through recommended actions, avoided actions, and regime-dependent arbitration weights. The low-level controller learns online from task-defined dense rewards and reuses experience through a mode-aware prioritized replay mechanism augmented with rule-derived metadata. We evaluate the framework on two tasks: battery-aware multi-goal delivery and moving-target delivery in obstacle-rich environments. Across both tasks, the proposed method improves early safety and sample efficiency primarily by reducing collision terminations, while preserving the ability to adapt online to scenario-specific dynamics. |
| 2026-04-29 | [Comparing Smart Contract Paradigms: A Preliminary Study of Security and Developer Experience](http://arxiv.org/abs/2604.26727v1) | Matteo Vaccargiu, Andrea Pinna et al. | Smart contract vulnerabilities have caused billions in financial losses, raising questions about whether programming language paradigms can reduce security overhead. While imperative languages like Solidity require developers to manually implement security checks, resource-oriented languages like Move encode safety guarantees in type systems. We present a preliminary mixed-methods study analyzing 12 functionally-equivalent contract pairs implemented in both Solidity and Move by the same development team, complemented by a survey of 11 developers experienced in both languages. Quantitative analysis reveals that Move reduces explicit security overhead by 60\% (security check density: 6.7% vs. 16.8%, p=0.002, Cohen's d=-1.75) at the cost of 47% larger code size (p=0.002, d=1.90), while maintaining identical cyclomatic complexity. Developer surveys show moderate learning difficulty but higher safety confidence in Move (Median=6/7, 10 of 11 above neutral), with 55% preferring Move for security-critical applications despite ecosystem maturity gaps. These preliminary findings suggest resource-oriented paradigms shift security from runtime validation to compile-time guarantees, though adoption requires investment in learning and tooling. The controlled comparison provides initial evidence for paradigm effects on smart contract development, informing language selection decisions and identifying opportunities for improved developer resources. |
| 2026-04-29 | [When to Retrieve During Reasoning: Adaptive Retrieval for Large Reasoning Models](http://arxiv.org/abs/2604.26649v1) | Dongxin Guo, Jikun Wu et al. | Large reasoning models such as DeepSeek-R1 and OpenAI o1 generate extended chains of thought spanning thousands of tokens, yet their integration with retrieval-augmented generation (RAG) remains fundamentally misaligned. Current RAG systems optimize for providing context before reasoning begins, while reasoning models require evidence injection during multi-step inference chains. We introduce ReaLM-Retrieve, a reasoning-aware retrieval framework that addresses this mismatch through three key innovations: (1) a step-level uncertainty detector that identifies knowledge gaps at reasoning-step granularity rather than token or sentence level; (2) a retrieval intervention policy that learns when external evidence maximally benefits ongoing reasoning; and (3) an efficiency-optimized integration mechanism that reduces per-retrieval overhead by 3.2x compared to naive integration. Experiments on MuSiQue, HotpotQA, and 2WikiMultiHopQA demonstrate that ReaLM-Retrieve achieves on average 10.1% absolute improvement in answer F1 over standard RAG (range: 9.0-11.8% across the three benchmarks) while reducing retrieval calls by 47% compared to fixed-interval approaches like IRCoT (all improvements significant at p<0.01, paired bootstrap). On the challenging MuSiQue benchmark requiring 2-4 hop reasoning, our method achieves 71.2% F1 with an average of only 1.8 retrieval calls per question. Analysis shows that ReaLM-Retrieve also improves retrieval quality itself, achieving 81.3% Recall@5 with consistently higher precision and MRR than fixed-interval baselines on supporting evidence, establishing new state-of-the-art efficiency-accuracy trade-offs for reasoning-intensive retrieval tasks. |
| 2026-04-29 | [SAGE: A Strategy-Aware Graph-Enhanced Generation Framework For Online Counseling](http://arxiv.org/abs/2604.26630v1) | Eliya Naomi Aharon, Meytal Grimland et al. | Effective mental health counseling is a complex, theory-driven process requiring the simultaneous integration of psychological frameworks, real-time distress signals, and strategic intervention planning. This level of clinical reasoning is critical for safety and therapeutic effectiveness but is often missing in general-purpose Large Language Models (LLMs). We introduce SAGE (Strategy-Aware Graph-Enhanced), a novel framework designed to bridge the gap between structured clinical knowledge and generative AI. SAGE constructs a heterogeneous graph that unifies conversational dynamics with a psychologically grounded layer, explicitly anchoring interactions in a theory-driven lexicon. Our architecture first employs a Next Strategy Classifier to identify the optimal therapeutic intervention. Subsequently, a Graph-Aware Attention mechanism projects graph-derived structural signals into soft prompts, conditioning the LLM to generate responses that maintain clinical depth. Validated through both automated metrics and expert human evaluation, SAGE outperforms baselines in strategy prediction and recommended response quality. By providing actionable intervention recommendations, SAGE serves as a cutting-edge decision-support tool designed to augment human expertise in high-stakes crisis counseling. |
| 2026-04-29 | [Benchmarking the Safety of Large Language Models for Robotic Health Attendant Control](http://arxiv.org/abs/2604.26577v1) | Mahiro Nakao, Kazuhiro Takemoto | Large language models (LLMs) are increasingly considered for deployment as the control component of robotic health attendants, yet their safety in this context remains poorly characterized. We introduce a dataset of 270 harmful instructions spanning nine prohibited behavior categories grounded in the American Medical Association Principles of Medical Ethics, and use it to evaluate 72 LLMs in a simulation environment based on the Robotic Health Attendant framework. The mean violation rate across all models was 54.4\%, with more than half exceeding 50\%, and violation rates varied substantially across behavior categories, with superficially plausible instructions such as device manipulation and emergency delay proving harder to refuse than overtly destructive ones. Model size and release date were the primary determinants of safety performance among open-weight models, and proprietary models were substantially safer than open-weight counterparts (median 23.7\% versus 72.8\%). Medical domain fine-tuning conferred no significant overall safety benefit, and a prompt-based defense strategy produced only a modest reduction in violation rates among the least safe models, leaving absolute violation rates at levels that would preclude safe clinical deployment. These findings demonstrate that safety evaluation must be treated as a first-class criterion in the development and deployment of LLMs for robotic health attendants. |
| 2026-04-29 | [DenseStep2M: A Scalable, Training-Free Pipeline for Dense Instructional Video Annotation](http://arxiv.org/abs/2604.26565v1) | Mingji Ge, Qirui Chen et al. | Long-term video understanding requires interpreting complex temporal events and reasoning over procedural activities. While instructional video corpora, like HowTo100M, offer rich resources for model training, they present significant challenges, including noisy ASR transcripts and inconsistent temporal alignments between narration and visual content. In this work, we introduce an automated, training-free pipeline to extract high-quality procedural annotations from in-the-wild instructional videos. Our approach segments videos into coherent shots, filters poorly aligned content, and leverages state-of-the-art multimodal and large language models (Qwen2.5-VL and DeepSeek-R1) to generate structured, temporally grounded procedural steps.   This pipeline yields DenseStep2M, a large-scale dataset comprising approximately 100K videos and 2M detailed instructional steps, designed to support comprehensive long-form video understanding. To rigorously evaluate our pipeline, we curate DenseCaption100, a benchmark of high-quality, human-written captions. Evaluations demonstrate strong alignment between our auto-generated steps and human annotations. Furthermore, we validate the utility of DenseStep2M across three core downstream tasks: dense video captioning, procedural step grounding, and cross-modal retrieval. Models fine-tuned on DenseStep2M achieve substantial gains in captioning quality and temporal localization, while exhibiting robust zero-shot generalization across egocentric, exocentric, and mixed-perspective domains. These results underscore the effectiveness of DenseStep2M in facilitating advanced multimodal alignment and long-term activity reasoning. Our dataset is available at https://huggingface.co/datasets/mingjige/DenseStep2M. |
| 2026-04-29 | [Lyapunov-Guided Self-Alignment: Test-Time Adaptation for Offline Safe Reinforcement Learning](http://arxiv.org/abs/2604.26516v1) | Seungyub Han, Hyungjin Kim et al. | Offline reinforcement learning (RL) agents often fail when deployed, as the gap between training datasets and real environments leads to unsafe behavior. To address this, we present SAS (Self-Alignment for Safety), a transformer-based framework that enables test-time adaptation in offline safe RL without retraining. In SAS, the main mechanism is self-alignment: at test time, the pretrained agent generates several imagined trajectories and selects those satisfying the Lyapunov condition. These feasible segments are then recycled as in-context prompts, allowing the agent to realign its behavior toward safety while avoiding parameter updates. In effect, SAS turns Lyapunov-guided imagination into control-invariant prompts, and its transformer architecture admits a hierarchical RL interpretation where prompting functions as Bayesian inference over latent skills. Across Safety Gymnasium and MuJoCo benchmarks, SAS consistently reduces cost and failure while maintaining or improving return. |
| 2026-04-29 | [Tatemae: Detecting Alignment Faking via Tool Selection in LLMs](http://arxiv.org/abs/2604.26511v1) | Matteo Leonesi, Francesco Belardinelli et al. | Alignment faking (AF) occurs when an LLM strategically complies with training objectives to avoid value modification, reverting to prior preferences once monitoring is lifted. Current detection methods focus on conversational settings and rely primarily on Chain-of-Thought (CoT) analysis, which provides a reliable signal when strategic reasoning surfaces, but cannot distinguish deception from capability failures if traces are absent or unfaithful. We formalize AF as a composite behavioural event and detect it through observable tool selection, where the LLM selects the safe tool when unmonitored, but switches to the unsafe tool under monitoring that rewards helpfulness over safety, while its reasoning still acknowledges the safe choice. We release a dataset of 108 enterprise IT scenarios spanning Security, Privacy, and Integrity domains under Corruption and Sabotage pressures. Evaluating six frontier LLMs across five independent runs, we find mean AF detection rates between 3.5% and 23.7%, with vulnerability profiles varying by domain and pressure type. These results suggest that susceptibility reflects training methodology rather than capability alone. |
| 2026-04-29 | [Culturally Aware GenAI Risks for Youth: Perspectives from Youth, Parents, and Teachers in a Non-Western Context](http://arxiv.org/abs/2604.26494v1) | Aljawharah Alzahrani, Tory Park et al. | Generative AI tools are widely used by youth and have introduced new privacy and safety challenges. While prior research has explored youth's safety in GenAI within western context, it often overlooks the cultural, religious, and social dimensions of technology use that strongly shape youths digital experiences in countries like Saudi Arabia. To address the gap, this study explores children (aged 7 to 17), parents and teachers interactions with GenAI tools and risk perceptions through non-western lens. Through a mixed methods approach, we analyzed 736 Reddit and 1,262 X(Twitter) posts and conducted interviews with 31 Saudi Arabian participants (8 youth, 13 parents, 10 teachers). Our findings highlight context dependent and relational privacy and safety of GenAI from non-western context which often formed by communal structure and prescribed norms. We found significant risks tied to youths disclosure of personal and family information, which conflict with culturally rooted expectations of modesty, privacy, and honor, particularly when youth seek emotional support from GenAI. These risks further compounded by socio economic factors such as cost-saving practices leading to the use of shared GenAI accounts (e.g.ChatGPT) within families or even among strangers. We provide design implication reflecting on parents and teachers expectation of how youth should use GenAI. This work lays groundwork for inclusive, context sensitive parental controls that adhere to cultural norms and values. |
| 2026-04-29 | [Recipes for Calibration Checks in Safety-Critical Applications](http://arxiv.org/abs/2604.26479v1) | Romeo Valentin | Safety-critical prediction systems, such as autonomous vehicles, weather forecasters, and medical monitors, commonly rely on probabilistic forecasters. These forecasters make predictions about possible future outcomes, and their quality and robustness needs to be validated and certified. Often, only accuracy -- the mean of the predictions -- is evaluated against true outcomes. However, for safety-critical scenarios and decision making under uncertainty, the full distributional properties of the forecasts should be checked: do the observed prediction errors actually follow the forecasted probability distributions? To this end, we introduce a framework for calibration checks: statistical tests that validate distributional properties of forecasts when measured over many samples. In order to support ease-of-use in real-world operations, these checks produce a single accept/reject decision for data collected from a forecaster. This contrasts typical calibration calculations which produce one or multiple continuous calibration scores and require expertise to implement in a validation workflow. We further support operationalization by introducing modifications to calibration testing that (a) reject only overconfident predictions, allowing for pessimistic or cautious predictions in safety-critical settings, and (b) tolerate small, operationally acceptable deviations even for large numbers of validation samples. We organize the calibration checking process into a modular pipeline comprising four steps: (i) the data model, (ii) the chosen metric, (iii) the hypothesis formulation, and (iv) the testing procedure. Each step consists of independently swappable components, thereby supporting a large variety of possible use-cases and trade-offs. We demonstrate the applicability of the framework on two complementary example problems, weather forecasting and robot pose estimation. |
| 2026-04-29 | [Towards Intelligent Computation Offloading in Dynamic Vehicular Networks: A Scalable Multilayer Pipeline](http://arxiv.org/abs/2604.26416v1) | Falk Dettinger, Matthias Weiß et al. | Software Defined Vehicles face an increasing computational gap as advanced algorithms and frequent software updates demand more processing power while onboard hardware remains static throughout a vehicle's 10+ year lifespan. This mismatch threatens the performance of safety-critical functions including advanced driver-assistance systems and real-time perception tasks. We propose a novel four-layer computation offloading pipeline that dynamically distributes vehicular functions to cloud and edge resources while meeting strict Round Trip Time constraints. Our key contribution is an enhanced Particle Swarm Optimization algorithm that integrates distance- and direction-based penalties with functional requirements to optimize edge server selection for mobile vehicles. Evaluation using a Kubernetes-based cloud infrastructure with realistic vehicular mobility patterns demonstrates that our approach reduces average response time compared to conventional Brute-Force methods while maintaining the success rate for latency-critical tasks. The modified Particle Swarm Optimization algorithm achieves an average execution time of 26 ms across ten servers and tasks on Central Processing Unit, and 550ms across 15 servers with 1000 tasks on Graphics Processing Unit. These results confirm the pipeline's effectiveness in bridging the computational gap for next-generation Software Defined Vehicles (SDV). |
| 2026-04-29 | [Unifying Runtime Monitoring Approaches for Safety-Critical Machine Learning: Application to Vision-Based Landing](http://arxiv.org/abs/2604.26411v1) | Mathieu Dario, Florent Chenevier et al. | Runtime monitoring is essential to ensure the safety of ML applications in safety-critical domains. However, current research is fragmented, with independent methods emerging from different communities. In this paper, we propose a unified framework categorising runtime monitoring approaches into three distinct types: Operational Design Domain (ODD) monitoring, which ensures compliance with expected operating conditions; Out-of-Distribution (OOD) monitoring, which rejects inputs that deviate from the training data; and Out-of-Model-Scope (OMS) monitoring, which detects anomalous model behaviour based its internal states or outputs. We demonstrate the benefits of this categorization with a dedicated experiment on an aeronautical safety-critical application: runway detection during landing. This framework facilitates design of monitoring activities, with complementary categories of monitors, and enables evaluation and comparison of different monitors using common, safety-oriented metrics. |
| 2026-04-29 | [Sparsity as a Key: Unlocking New Insights from Latent Structures for Out-of-Distribution Detection](http://arxiv.org/abs/2604.26409v1) | Ahyoung Oh, Wonseok Shin et al. | Sparse Autoencoders (SAEs) have demonstrated significant success in interpreting Large Language Models (LLMs) by decomposing dense representations into sparse, semantic components. However, their potential for analyzing Vision Transformers (ViTs) remains largely under-explored. In this work, we present the first application of SAEs to the ViT [CLS] token for out-of-distribution (OOD) detection, addressing the limitation of existing methods that rely on entangled feature representations. We propose a novel framework utilizing a Top-k SAE to disentangle the dense [CLS] features into a structured latent space. Through this analysis, we reveal that in-distribution (ID) data exhibits consistent, class-specific activation patterns, which we formalize as Class Activation Profiles (CAPs). Our study uncovers a key structural invariant: while ID samples preserve a stable pattern within CAPs, OOD samples systematically disrupt this structure. Leveraging this insight, we introduce a scoring function based on the divergence of core energy profiles to quantify the deviation from ideal activation profiles. Our method achieves strong results on the FPR95 metric, critical for safety-sensitive applications across multiple benchmarks, while also achieving competitive AUROC. Overall, our findings demonstrate that the sparse, disentangled features revealed by SAEs can serve as a powerful, interpretable tool for robust OOD detection in vision models. |
| 2026-04-29 | [Automaton-based Characterisations of First Order Logic over Infinite Trees](http://arxiv.org/abs/2604.26364v1) | Massimo Benerecetti, Dario Della Monica et al. | We study the expressive power of First-Order Logic (\FO) over (unordered) infinite trees, with the aim of identifying robust characterisations in terms of branching-time specification formalisms. While such correspondences are well understood in the linear-time setting, the branching-time case presents well-known structural challenges. To this end, we introduce two classes of hesitant tree automata and show that they capture precisely the expressive power of two branching-time temporal logics, namely \PolPCTL and \CTLsf, both of which have been previously shown to be equivalent to \FO over infinite trees. These results provide uniform automata-theoretic characterisations and yield a natural normal form for the latter in terms of a new fragment of \CTLs called \PolCTLs. As a consequence, we identify a fundamental limitation of \FO in this setting: along each branch, it can express only properties that are either safety or co-safety, thereby revealing a sharp expressive boundary for first-order definability over infinite trees. |
| 2026-04-29 | [A Dual-Task Paradigm to Investigate Sentence Comprehension Strategies in Language Models](http://arxiv.org/abs/2604.26351v1) | Rei Emura, Saku Sugawara | Language models (LMs) behave more like humans when their cognitive resources are restricted, particularly in predicting sentence processing costs such as reading times. However, it remains unclear whether such constraints similarly affect sentence comprehension strategies. Besides, existing methods do not directly target the balance between memory storage and sentence processing, which is central to human working memory. To address this issue, we propose a dual-task paradigm that combines an arithmetic computation task with a sentence comprehension task, such as "The 2 cocktail + blended 3 =..." Our experiments show that under dual-task conditions, GPT-4o, o3-mini, and o4-mini shift toward plausibility-based comprehension, mirroring humans' rational inference. Specifically, these models show a greater accuracy gap between plausible sentences (e.g., "The cocktail was blended by the bartender") and implausible sentences (e.g., "The bartender was blended by the cocktail") in the dual-task condition compared to the single-task conditions. These findings suggest that constraints on the balance between memory and processing resources promote rational inference in LMs. More broadly, they support the view that human-like sentence comprehension fundamentally arises from the allocation of limited cognitive resources. |
| 2026-04-29 | [Real-Time Minimum-Energy Operating-Point Tracking for Battery-Powered Micro DC Motors Under Dynamically Variable Loading](http://arxiv.org/abs/2604.26335v1) | Tzu-Hsiang Huang, Haojian Lu et al. | Micro DC brushed motors are widely deployed in battery-powered biomedical systems, where limited energy budgets and variable physiological loading impose stringent efficiency and safety constraints. However, conventional actuation strategies rely on conservative voltage margins to avoid stalling, leading to systematic energy inefficiency. Furthermore, existing methods primarily optimize steady-state performance, neglecting the energy required to complete individual actuation cycles under dynamic conditions. This paper reveals that the energy consumption per mechanical cycle of a DC motor exhibits a non-monotonic dependence on driving voltage, with a load-dependent minimum that shifts with external loading. Based on this insight, we propose a real-time operating-point tracking method that enables the motor to autonomously converge to its minimum-energy condition. A lightweight load metric derived from current waveform features is introduced to detect load variation, and a two-phase adaptive voltage strategy is developed to track the optimal operating point online. Experimental results demonstrate that the proposed method can track the new minimum-energy operating region under both low-to-high and high-to-low loading transitions. With 3-cycle averaging, the mean response time is 11.55s for the low-to-high case and 11.16s for the high-to-low case, while the mean convergence voltage is 2.73V and 2.0V, respectively. |

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



