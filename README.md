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
| 2026-05-14 | [MetaBackdoor: Exploiting Positional Encoding as a Backdoor Attack Surface in LLMs](http://arxiv.org/abs/2605.15172v1) | Rui Wen, Mark Russinovich et al. | Backdoor attacks pose a serious security threat to large language models (LLMs), which are increasingly deployed as general-purpose assistants in safety- and privacy-critical applications. Existing LLM backdoors rely primarily on content-based triggers, requiring explicit modification of the input text. In this work, we show that this assumption is unnecessary and limiting. We introduce MetaBackdoor, a new class of backdoor attacks that exploits positional information as the trigger, without modifying textual content. Our key insight is that Transformer-based LLMs necessarily encode token positions to process ordered sequences. As a result, length-correlated positional structure is reflected in the model's internal computation and can be used as an effective non-content trigger signal.   We demonstrate that even a simple length-based positional trigger is sufficient to activate stealthy backdoors. Unlike prior attacks, MetaBackdoor operates on visibly and semantically clean inputs and enables qualitatively new capabilities. We show that a backdoored LLM can be induced to disclose sensitive internal information, including proprietary system prompts, once a length condition is satisfied. We further demonstrate a self-activation scenario, where normal multi-turn interaction can move the conversation context into the trigger region and induce malicious tool-call behavior without attacker-supplied trigger text. In addition, MetaBackdoor is orthogonal to content-based backdoors and can be composed with them to create more precise and harder-to-detect activation conditions.   Our results expand the threat model of LLM backdoors by revealing positional encoding as a previously overlooked attack surface. This challenges defenses that focus on detecting suspicious text and highlights the need for new defense strategies that explicitly account for positional triggers in modern LLM architectures. |
| 2026-05-14 | [Due Process on Hold: A Queueing Framework for Improving Access in SNAP](http://arxiv.org/abs/2605.15165v1) | Andrew Daw, Chloe Pache et al. | The U.S. social safety net delivers essential services at mass scale, but access burdens persist, as congested contact or call centers serve as a primary mode of application completion and assistance. In Holmes v. Knodell, Missouri's SNAP call centers were so congested that nearly half of all application denials were procedural, caused by applicants' inability to complete required interviews, rather than underlying ineligibility. The judge ruled these system failures led to a violation of procedural due process. We propose a performance evaluation framework based on queueing models from operations research and management to assess and improve access in such systems. Operational access failures of call centers are distinct from prior automation failures in benefits provision. Emergent arbitrariness arises from interactions between system dynamics and access demand, rather than from an explicit algorithmic rule, making diagnosis and repair inherently system-level. We develop a queueing model that incorporates phenomena that distinguish social services from standard service domains, redials and abandonment, through which backlogs generate endogenous congestion. Standard queueing guidance from Erlang-A that does not address endogenous congestion fundamentally understaffs, which could lead to persistent shortfalls in practice. Using a fluid approximation, we derive steady-state performance metrics to analytically characterize the impacts of bundled staffing and service delivery changes. We fit model parameters to call-center data disclosed in court documents. Our queueing model can support ex-ante evaluation and design of access systems, inform policy levers for improving access, and provide evidence about whether applicants are afforded a meaningful opportunity to be served at scale. |
| 2026-05-14 | [Position: Behavioural Assurance Cannot Verify the Safety Claims Governance Now Demands](http://arxiv.org/abs/2605.15164v1) | Pratinav Seth, Vinay Kumar Sankarapu | This position paper argues that behavioural assurance, even when carefully designed, is being asked to carry safety claims it cannot verify. AI governance frameworks enacted between 2019 and early 2026 require reviewable evidence of properties such as the absence of hidden objectives, resistance to loss-of-control precursors, and bounded catastrophic capability; current assurance methodologies (primarily behavioural evaluations and red-teaming) are epistemically limited to observable model outputs and cannot verify the latent representations or long-horizon agentic behaviours these frameworks presume to regulate. We formalize this structural mismatch as the audit gap, the divergence between required and achievable verification access, and introduce the concept of fragile assurance to describe cases where the evidential structure does not support the asserted safety claim. Through an analysis of a 21-instrument inventory, we identify an incentive gradient where geopolitical and industrial pressures systematically reward surface-level behavioral proxies over deep structural verification. Finally, we propose a technical pivot: bounding the weight of behavioral evidence in legal text and extending voluntary pre-deployment access with mechanistic-evidence classes, specifically linear probes, activation patching, and before/after-training comparisons. |
| 2026-05-14 | [Complete Local Reasoning About Parameterized Programs Over Topologies](http://arxiv.org/abs/2605.15143v1) | Ruotong Cheng, Azadeh Farzan | This paper investigates the algorithmic safety verification problem of infinite-state parameterized concurrent programs over a rich set of communication topologies. The goal is to automatically produce a proof of correctness in the form of a universally quantified inductive invariant, where the quantification is over the nodes in the topology. We illustrate that under reasonable assumptions on the underlying topology, the problem can be reduced to and solved as a compositional scheme, that is, the verification of the parameterized family is reduced to a set of local proofs, in a complete manner. We propose a verification algorithm, which is implemented as a tool, and demonstrate through a set of benchmarks over several different topologies that our approach is effective in proving parameterized programs safe. |
| 2026-05-14 | [Training ML Models with Predictable Failures](http://arxiv.org/abs/2605.15134v1) | Will Schwarzer, Scott Niekum | Estimating how often an ML model will fail at deployment scale is central to pre-deployment safety assessment, but a feasible evaluation set is rarely large enough to observe the failures that matter. Jones et al. (2025) address this by extrapolating from the largest k failure scores in an evaluation set to predict deployment-scale failure rates. We give a finite-k decomposition of this estimator's forecast error and show that it has a built-in bias toward over-prediction in the typical case, which is the safety-favorable direction. This bias is offset when the evaluation set misses a rare high-failure mode that the deployment set contains, leaving the forecast to under-predict at deployment scale. We propose a fine-tuning objective, the forecastability loss, that addresses this failure mode. In two proof-of-concept experiments, a language-model password game and an RL gridworld, fine-tuning substantially reduces held-out forecast error while preserving primary-task capability and achieving safety similar to that of supervised baselines. |
| 2026-05-14 | [CLOVER: Closed-Loop Value Estimation \& Ranking for End-to-End Autonomous Driving Planning](http://arxiv.org/abs/2605.15120v1) | Sining Ang, Yuguang Yang et al. | End-to-end autonomous driving planners are commonly trained by imitating a single logged trajectory, yet evaluated by rule-based planning metrics that measure safety, feasibility, progress, and comfort. This creates a training--evaluation mismatch: trajectories close to the logged path may violate planning rules, while alternatives farther from the demonstration can remain valid and high-scoring. The mismatch is especially limiting for proposal-selection planners, whose performance depends on candidate-set coverage and scorer ranking quality. We propose CLOVER, a Closed-LOop Value Estimation and Ranking framework for end-to-end autonomous driving planning. CLOVER follows a lightweight generator--scorer formulation: a generator produces diverse candidate trajectories, and a scorer predicts planning-metric sub-scores to rank them at inference time. To expand proposal support beyond single-trajectory imitation, CLOVER constructs evaluator-filtered pseudo-expert trajectories and trains the generator with set-level coverage supervision. It then performs conservative closed-loop self-distillation: the scorer is fitted to true evaluator sub-scores on generated proposals, while the generator is refined toward teacher-selected top-$k$ and vector-Pareto targets with stability regularization. We analyze when an imperfect scorer can improve the generator, showing that scorer-mediated refinement is reliable when scorer-selected targets are enriched under the true evaluator and updates remain conservative. On NAVSIM, CLOVER achieves 94.5 PDMS and 90.4 EPDMS, establishing a new state of the art. On the more challenging NavHard split, it obtains 48.3 EPDMS, matching the strongest reported result. On supplementary nuScenes open-loop evaluation, CLOVER achieves the lowest L2 error and collision rate among compared methods. Code data will be released at https://github.com/WilliamXuanYu/CLOVER. |
| 2026-05-14 | [Talk is (Not) Cheap: A Taxonomy and Benchmark Coverage Audit for LLM Attacks](http://arxiv.org/abs/2605.15118v1) | Karthik Raghu Iyer, Yazdan Jamshidi et al. | We introduce a reusable framework for auditing whether LLM attack benchmarks collectively cover the threat surface: a 4$\times$6 Target $\times$ Technique matrix grounded in STRIDE, constructed from a 507-leaf taxonomy -- 401 data-populated and 106 threat-model-derived leaves -- of inference-time attacks extracted from 932 arXiv security studies (2023--2026). The matrix enables benchmark-external validation -- auditing collective coverage rather than individual benchmark consistency. Applying it to six public benchmarks reveals that the three primary frameworks (HarmBench, InjecAgent, AgentDojo) occupy non-overlapping cells covering at most 25\% of the matrix, while entire STRIDE threat categories (Service Disruption, Model Internals) lack any standardized evaluation, despite published attacks in these categories achieving 46$\times$ token amplification and 96\% attack success rates through mechanisms which no benchmark tests. The corpus of 2,521 unique attack groups further reveals pervasive naming fragmentation (up to 29 surface forms for a single attack) and heavy concentration in Safety \& Alignment Bypass, structural properties invisible at smaller scale. The taxonomy, attack records, and coverage mappings are released as extensible artifacts; as new benchmarks emerge, they can be mapped onto the same matrix, enabling the community to track whether evaluation gaps are closing. |
| 2026-05-14 | [Novel Dynamic Batch-Sensitive Adam Optimiser for Vehicular Accident Injury Severity Prediction](http://arxiv.org/abs/2605.15083v1) | Daniel Asare Kyei, Alimatu Saadia-Yussiff et al. | The choice of optimiser is important in deep learning, as it strongly influences model efficiency and speed of convergence. However, many commonly used optimisers encounter difficulties when applied to imbalanced and sequential datasets, limiting their ability to capture patterns of minority classes. In this study, we propose Dynamic Batch-Sensitive Adam (DBS-Adam), an optimiser that dynamically scales the learning rate using a batch difficulty score derived from exponential moving averages of gradient norms and batch loss. DBS-Adam improves training stability and accelerates convergence by increasing updates for difficult batches and reducing them for easier ones. We evaluate DBS-Adam by integrating it with Bi-Directional LSTM networks for accident injury severity prediction, addressing class imbalance through SMOTE-ENN resampling and Focal Loss. Four experimental configurations compare baseline Bi-LSTM models and alternative architectures to assess optimiser impact. Rigorous comparison against state-of-the-art optimisers (AMSGrad, AdamW, AdaBound) across five random seeds demonstrated DBS-Adam's competitive performance with statistically significant precision improvements (p=0.020). Results indicate that DBS-Adam outperforms standard optimisation approaches, achieving 95.22% test accuracy, 96.11% precision, 95.28% recall, 95.39% F1-score, and a test loss of 0.0086. The proposed framework enables effective real-time accident severity classification for targeted emergency response and road safety interventions, demonstrating the value of DBS-Adam for learning from imbalanced sequential data. |
| 2026-05-14 | [Analyzing Codes of Conduct for Online Safety in Video Games at Scale](http://arxiv.org/abs/2605.15047v1) | Jiuming Jiang, Shidong Pan et al. | Online video games have become major online social spaces where users interact, compete, and create together. These spaces, however, expose users to a wide spectrum of online harms, including harassment, discrimination, inappropriate content, privacy breach, cheating, and more. The shape and severity of such harms vary across game design, mechanics, and community context. To mitigate these harms, game companies issue Codes of Conduct (CoCs) that articulate online safety rules and direct players to safety resources. However, it remains unclear how prevalent CoCs are, what safety, security and privacy violations they govern, and whether they meet growing regulatory and industry expectations. We develop and leverage CONDUCTIFY, a pipeline for identifying and analyzing CoCs at scale. Applied to Steam, the largest PC game marketplace, it located the available CoCs for 350 of the 9,586 multiplayer titles on Steam. We found that CoCs are more available among popular, adult-oriented, and community-driven games, while most multiplayer games operate without CoCs despite regulatory and industry recommendations. Although over 80% of the games with CoCs available consistently address traditional security and safety violations, their governance approaches vary substantially across types of violations. A further asymmetry emerges in specificity. Compared with harms related to gameplay mechanics, the articulations of interpersonal harm and the underage player safety are often less specific, despite their relevance to many game communities. Together, these results inform the improvement of online safety governance and CoC enforcement practices, and building better safety infrastructure for the community of players and developers. |
| 2026-05-14 | [Refactoring-as-Propositions: Proved Refactoring of Hybrid Systems via Proved Refinements](http://arxiv.org/abs/2605.15001v1) | Enguerrand Prebet, André Platzer | Cyber-physical systems are inherently complex due to their connection between software and the physical world. Iterative design reduces their complexity, but increases the need to repeatedly recheck their safety in full after every change. We introduce the refactoring-as-propositions principle in which refactorings are represented as propositions along with a method for proving that system refactorings preserve their required properties by transferring the proof along the respective modification. It is based on differential refinement logic (dRL), with which one can simultaneously and rigorously refer to properties of the systems and the relation between a refactored system and its original version. Refinements represent a uniform way of expressing different types of hybrid system refactorings, including those that introduce auxiliary variables. Furthermore, we show how these refactorings can be proved automatically, and/or reduce to a modular proof solely about the local change rather than about the whole system. |
| 2026-05-14 | [Quantifying and Mitigating Premature Closure in Frontier LLMs](http://arxiv.org/abs/2605.15000v1) | Rebecca Handler, Suhana Bedi et al. | Premature closure, or committing to a conclusion before sufficient information is available, is a recognized contributor to diagnostic error but remains underexamined in large language models (LLMs). We define LLM premature closure as inappropriate commitment under uncertainty: providing an answer, recommendation, or clinical guidance when the safer response would be clarification, abstention, escalation, or refusal. We evaluated five frontier LLMs across structured and open-ended medical tasks. In MedQA (n = 500) and AfriMed-QA (n = 490) questions where the correct choice had been removed, models still selected an answer at high rates, with baseline false-action rates of 55-81% and 53-82%, respectively. In open-ended evaluation, models gave inappropriate answers on an average of 30% of 861 HealthBench questions and 78% of 191 physician-authored adversarial queries. Safety-oriented prompting reduced premature closure across models, but residual failure persisted, highlighting the need to evaluate whether medical LLMs know when not to answer. |
| 2026-05-14 | [Viverra: Text-to-Code with Guarantees](http://arxiv.org/abs/2605.14972v1) | Haoze Wu, Rocky Klopfenstein et al. | A fundamental limitation of Text-to-Code is that no guarantee can be obtained about the correctness of the generated code. Therefore, to ensure its correctness, the generated code still has to be reviewed, tested, and maintained by developers. However, parsing through LLM-generated code can be tedious and time-consuming, potentially negating the productivity gains promised by AI-coding tools. To address this challenge, we present Viverra, a system that automatically produces formally verified annotations alongside generated code to aid user's understanding of the generated program. Given a natural-language task description, Viverra prompts an LLM to synthesize a C program together with candidate assertions expressing safety and correctness properties. It then verifies those assertions in a compositional and best-effort manner via a portfolio of bounded model checkers. Evaluation on 18 diverse programming tasks suggests that Viverra can efficiently generate code with verified assertions, and that these assertions improve users' performance on code-comprehension tasks in a user study with more than 400 participants. |
| 2026-05-14 | [From Particles to Policy: Technical Building Blocks for Multi-State SAI Coordination](http://arxiv.org/abs/2605.14947v1) | R. Yahav, A. Spector et al. | Stratospheric aerosol injection (SAI) is a solar radiation modification technique, proposed as an interim measure to offset warming while greenhouse gas (GHG) emissions are reduced. This paper discusses a possible SAI implementation route - an alternative to sulfate aerosols formed in situ - based on engineered solid particles having dedicated properties such as size, composition, surface chemistry, and traceable origin, supporting safety, controllability, and functionality needed for SAI systems. These engineered properties also open up options for any future multi-state coordination of SAI through two technical building blocks: (1) the SAI-induced radiative forcing (SRF) - the magnitude of the cooling effect attributable specifically to the SAI layer - as an operator-independent quantity, derivable from direct aerosol-layer measurements; and (2) particle traceability through identifying signatures embedded at production. Both could feed into a shared, publicly accessible monitoring database open to independent interrogation, addressing several governance challenges by anchoring compliance assessments in measurable parameters. Drawing on precedents from the Montreal Protocol, IAEA safeguards, and other regimes, we show that shared technical metrics have historically enabled multi-state cooperation, and we argue the same could apply to SAI. We describe a phased pathway in which the technical capabilities and coordination practices that would use them are developed and tested together, at scales orders of magnitude below operational deployment. To be clear - we regard SAI deployment as premature; the conditions under which it might be considered have not been met. The paper does not propose a governance framework; rather, it identifies technical infrastructure that could support a wide range of such frameworks. |
| 2026-05-14 | [Behavioral Data-Driven Optimal Trajectory Generation for Rotary Cranes](http://arxiv.org/abs/2605.14944v1) | Iskandar Khemakhem, Manuel Zobel et al. | With the growth of the construction industry and the global shortage of skilled labor, the automation of crane control has become increasingly important for safe and efficient operations. A central challenge in automatic crane control is the reduction of load oscillations during motion, which is primarily addressed through appropriate slewing trajectories. In this context, classical model-based control methods rely on accurate dynamical models and expert tuning, and often struggle to meet safety and precision requirements, while many learning-based approaches require large data sets and significant computational resources. This paper proposes a behavioral data-driven framework for generating open-loop slewing trajectories for rotary cranes that suppress load sway while reducing operation time and energy consumption. The approach builds on Willems' fundamental lemma and its generalizations, to bypass explicit system modeling and operate directly on measured input-output data. A practical workflow is presented in this paper to reduce the need for expert knowledge. Despite the underactuated nature of the crane dynamics, the method identifies a nonparametric representation of the system behavior and generates smooth, optimal trajectories using limited data and convex optimization. The proposed trajectory generation method is validated on a laboratory crane setup and compared against an established model-based approach, achieving up to 35% reduction in load sway, 43% reduction in tracking error, and 50% reduction in travel time. |
| 2026-05-14 | [Radioactive Source Seeking using Bayesian Optimisation with Movement Penalty](http://arxiv.org/abs/2605.14942v1) | Lysander Miller, Joshua Keene et al. | The use of mobile robotics in radioactive source seeking has become an important part of modern radiation-safety practices, supporting timely mitigation of contamination risks and helping protect public health. However, measuring radiation is often time-consuming, rendering traditional gradient-based source-seeking methods less effective due to lower sample efficiency. This paper proposes a sample-efficient Bayesian-Optimisation source-seeking strategy that utilises a heteroscedastic Gaussian process surrogate to balance exploration and exploitation. Excessive inter-sample travel is discouraged through a movement switching cost. The strategy is shown to generate sublinear regret in the source-seeking task, while simulations demonstrate its effectiveness in localising radioactive sources. |
| 2026-05-14 | [EVA: Editing for Versatile Alignment against Jailbreaks](http://arxiv.org/abs/2605.14750v1) | Yi Wang, Hongye Qiu et al. | Large Language Models (LLMs) and Vision Language Models (VLMs) have demonstrated impressive capabilities but remain vulnerable to jailbreaking attacks, where adversaries exploit textual or visual triggers to bypass safety guardrails. Recent defenses typically rely on safety fine-tuning or external filters to reduce the model's likelihood of producing harmful content. While effective to some extent, these methods often incur significant computational overheads and suffer from the safety utility trade-off, degrading the model's performance on benign tasks. To address these challenges, we propose EVA (Editing for Versatile Alignment against Jailbreaks), a novel framework that pioneers the application of direct model editing for safety alignment. EVA reframes safety alignment as a precise knowledge correction task. Instead of retraining massive parameters, EVA identifies and surgically edits specific neurons responsible for the model's susceptibility to harmful instructions, while leaving the vast majority of the model unchanged. By localizing the updates, EVA effectively neutralizes harmful behaviors without compromising the model's general reasoning capabilities. Extensive experiments demonstrate that EVA outperforms baselines in mitigating jailbreaks across both LLMs and VLMs, offering a precise and efficient solution for post-deployment safety alignment. |
| 2026-05-14 | [Selective Safety Steering via Value-Filtered Decoding](http://arxiv.org/abs/2605.14746v1) | Bat-Sheva Einbinder, Hen Davidov et al. | While large language models (LLMs) are trained to align with human values, their generations may still violate safety constraints. A growing line of work addresses this problem by modifying the model's sampling policy at decoding time using a safety reward. However, existing decoding-time steering methods often intervene unnecessarily, modifying generations that would have been safe under the base model. Such unnecessary interventions are undesirable, as they can distort key properties of the base model such as helpfulness, fluency, style, and coherence. We propose a new test-time steering method designed to reduce such unnecessary interventions while improving the safety of unsafe responses. Our approach filters tokens using a value-based safety criterion and provides an explicit bound on the probability of false interventions. A single threshold hyperparameter controls this bound, allowing practitioners to trade off higher rates of unnecessary intervention for better output safety. Across multiple datasets and experiments, we show that our value-filtered decoding method outperforms existing baselines, achieving better trade-offs between safety, helpfulness, and similarity to the base model. |
| 2026-05-14 | [Wide parameter-space O3 search for continuous gravitational waves from unknown neutron stars in binary systems](http://arxiv.org/abs/2605.14728v1) | P. B. Covas, M. A. Papa et al. | Continuous gravitational waves, i.e., persistent and nearly-monochromatic signals emitted by asymmetric spinning neutron stars, remain elusive. Searches for these signals from unknown binary systems are the most computationally challenging, but they are essential, given that binary accretion provides a natural mechanism for creating the required asymmetry, and around half of the known pulsars rotating above 25 Hz are part of a binary system. Here we report on a search of a large uncharted parameter-space region: for the first time we cover gravitational-wave frequencies above 520 Hz (from 50 to 1000 Hz), and, for the first time with advanced detectors, orbital periods lower than 3 days are explored. No signal is detected, and we set the most stringent constraints to date on the amplitude of signals of this kind. Our results exclude with $95\%$ confidence neutron stars within 100 pc and rotating faster than $\sim$ 495 Hz from having ellipticities above $5.2 \times 10^{-8}$. Within the same distance our results also exclude r-mode amplitudes above $1.5 \times 10^{-6}$ for stars rotating faster than $\sim$ 740 Hz. |
| 2026-05-14 | [Agentifying Patient Dynamics within LLMs through Interacting with Clinical World Model](http://arxiv.org/abs/2605.14723v1) | Minghao Wu, Yuting Yan et al. | Sepsis management in the ICU requires sequential treatment decisions under rapidly evolving patient physiology. Although large language models (LLMs) encode broad clinical knowledge and can reason over guidelines, they are not inherently grounded in action-conditioned patient dynamics. We introduce SepsisAgent, a world model-augmented LLM agent for sepsis treatment recommendation. SepsisAgent uses a learned Clinical World Model to simulate patient responses under candidate fluid--vasopressor interventions, and follows a propose--simulate--refine workflow before committing to a prescription. We first show that world-model access alone yields inconsistent LLM decision performance, motivating agent-specific training. We then train SepsisAgent through a three-stage curriculum: patient-dynamics supervised fine-tuning, propose--simulate--refine behavior cloning, and world-model-based agentic reinforcement learning. On MIMIC-IV sepsis trajectories, SepsisAgent outperforms all traditional RL and LLM-based baselines in off-policy value while achieving the best safety profile under guideline adherence and unsafe-action metrics. Further analysis shows that repeated interaction with the Clinical World Model enables the agent to learn regularities in patient evolution, which remain useful even when simulator access is removed. |
| 2026-05-14 | [Agentic AI in Industry: Adoption Level and Deployment Barriers](http://arxiv.org/abs/2605.14675v1) | Spyridon Alvanakis Apostolou, Jan Bosch et al. | Agentic AI systems are entering software engineering workflows, yet empirical evidence on how industrial organizations actually adopt them remains sparse. We present a qualitative interview study with sixteen practitioners across twelve companies of varying size and domain. This study characterizes the current agentic AI adoption state of these companies, employing a six-level maturity framework adapted from established AI-driven organizations. The findings reveal that seven companies operate at Level~1 (AI Assistants), four companies at Level~2 (AI Compensators), and only one in Level~3 (Multi-Agent Orchestration), with large and safety-regulated organizations among the most advanced adopters. The primary finding is a capability-deployment verification gap, four companies demonstrated higher-level experimental AI capabilities but cannot integrate them into production workflows because adequate output verification mechanisms are absent, leaving human-in-the-loop as the only trusted verification mechanism. This gap is shaped by four recurring barriers: context window of LLMs constraints especially when diverse knowledge aggregation is needed, under-performance on proprietary programming languages and protocols, non-determinism incompatible with qualification standards, and data confidentiality concerns. Two interdependent dimensions of this gap emerge from these findings (information asymmetry and qualification absence) framing a core open problem for industrial agentic integration. |

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



