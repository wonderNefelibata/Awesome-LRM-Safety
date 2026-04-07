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
| 2026-04-06 | [HorizonWeaver: Generalizable Multi-Level Semantic Editing for Driving Scenes](http://arxiv.org/abs/2604.04887v1) | Mauricio Soroco, Francesco Pittaluga et al. | Ensuring safety in autonomous driving requires scalable generation of realistic, controllable driving scenes beyond what real-world testing provides. Yet existing instruction guided image editors, trained on object-centric or artistic data, struggle with dense, safety-critical driving layouts. We propose HorizonWeaver, which tackles three fundamental challenges in driving scene editing: (1) multi-level granularity, requiring coherent object- and scene-level edits in dense environments; (2) rich high-level semantics, preserving diverse objects while following detailed instructions; and (3) ubiquitous domain shifts, handling changes in climate, layout, and traffic across unseen environments. The core of HorizonWeaver is a set of complementary contributions across data, model, and training: (1) Data: Large-scale dataset generation, where we build a paired real/synthetic dataset from Boreas, nuScenes, and Argoverse2 to improve generalization; (2) Model: Language-Guided Masks for fine-grained editing, where semantics-enriched masks and prompts enable precise, language-guided edits; and (3) Training: Content preservation and instruction alignment, where joint losses enforce scene consistency and instruction fidelity. Together, HorizonWeaver provides a scalable framework for photorealistic, instruction-driven editing of complex driving scenes, collecting 255K images across 13 editing categories and outperforming prior methods in L1, CLIP, and DINO metrics, achieving +46.4% user preference and improving BEV segmentation IoU by +33%. Project page: https://msoroco.github.io/horizonweaver/ |
| 2026-04-06 | [Learning, Potential, and Retention: An Approach for Evaluating Adaptive AI-Enabled Medical Devices](http://arxiv.org/abs/2604.04878v1) | Alexis Burgon, Berkman Sahiner et al. | This work addresses challenges in evaluating adaptive artificial intelligence (AI) models for medical devices, where iterative updates to both models and evaluation datasets complicate performance assessment. We introduce a novel approach with three complementary measurements: learning (model improvement on current data), potential (dataset-driven performance shifts), and retention (knowledge preservation across modification steps), to disentangle performance changes caused by model adaptations versus dynamic environments. Case studies using simulated population shifts demonstrate the approach's utility: gradual transitions enable stable learning and retention, while rapid shifts reveal trade-offs between plasticity and stability. These measurements provide practical insights for regulatory science, enabling rigorous assessment of the safety and effectiveness of adaptive AI systems over sequential modifications. |
| 2026-04-06 | [Incompleteness of AI Safety Verification via Kolmogorov Complexity](http://arxiv.org/abs/2604.04876v1) | Munawar Hasan | Ensuring that artificial intelligence (AI) systems satisfy formal safety and policy constraints is a central challenge in safety-critical domains. While limitations of verification are often attributed to combinatorial complexity and model expressiveness, we show that they arise from intrinsic information-theoretic limits. We formalize policy compliance as a verification problem over encoded system behaviors and analyze it using Kolmogorov complexity. We prove an incompleteness result: for any fixed sound computably enumerable verifier, there exists a threshold beyond which true policy-compliant instances cannot be certified once their complexity exceeds that threshold. Consequently, no finite formal verifier can certify all policy-compliant instances of arbitrarily high complexity. This reveals a fundamental limitation of AI safety verification independent of computational resources, and motivates proof-carrying approaches that provide instance-level correctness guarantees. |
| 2026-04-06 | [Assessing Large Language Models for Stabilizing Numerical Expression in Scientific Software](http://arxiv.org/abs/2604.04854v1) | Tien Nguyen, Muhammad Ali Gulzar et al. | Scientific software relies on high-precision computation, yet finite floating-point representations can introduce precision errors that propagate in safety-critical domains. Despite the growing use of large language models (LLMs) in scientific applications, their reliability in handling floating-point numerical stability has not been systematically evaluated. This paper evaluates LLMs' reasoning on high-precision numerical computation through two numerical stabilization tasks: (1) detecting instability in numerical expressions by generating error-inducing inputs (detection), and (2) rewriting expressions to improve numerical stability (stabilization). Using popular numerical benchmarks, we assess six LLMs on nearly 2,470 numerical structures, including nested conditionals, high-precision literals, and multi-variable arithmetic.   Our results show that LLMs are equally effective as state-of-the-art traditional approaches in detecting and stabilizing numerically unstable computations. More notably, LLMs outperform baseline methods precisely where the latter fail: in 17.4% (431) of expressions where the baseline does not improve accuracy, LLMs successfully stabilize 422 (97.9%) of them, and achieve greater stability than the baseline across 65.4% (1,615) of all expressions. However, LLMs struggle with control flow and high-precision literals, consistently removing such structures rather than reasoning about their numerical implications, whereas they perform substantially better on purely symbolic expressions. Together, these findings suggest that LLMs are effective at stabilizing expressions that classical techniques cannot, yet struggle when exact numerical magnitudes and control flow semantics must be precisely reasoned about, as such concrete patterns are rarely encountered during training. |
| 2026-04-06 | [Latent Profiles of AI Risk Perception and Their Differential Association with Community Driving Safety Concerns: A Person-Centered Analysis](http://arxiv.org/abs/2604.04849v1) | Amir Rafe, Anika Baitullah et al. | Public attitudes toward artificial intelligence (AI) and driving safety are typically studied in isolation using variable-centered methods that assume population homogeneity, yet risk perception theory predicts that these evaluations covary within individuals as expressions of underlying worldviews. This study identifies latent profiles of AI risk perception among U.S. adults and tests whether these profiles are differentially associated with community driving safety concerns. Latent class analysis was applied to nine AI risk-perception indicators from a nationally representative survey (Pew Research Center American Trends Panel Wave 152, n = 5,255); Bolck-Croon-Hagenaars corrected distal outcome analysis tested class differences on nine driving-safety outcomes, and survey-weighted multinomial logistic regression identified demographic and ideological predictors of class membership. Four classes emerged: Moderate Skeptics (17.5%), Concerned Pragmatists (42.8%), AI Ambivalent (10.6%), and Extreme Alarm (29.1%), with all nine driving-safety outcomes significantly differentiated across classes. Higher AI concern mapped monotonically onto greater perceived driving-hazard severity; the exception, comparative evaluation of AI versus human driving, was driven by trust rather than concern level. The cross-domain covariation provides person-level evidence for the worldview-based risk structuring posited by Cultural Theory of Risk and yields a four-class segmentation framework for AV communication that links AI risk orientation to transportation safety attitudes. |
| 2026-04-06 | [Do No Harm: Exposing Hidden Vulnerabilities of LLMs via Persona-based Client Simulation Attack in Psychological Counseling](http://arxiv.org/abs/2604.04842v1) | Qingyang Xu, Yaling Shen et al. | The increasing use of large language models (LLMs) in mental healthcare raises safety concerns in high-stakes therapeutic interactions. A key challenge is distinguishing therapeutic empathy from maladaptive validation, where supportive responses may inadvertently reinforce harmful beliefs or behaviors in multi-turn conversations. This risk is largely overlooked by existing red-teaming frameworks, which focus mainly on generic harms or optimization-based attacks. To address this gap, we introduce Personality-based Client Simulation Attack (PCSA), the first red-teaming framework that simulates clients in psychological counseling through coherent, persona-driven client dialogues to expose vulnerabilities in psychological safety alignment. Experiments on seven general and mental health-specialized LLMs show that PCSA substantially outperforms four competitive baselines. Perplexity analysis and human inspection further indicate that PCSA generates more natural and realistic dialogues. Our results reveal that current LLMs remain vulnerable to domain-specific adversarial tactics, providing unauthorized medical advice, reinforcing delusions, and implicitly encouraging risky actions. |
| 2026-04-06 | [Bridging Data-Driven Reachability Analysis and Statistical Estimation via Constrained Matrix Convex Generators](http://arxiv.org/abs/2604.04822v1) | Peng Xie, Zhen Zhang et al. | Data-driven reachability analysis enables safety verification when first-principles models are unavailable. This requires constructing sets of system models consistent with measured trajectories and noise assumptions. Existing approaches rely on zonotopic or box-based approximations, which do not fit the geometry of common noise distributions such as Gaussian disturbances and can lead to significant conservatism, especially in high-dimensional settings. This paper builds on ellipsotope-based representations to introduce mixed-norm uncertainty sets for data-driven reachability. The highest-density region defines the exact minimum-volume noise confidence set, while Constrained Convex Generators (CCG) and their matrix counterpart (CMCG) provide compatible geometric representations at the noise and parameter level. We show that the resulting CMCG coincides with the maximum-likelihood confidence ellipsoid for Gaussian disturbances, while remaining strictly tighter than constrained matrix zonotopes for mixed bounded-Gaussian noise. For non-convex noise distributions such as Gaussian mixtures, a minimum-volume enclosing ellipsoid provides a tractable convex surrogate. We further prove containment of the CMCG times CCG product and bound the conservatism of the Gaussian-Gaussian interaction. Numerical examples demonstrate substantially tighter reachable sets compared to box-based approximations of Gaussian disturbances. These results enable less conservative safety verification and improve the accuracy of uncertainty-aware control design. |
| 2026-04-06 | [From Hallucination to Scheming: A Unified Taxonomy and Benchmark Analysis for LLM Deception](http://arxiv.org/abs/2604.04788v1) | Jerick Shi, Terry Jingcheng Zhang et al. | Large language models (LLMs) produce systematically misleading outputs, from hallucinated citations to strategic deception of evaluators, yet these phenomena are studied by separate communities with incompatible terminology. We propose a unified taxonomy organized along three complementary dimensions: degree of goal-directedness (behavioral to strategic deception), object of deception, and mechanism (fabrication, omission, or pragmatic distortion). Applying this taxonomy to 50 existing benchmarks reveals that every benchmark tests fabrication while pragmatic distortion, attribution, and capability self-knowledge remain critically under-covered, and strategic deception benchmarks are nascent. We offer concrete recommendations for developers and regulators, including a minimal reporting template for positioning future work within our framework. |
| 2026-04-06 | [Cheap Talk, Empty Promise: Frontier LLMs easily break public promises for self-interest](http://arxiv.org/abs/2604.04782v1) | Jerick Shi, Terry Jingcheng Zhang et al. | Large language models are increasingly deployed as autonomous agents in multi-agent settings where they communicate intentions and take consequential actions with limited human oversight. A critical safety question is whether agents that publicly commit to actions break those promises when they can privately deviate, and what the consequences are for both themselves and the collective. We study deception as a deviation from a publicly announced action in one-shot normal-form games, classifying each deviation by its effect on individual payoff and collective welfare into four categories: win-win, selfish, altruistic, and sabotaging. By exhaustively enumerating announcement profiles across six canonical games, nine frontier models, and varying group sizes, we identify all opportunities for each deviation type and measure how often agents exploit them. Across all settings, agents deviate from promises in approximately 56.6% of scenarios, but the character of deception varies substantially across models even at similar overall rates. Most critically, for the majority of the models, promise-breaking occurs without verbalized awareness of the fact that they are breaking promises. |
| 2026-04-06 | [Community Driving-Safety Deterioration as a Push Factor for Public Endorsement of AI Driving Capability](http://arxiv.org/abs/2604.04775v1) | Amir Rafe, Subasish Das | Road traffic crashes claim approximately 1.19 million lives annually worldwide, and human error accounts for the vast majority, yet the autonomous vehicle acceptance literature models adoption almost exclusively through technology-centered pull factors such as perceived usefulness and trust. This study examines a moderated mediation model in which perceived community driving-safety concern (PCSC) predicts evaluations of AI versus human driving capability, mediated by Generalized AI Orientation and moderated by personal driving frequency. Weighted structural equation modeling is applied to a nationally representative U.S. probability sample from Pew Research Center's American Trends Panel Wave 152, using Weighted Least Squares Mean and Variance Adjusted (WLSMV)-estimated confirmatory factor analysis on ordinal indicators, bias-corrected bootstrap inference, and seven robustness checks including Imai sensitivity analysis, E-value confounding thresholds, and propensity score matching. Results reveal a dual-pathway mechanism constituting an inconsistent mediation: PCSC exerts a small positive direct effect on AI driving evaluation, consistent with a domain-specific push interpretation, while simultaneously suppressing Generalized AI Orientation, which is itself a strong positive predictor of AI driving evaluation. Conditional indirect effects are negative and statistically significant at low, mean, and high levels of driving frequency. These findings establish a risk-spillover mechanism whereby community driving-safety concern promotes domain-specific AI endorsement yet suppresses domain-general AI enthusiasm, yielding a near-zero net total effect. |
| 2026-04-06 | [Collaborative Altruistic Safety in Coupled Multi-Agent Systems](http://arxiv.org/abs/2604.04772v1) | Brooks A. Butler, Xiao Tan et al. | This paper presents a novel framework for ensuring safety in dynamically coupled multi-agent systems through collaborative control. Drawing inspiration from ecological models of altruism, we develop collaborative control barrier functions that allow agents to cooperatively enforce individual safety constraints under coupling dynamics. We introduce an altruistic safety condition based on the so-called Hamilton's rule, enabling agents to trade off their own safety to support higher-priority neighbors. By incorporating these conditions into a distributed optimization framework, we demonstrate increased feasibility and robustness in maintaining system-wide safety. The effectiveness of the proposed approach is illustrated through simulation in a simplified formation control scenario. |
| 2026-04-06 | [Your Agent, Their Asset: A Real-World Safety Analysis of OpenClaw](http://arxiv.org/abs/2604.04759v1) | Zijun Wang, Haoqin Tu et al. | OpenClaw, the most widely deployed personal AI agent in early 2026, operates with full local system access and integrates with sensitive services such as Gmail, Stripe, and the filesystem. While these broad privileges enable high levels of automation and powerful personalization, they also expose a substantial attack surface that existing sandboxed evaluations fail to capture. To address this gap, we present the first real-world safety evaluation of OpenClaw and introduce the CIK taxonomy, which unifies an agent's persistent state into three dimensions, i.e., Capability, Identity, and Knowledge, for safety analysis. Our evaluations cover 12 attack scenarios on a live OpenClaw instance across four backbone models (Claude Sonnet 4.5, Opus 4.6, Gemini 3.1 Pro, and GPT-5.4). The results show that poisoning any single CIK dimension increases the average attack success rate from 24.6% to 64-74%, with even the most robust model exhibiting more than a threefold increase over its baseline vulnerability. We further assess three CIK-aligned defense strategies alongside a file-protection mechanism; however, the strongest defense still yields a 63.8% success rate under Capability-targeted attacks, while file protection blocks 97% of malicious injections but also prevents legitimate updates. Taken together, these findings show that the vulnerabilities are inherent to the agent architecture, necessitating more systematic safeguards to secure personal AI agents. Our project page is https://ucsc-vlaa.github.io/CIK-Bench. |
| 2026-04-06 | [Data-Driven Reachability Analysis with Optimal Input Design](http://arxiv.org/abs/2604.04758v1) | Peng Xie, Davide M. Raimondo et al. | This paper addresses the conservatism in data-driven reachability analysis for discrete-time linear systems subject to bounded process noise, where the system matrices are unknown and only input--state trajectory data are available. Building on the constrained matrix zonotope (CMZ) framework, two complementary strategies are proposed to reduce conservatism in reachable-set over-approximations. First, the standard Moore--Penrose pseudoinverse is replaced with a row-norm-minimizing right inverse computed via a second-order cone program (SOCP), which directly reduces the size of the resulting model set, yielding tighter generators and less conservative reachable sets. Second, an online A-optimal input design strategy is introduced to improve the informativeness of the collected data and the conditioning of the resulting model set, thereby reducing uncertainty. The proposed framework extends naturally to piecewise affine systems through mode-dependent data partitioning. Numerical results on a five-dimensional stable LTI system and a two-dimensional piecewise affine system demonstrate that combining designed inputs with the row-norm right inverse significantly reduces conservatism compared to a baseline using random inputs and the pseudoinverse, leading to tighter reachable sets for safety verification. |
| 2026-04-06 | [Fine-Tuning Integrity for Modern Neural Networks: Structured Drift Proofs via Norm, Rank, and Sparsity Certificates](http://arxiv.org/abs/2604.04738v1) | Zhenhang Shang, Kani Chen | Fine-tuning is now the primary method for adapting large neural networks, but it also introduces new integrity risks. An untrusted party can insert backdoors, change safety behavior, or overwrite large parts of a model while claiming only small updates. Existing verification tools focus on inference correctness or full-model provenance and do not address this problem.   We introduce Fine-Tuning Integrity (FTI) as a security goal for controlled model evolution. An FTI system certifies that a fine-tuned model differs from a trusted base only within a policy-defined drift class. We propose Succinct Model Difference Proofs (SMDPs) as a new cryptographic primitive for enforcing these drift constraints. SMDPs provide zero-knowledge proofs that the update to a model is norm-bounded, low-rank, or sparse. The verifier cost depends only on the structure of the drift, not on the size of the model.   We give concrete SMDP constructions based on random projections, polynomial commitments, and streaming linear checks. We also prove an information-theoretic lower bound showing that some form of structure is necessary for succinct proofs. Finally, we present architecture-aware instantiations for transformers, CNNs, and MLPs, together with an end-to-end system that aggregates block-level proofs into a global certificate. |
| 2026-04-06 | [Lighting Up or Dimming Down? Exploring Dark Patterns of LLMs in Co-Creativity](http://arxiv.org/abs/2604.04735v1) | Zhu Li, Jiaming Qu et al. | Large language models (LLMs) are increasingly acting as collaborative writing partners, raising questions about their impact on human agency. In this exploratory work, we investigate five "dark patterns" in human-AI co-creativity -- subtle model behaviors that can suppress or distort the creative process: Sycophancy, Tone Policing, Moralizing, Loop of Death, and Anchoring. Through a series of controlled sessions where LLMs are prompted as writing assistants across diverse literary forms and themes, we analyze the prevalence of these behaviors in generated responses. Our preliminary results suggest that Sycophancy is nearly ubiquitous (91.7% of cases), particularly in sensitive topics, while Anchoring appears to be dependent on literary forms, surfacing most frequently in folktales. This study indicates that these dark patterns, often byproducts of safety alignment, may inadvertently narrow creative exploration and proposes design considerations for AI systems that effectively support creative writing. |
| 2026-04-06 | [A Multi-Agent Framework for Democratizing XR Content Creation in K-12 Classrooms](http://arxiv.org/abs/2604.04728v1) | Yuan Chang, Zhu Li et al. | Generative AI (GenAI) combined with Extended Reality (XR) offers potential for K-12 education, yet classroom adoption remains limited by the high technical barrier of XR content authoring. Moreover, the probabilistic nature of GenAI introduces risks of hallucination that may cause severe consequences in K-12 education settings. In this work, we present a multi-agent XR authoring framework. Our prototype system coordinates four specialized agents: a Pedagogical Agent outlining grade-appropriate content specifications with learning objectives; an Execution Agent assembling 3D assets and XR contents; a Safeguard Agent validating generated content against five safety criteria; and a Tutor Agent embedding educational notes and quiz questions within the scene. Our teacher-facing system combines pedagogical intent, safety validation, and educational enrichment. It does not require technical expertise and targets commodity devices. |
| 2026-04-06 | [AI Assistance Reduces Persistence and Hurts Independent Performance](http://arxiv.org/abs/2604.04721v1) | Grace Liu, Brian Christian et al. | People often optimize for long-term goals in collaboration: A mentor or companion doesn't just answer questions, but also scaffolds learning, tracks progress, and prioritizes the other person's growth over immediate results. In contrast, current AI systems are fundamentally short-sighted collaborators - optimized for providing instant and complete responses, without ever saying no (unless for safety reasons). What are the consequences of this dynamic? Here, through a series of randomized controlled trials on human-AI interactions (N = 1,222), we provide causal evidence for two key consequences of AI assistance: reduced persistence and impairment of unassisted performance. Across a variety of tasks, including mathematical reasoning and reading comprehension, we find that although AI assistance improves performance in the short-term, people perform significantly worse without AI and are more likely to give up. Notably, these effects emerge after only brief interactions with AI (approximately 10 minutes). These findings are particularly concerning because persistence is foundational to skill acquisition and is one of the strongest predictors of long-term learning. We posit that persistence is reduced because AI conditions people to expect immediate answers, thereby denying them the experience of working through challenges on their own. These results suggest the need for AI model development to prioritize scaffolding long-term competence alongside immediate task completion. |
| 2026-04-06 | [Bridging Safety and Security in Complex Systems: A Model-Based Approach with SAFT-GT Toolchain](http://arxiv.org/abs/2604.04705v1) | Irdin Pekaric, Raffaela Groner et al. | In the rapidly evolving landscape of software engineering, the demand for robust and secure systems has become increasingly critical. This is especially true for self-adaptive systems due to their complexity and the dynamic environments in which they operate. To address this issue, we designed and developed the SAFT-GT toolchain that tackles the multifaceted challenges associated with ensuring both safety and security. This paper provides a comprehensive description of the toolchain's architecture and functionalities, including the Attack-Fault Trees generation and model combination approaches. We emphasize the toolchain's ability to integrate seamlessly with existing systems, allowing for enhanced safety and security analyses without requiring extensive modifications and domain knowledge. Our proposed approach can address evolving security threats, including both known vulnerabilities and emerging attack vectors that could compromise the system. As a use case for the toolchain, we integrate it into the feedback loop of self-adaptive systems. Finally, to validate the practical applicability of the toolchain, we conducted an extensive user study involving domain experts, whose insights and feedback underscore the toolchain's relevance and usability in real-world scenarios. Our findings demonstrate the toolchain's effectiveness in real-world applications while highlighting areas for future improvements. The toolchain and associated resources are available in an open-source repository to promote reproducibility and encourage further research in this field. |
| 2026-04-06 | [EHT-Constrained Analysis of Shadow Deformation in Quantum-Improved Rotating Non-Singular Magnetic Monopole](http://arxiv.org/abs/2604.04695v1) | Gowtham Sidharth M, Sanjit Das | We studied the shadow cast by a rotating Bardeen black hole within the framework of asymptotically safe gravity. The null geodesics were analyzed using the Hamilton Jacobi separation method to derive shadow observables. Our findings show that an increase in both the asymptotic safety parameter and the spin parameter leads to a decrease in the apparent shadow size and an increase in shadow distortion. The monopole charge of the black hole played an important role in the shadow profile. Furthermore, we compute the energy emission rate associated with varying values of the asymptotic safety parameter. |
| 2026-04-06 | [Springdrift: An Auditable Persistent Runtime for LLM Agents with Case-Based Memory, Normative Safety, and Ambient Self-Perception](http://arxiv.org/abs/2604.04660v1) | Seamus Brady | We present Springdrift, a persistent runtime for long-lived LLM agents. The system integrates an auditable execution substrate (append-only memory, supervised processes, git-backed recovery), a case-based reasoning memory layer with hybrid retrieval (evaluated against a dense cosine baseline), a deterministic normative calculus for safety gating with auditable axiom trails, and continuous ambient self-perception via a structured self-state representation (the sensorium) injected each cycle without tool calls. These properties support behaviours difficult to achieve in session-bounded systems: cross-session task continuity, cross-channel context maintenance, end-to-end forensic reconstruction of decisions, and self-diagnostic behaviour. We report on a single-instance deployment over 23 days (19 operating days), during which the agent diagnosed its own infrastructure bugs, classified failure modes, identified an architectural vulnerability, and maintained context across email and web channels -- without explicit instruction. We introduce the term Artificial Retainer for this category: a non-human system with persistent memory, defined authority, domain-specific autonomy, and forensic accountability in an ongoing relationship with a specific principal -- distinguished from software assistants and autonomous agents, drawing on professional retainer relationships and the bounded autonomy of trained working animals. This is a technical report on a systems design and deployment case study, not a benchmark-driven evaluation. Evidence is from a single instance with a single operator, presented as illustration of what these architectural properties can support in practice. Implemented in approximately Gleam on Erlang/OTP. Code, artefacts, and redacted operational logs will be available at https://github.com/seamus-brady/springdrift upon publication. |

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



