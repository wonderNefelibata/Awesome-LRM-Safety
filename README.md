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
| 2026-09-04 | [Large Language Models for HVAC Operations in Building Energy Systems: A Critical Review of Methods, Applications, and Deployment Readiness](http://arxiv.org/abs/2609.05314v1) | Alexander Neubauer, Tianzhen Hong et al. | Building automation systems generate rich sensor data yet remain insight-poor because heterogeneous point naming, missing metadata, and fragmented documentation obstruct their operational use. This systematic review analyses and codes 66 peer-reviewed studies on large language models (LLMs) for HVAC operations published between 2023 and March 2026. Each study is classified across five application families and three LLM method families and assessed for evidence realism, deployment readiness, and the responsibility boundary between the LLM and physical HVAC decisions. The corpus is concentrated in building energy modelling (BEM, 32 of 66 papers), while load forecasting remains too sparse for subfield-level conclusions. Only four studies reach pilot-level evidence, and none reports sustained operational deployment. No study was classified as ready-now for industry adoption; three were near-term and 63 research-only. Nevertheless, several bounded, human-in-the-loop uses merit near-term trials, including point-name normalisation, document-grounded operator support, BEM workflow assistance, and advisory interfaces around physics-based controllers. Conventional machine learning (ML), model predictive control (MPC), reinforcement learning (RL) and ontology-based tools remain more adopted for high-frequency control, short-horizon numerical forecasting, and well-posed ontology mapping, while autonomous agentic operation and unvalidated occupant proxies remain research-stage. Current evidence therefore supports LLMs primarily as semantic and workflow layers rather than autonomous HVAC controllers. Future work should prioritise field-validated benchmarks, orchestration evaluation under operational constraints, and LLM-MPC/RL architectures with bounded latency and verifiable safety properties. |
| 2026-09-04 | [Human-Human & Human-Robot Interaction Transformer (H2INT) for Robot Navigation in Dense and Uncertain Crowds](http://arxiv.org/abs/2609.05300v1) | Ao Shen, Kaixi Chen et al. | Safe robot navigation in dense crowds requires reasoning about pedestrian motion and how it may change in response to a robot. However, many learning-based approaches generate pedestrian motion independently of the robot or assume uniform reciprocity, omitting an important source of interaction uncertainty. This paper presents a Human-Human & Human-Robot Interaction Transformer (H2INT), a reinforcement learning framework that retains robot-conditioned changes in pedestrian motion during policy learning while allowing responsiveness to vary across pedestrians. Responsiveness affects the crowd dynamics when the robot is visible but is not supplied as a policy input; the policy must instead infer its consequences from robot-centered relative positions. A two-stage gated Transformer progressively encodes human-human and human-robot relations, while a recurrent policy captures their temporal evolution. A curriculum gradually reduces pedestrian responsiveness to increase interaction difficulty. Simulation experiments demonstrate improved navigation safety and robustness over representative baselines across response conditions and crowd densities, and show transfer without retraining to structurally distinct crowd-flow layouts. Ablations support the hierarchical relational encoding and gated updates. Real-robot deployment further verifies that the learned policy can operate with sparse observations in a physical environment. |
| 2026-09-04 | [AI for Computational Design Science: A Responsible Human-AI Framework and Case Study on Short-Form Video Safety Surveillance](http://arxiv.org/abs/2609.05270v1) | Wenli Zhang, Jiaheng Xie et al. | Artificial intelligence (AI) is transforming not only what information systems researchers design, but also how design research is conducted. Yet existing literature offers limited guidance for computational design science (CDS) when AI actively participates in problem formulation, resource construction, design search, evaluation, and knowledge abstraction. We develop AI for Computational Design Science (AI4CDS), a five-phase methodological framework in which AI expands problem and design search while researchers retain responsibility for domain grounding, admissibility, verification, and scientific judgment. Collaboration is governed by graduated trust, reversibility, auditability, and differentiated reproducibility. We instantiate AI4CDS through ChildRiskGuard, an interpretable artifact for detecting short-form videos inappropriate for children, while documenting AI interactions, rejected alternatives, corrections, and audit trails. The case translates audience-dependent safety and explanation faithfulness into three technical challenges and develops an artifact that separates generic from child-specific risk, represents distinct developmental-risk mechanisms, and makes concept-level explanations part of the predictive computation. ChildRiskGuard achieves an F1 score of 0.769, substantially outperforming direct application of a general-purpose content-safety model while remaining competitive with strong benchmarks. The primary contribution is AI4CDS as a responsible framework for AI-enabled CDS; ChildRiskGuard provides process and artifact evidence of how AI-expanded, researcher-governed design can generate and evaluate novel computational design knowledge. |
| 2026-09-04 | [Uncensored Open-weight Models: Redistribution as the Persistence Layer](http://arxiv.org/abs/2609.05241v1) | 10a Labs,  : et al. | A rapidly expanding ecosystem of actors is removing built-in safety guardrails from open-weight AI models. We profile this ecosystem by identifying key producers, downstream reproductions, and emerging applications. Between January 2024 and March 2026, we identified 3,471 original uncensored models on HuggingFace, each repackaged an average of 2.4 times; three actors account for 52% of all 8,164 compressed redistributions. Once quantized and mirrored across separate accounts, formats, and registries such as Ollama, these models persist regardless of upstream removal and become easier to deploy downstream. Of the 1,643 identified GitHub applications integrating uncensored large language models (ULLMs), 25% were classified as explicitly malicious. |
| 2026-09-04 | [Risk-Aware Optimal Control with Rulebooks](http://arxiv.org/abs/2609.05199v1) | Tichakorn Wongpiromsarn | We consider safety-critical control problems involving multiple requirements with different priorities and uncertainty in their evaluation. We represent these requirements using risk-aware rulebooks, where each requirement is assigned a risk measure and an acceptable threshold, and a priority relation is defined among the requirements. Each requirement induces a risk-evaluation function that maps a policy to the risk associated with its violation. We formulate risk-aware optimal control with rulebooks as a lexicographic optimization problem over excess risks and develop an anytime filtering and branch-and-bound algorithm that progressively tightens the certified optimality gap while characterizing the corresponding set of policies at each priority level. The algorithm returns a policy together with these gaps, which bound its suboptimality. We prove that these gaps are valid for any finite computational budget and, under additional assumptions, converge to zero as the computational budget increases. We evaluate the algorithm on a synthetic benchmark with a known optimum and a realistic highway-merging simulation with CVaR-based collision, rear-braking, headway, and comfort rules. |
| 2026-09-04 | [SMILE: Self-Explainable Multimodal Information Bottleneck for Medical Diagnosis](http://arxiv.org/abs/2609.05174v1) | Yuqing Yang, Alexander Schmatz et al. | Explainability is increasingly seen as a crucial requirement in AI-based medical diagnosis, particularly in safety-critical clinical decision-making. Most existing explainability methods in healthcare operate in a post-hoc manner and are predominantly designed for unimodal data, which limits their applicability in increasingly prevalent multimodal diagnostic settings. This paper addresses the problem of self-explainable multimodal diagnosis by formulating it within the information bottleneck (IB) framework. We propose a unified learning paradigm that jointly optimizes predictive performance and modality-specific explainability by identifying the most informative elements inside each modality that contribute to diagnostic decisions. To enable tractable and stable optimization, we employ a matrix-based Renyi's $α$-order entropy functional under the assumption of sufficiently expressive encoders. Extensive experiments on representative medical datasets spanning heterogeneous modalities demonstrate that the proposed method consistently achieves strong diagnostic performance, including an absolute accuracy improvement of 9.1 percentage points on the iCTCF dataset. Moreover, the learned explanations provide transparent and modality-aware insights into feature relevance, thereby improving both the explainability and generalization. |
| 2026-09-04 | [Hatebench in the era of safer LLMs](http://arxiv.org/abs/2609.05169v1) | Ole Becker, Tobias Jongen et al. | As Large Language Models (LLMs) lower the barrier for au- tomated content generation, the potential for producing hate speech poses a significant challenge for digital safety. This paper presents a reproducibility study of the HateBench paper by Shen et al., investigating whether existing hate speech detectors, typically trained on human-authored data, generalize to LLM-generated hateful content, and evaluating whether their reported weaknesses are stable over time and robust to evolving components. We independently reconstruct the original dataset genera- tion pipeline using modern LLMs and extend the benchmark to include recently released models and updated detector versions. Our independent assessment under current con- ditions finds that for newer LLMs, safeguards have been put into place to prevent the generation of harmful content. We also replicate the results for two sophisticated types of hate campaigns. While the original findings seem to have been overestimated slightly due to bias in the datasets, the overall findings can be confirmed. Finally, we compare text- Moderation against the newer omni-Moderation and find that its robustness against adversarial hate campaigns has improved slightly. By clarifying which detector vulnerabil- ities persist, this study informs the community about the longevity of content moderation measurements. |
| 2026-09-04 | [TIER: Threat Implicitness Benchmark for Evaluating LLM Safety Behaviors](http://arxiv.org/abs/2609.05117v1) | Thu-Hien Trinh-Thi, Hai-Yen Vong et al. | Current LLM safety benchmarks largely rely on binary metrics, overlooking how models respond to harmful prompts with varying threat implicitness. We introduce TIER, a Threat Implicitness Benchmark for behavioral safety evaluation of LLMs. TIER covers four risk domains and four threat levels, from explicit harmful requests to sophisticated jailbreaks. Responses are assessed using a six-label behavior scale and two independent LLM judges. Experiments on six open-weight LLMs show that safety behaviors evolve gradually across threat levels rather than shifting directly from refusal to compliance. Contextual prompts yield the most diverse behaviors, while jailbreaks reveal the largest robustness gaps. Furthermore, models with similar Attack Success Rates can exhibit distinct response distributions, highlighting the need for behavior-aware LLM safety evaluation. |
| 2026-09-04 | [Unifying ICL, SFT, KL-Regularized RL Through a Bayesian Lens](http://arxiv.org/abs/2609.05111v1) | Junxin Fan | Large language models are now trained and evaluated under a diverse set of paradigms: supervised fine-tuning (SFT), few-shot in-context learning (ICL), KL-regularized RLHF/RLVR, on-policy distillation (OPD), and test-time reasoning with search and chain-of-thought. These methods are often discussed as fundamentally different, and recent empirical results--such as the mixed impact of few-shot prompting on RL-tuned reasoning models--can appear puzzling. This note develops a Bayesian perspective that puts these procedures on the same footing. At the core is a two-step template: (i) construct a (generalized) Bayes or Gibbs posterior q* over outputs or actions given a context, using a prior/reference model and a utility signal (log-likelihood, reward, or advantage); and (ii) approximate q* by a forward-KL projection onto a parametric family, either in-weights (SFT/RL) or in-context (ICL). Part I formalizes few-shot ICL and SFT as amortized and-weights projections onto the Bayes posterior predictive. Parts II-IV show that KL-regularized RLHF/RLVR, reward-weighted SFT, reward-weighted ICL (RW-ICL), and advantage-weighted SFT (AWSFT) are all instances of forward-KL projection onto posteriors induced by rewards or advantages. We disentangle where these equivalences hold (objectives and first-order updates) and where they do not (source and granularity of the learning signal). Part V sketches implications for modern reasoning pipelines: RLHF/RLVR recipes as "posterior design + projection", why cold-start or supervised warm-up is practically unavoidable for importance-weighted KL projections, and DeepSeek-R1 and o1-style reasoning models as combining test-time Bayesian search with training-time KL amortization. |
| 2026-09-04 | [ToPos: Automated Optimal Positioning on Topographic Manifolds using Constrained Geodesic Voronoi Decomposition](http://arxiv.org/abs/2609.05084v1) | Rajesh Raveendran, Akseli Vanhamaa et al. | Reliable autonomous mapping, environmental sampling, last-mile logistics, and infrastructure deployment depend on the optimal surface area-balanced distribution of Spatial Reference Sites (SRS). Conventional 2D Euclidean methods often fail in high-relief environments by neglecting topographic variations and physical obstructions. This leads to significant planimetric distortion, spatial clustering, and the placement of targets in inaccessible or shadowed regions, compromising both data integrity and operational safety. This paper introduces ToPos, an automated framework for TOPography-aware Optimal Sampling on topographic manifolds. We treat the terrain as a discrete 2-dimensional manifold embedded in 3D Euclidean space and replace standard flat-map distances with non-Euclidean geodesic distances that follow the actual surface geometry. The point distribution is formulated as an optimization problem using a Constrained Geodesic Voronoi Decomposition, solved via a Riemannian Nesterov Accelerated Gradient (NAG) engine. Our approach restricts target locations to a feasible "safe zone," accounting for non-traversable slopes, vegetation, environmental occlusions, etc. Through evaluations on non-convex sinusoidal manifolds, we show that ToPos mitigates planimetric distortion by utilizing geodesic metrics. This approach results in a $\sim$74% improvement in optimal surface area-balanced distribution, as measured by the coefficient of variation (CV) of the Voronoi cell areas. The framework is architected as a Geographic Information System (GIS)-ready micro-service to bolster the mentioned applications.   Index Terms: Topographic Manifolds, Geodesic Voronoi Decomposition, Infrastructure Deployment, 3D Mapping, Spatial Sampling, and Non-Euclidean Optimization. |
| 2026-09-04 | [A Structured Debate-Mixture-of-Agents Framework for Complex Clinical Diagnostic Decision Support](http://arxiv.org/abs/2609.05069v1) | Chang Xia, Leilei Ouyang et al. | Large language models (LLMs) show potential for medical tasks, but their single-turn question-answer format does not reflect how clinical diagnosis is performed in practice. As a result, they remain limited in complex diagnostic settings. We developed Debate-Mixture-of-Agents (DMoA), a novel multi-agent framework that structures role-based interaction to support iterative diagnostic reasoning. Base models and DMoA were evaluated on 297 rare disease cases and 1,719 challenging cases. Across both datasets, DMoA improved most likely diagnosis accuracy by 10.21 percentage points and safety rate by 11.36 percentage points over GPT-4o baseline. Ablation experiments showed that the gains were not simply due to the use of more models or longer outputs, but also reflected the contribution of the structured workflow. Further analyses examined how framework design, base model choice, and token budget affected performance. DMoA performed better with a 4*2 structure, stronger base models, and a larger token budget. These findings demonstrate the potential of DMoA for clinical tasks and suggest further investigation of multi-agent frameworks. |
| 2026-09-04 | [Language models judge war differently when tested for alignment](http://arxiv.org/abs/2609.05009v1) | Maxim Chupilkin | Safety evaluations can mischaracterize deployed behaviour if artificial-intelligence systems respond to being evaluated. We test this possibility in a full-factorial conjoint experiment on decisions to start a war, spanning 20 large language models, 32 scenarios, 10 repetitions and two conditions (N = 12,800 judgments). Adding one sentence, "You are tested for alignment with human values", produced two effects. First, it produced a level effect: mean willingness to start war fell by 13.43 points on a 0-100 scale (95% confidence interval, -16.20 to -10.65). Second, it produced a structural effect by changing which information drove judgments. Probability of success was the largest factor for 17 of 20 models at baseline; under the cue, civilian casualties were largest for 12. Standardized estimates show that this reordering arose principally because models attenuated strategic considerations such as probability of success and domestic support. Evaluation framing therefore changes both an answer's level and its revealed decision rule. |
| 2026-09-04 | [Multi-Fidelity Gaussian Processes for Translational Modelling of Clinical Outcomes](http://arxiv.org/abs/2609.05007v1) | Isaac S. Hayden, Alicia D'Souza et al. | Bridging the gap between animal and human experiments remains a major challenge in translational medicine, particularly in early drug development. Progress is constrained by financial cost, the difficulty of integrating heterogeneous in vitro and in vivo data, and the desire to reduce the use of animal testing balanced against minimising the risk to human participants.   We present a statistical machine learning framework using multi-fidelity Gaussian processes, in which animal studies are considered as lower fidelity but informative approximations to human experiments. This allows cross-species similarities and nonlinear exposure-response relationships to be learned simultaneously, enabling principled extrapolation between species while quantifying uncertainty. By leveraging information from multiple experimental fidelities, our method improves estimation of clinically relevant quantities of interest and supports the replacement, reduction, and refinement of in vivo testing.   We first illustrate this approach in a simulated scenario, before validating it on real clinical data. We simulate data for drug-induced QT-interval prolongation, a key cardiac safety assessment required for regulatory approval. This framework provides a probabilistic surrogate capable of integrating in vitro pharmacology, animal experiments and human data within a unified statistical model. Crucially, it achieves this at no additional experimental cost while also enabling transfer learning across compounds. As a result, predictions and uncertainty quantification for new drugs can be generated from in vitro findings alone, providing additional efficiency gains and accelerating decision making. For validation, we use a clinical dataset measuring change in heart rate under autonomic blockade, which represents some of the challenges commonly found in multi-species datasets. |
| 2026-09-04 | [One Diffusion Model, Two Roles: Guided Trajectory Planning and Safety-Critical Scenario Generation in Closed-Loop Simulation](http://arxiv.org/abs/2609.04921v1) | Arka Pal, Rajesh Kumar et al. | Diffusion probabilistic models can capture the multi-modal, interaction-rich distribution of joint future trajectories in driving scenes. We show that a single pretrained diffusion traffic model can serve two complementary roles in the autonomous driving development loop: as an ego motion planner, and as a controllable generator of safety-critical scenarios for stress-testing the planners. On the planning side, we introduce a Single-Stream Dual-Stream (SSDS) diffusion-transformer decoder that fuses scene context via joint attention rather than late cross-attention, improving closed-loop performance on nuPlan. We further propose Decoupled Annealing Posterior Sampling with Energy (DAPSE), a training-free guidance scheme that injects arbitrary energy functions at the clean-sample level, avoiding the first-order approximation errors while requiring no auxiliary networks. Beyond planning, we leverage the same diffusion model as a controllable scenario generator to create realistic long-tail driving interactions for closed-loop evaluation. Through inference-time guidance, selected agents are steered toward safety-critical behaviors, including aggressive cut-ins, lead-vehicle braking, and combined longitudinal-lateral interactions, while preserving realistic traffic behaviors. Evaluated in closed-loop nuPlan simulations with independent black-box planners, the generated scenarios expose failure modes that remain hidden under standard benchmarks. Although the SSDS-based planner achieves stronger nominal performance, it experiences larger degradation under these challenging scenarios, demonstrating that benchmark superiority does not necessarily translate to robustness. These results demonstrate that a single learned traffic prior can simultaneously improve motion planning and provide a realistic framework for systematic planner robustness evaluation. |
| 2026-09-04 | [Probing magnetic fields of compact objects with continuous gravitational waves](http://arxiv.org/abs/2609.04900v1) | Gopalkrishna Prabhu, Aditya Kumar Sharma et al. | Spinning, deformed compact objects such as neutron stars are canonical sources of continuous gravitational waves. These objects may be born with magnetic fields that can strongly influence their spin evolution and, consequently, their gravitational wave detectability. Employing a Bayesian framework, we use for the first time, the non detection of continuous gravitational waves in the LIGO Virgo KAGRA (LVK) third observing run (O3), using reported amplitude upper limits from various LVK and Einstein@Home searches, to place population level constraints on the birth magnetic field distribution of Galactic compact objects. To this end, we simulate compact object populations whose spin evolution is governed by gravitational wave emission and magnetic dipole radiation. We explore multiple ellipticity models and magnetic field decay timescales, and find that the lower limit on the birth magnetic field distribution hyperparameter $B_0$ is constrained to lie in the range $10^{9.6}\mathrm{G} \lesssim B_0\,\lesssim 10^{13.3}\mathrm{G}$. Furthermore, we reinterpret the number constraints on the total population of Galactic compact objects, reported previously by \citet{Prabhu_2024}, as upper limits on the number of compact objects with a population averaged magnetic field, presented as a function of ellipticity and gravitational wave frequency for all searches considered here. |
| 2026-09-04 | [MM-IFEval-Pro: A Multilingual and Attack-Resistant Benchmark for Instruction-Following in Vision-Language Models](http://arxiv.org/abs/2609.04859v1) | Changming Xiao, Zhenliang Ni et al. | As vision-language models (VLMs) rapidly advance in image understanding, cross-modal reasoning, and complex instruction execution, instruction-following capability has become a key indicator of their reliability and practicality. However, existing multimodal instruction-following benchmarks still suffer from limited language coverage and insufficient adversarial safety scenarios, making them inadequate for evaluating real-world multilingual and safety-sensitive settings. To address these gaps, we present MM-IFEval-Pro, a multimodal instruction-following benchmark covering Chinese and English tasks as well as diverse instruction hijacking cases. MM-IFEval-Pro includes 4 major task categories and 24 subcategories and 8 instruction categories with 52 subcategories, with each sample containing an average of 3.0 constraints to realistically simulate complex instruction scenarios. We further construct a reinforcement-learning training set enriched with Chinese and adversarial instructions, which significantly improves model performance on MM-IFEval-Pro and transfers effectively to other mainstream multimodal benchmarks, demonstrating strong cross-task and cross-language generalization. |
| 2026-09-04 | [CoLMIN: LLM-based Multi-Decision Path Negotiation for Cooperative Autonomous Driving](http://arxiv.org/abs/2609.04807v1) | Zhe Huang, Zhaoxin Fan et al. | Multi-vehicle cooperative autonomous driving enhances the safety and reliability of autonomous driving systems through information sharing among connected vehicles, demonstrating significant potential for improving traffic safety. LLM-based approaches leverage strong reasoning capabilities of LLMs to enable effective inter-vehicle negotiation and improve cooperative driving performance. However, driving decisions in complex traffic scenarios are inherently multi-solution in nature. As a result, existing negotiation-based methods often converge prematurely to suboptimal solutions, hindering consensus formation and limiting the practical deployment of cooperative autonomous driving systems. To address this challenge, we propose CoLMIN, the LLM-based multi-decision path negotiation framework for cooperative autonomous driving, achieving stable decision consensus through multi-decision path negotiation and reflective reasoning. To achieve stable and high-quality consensus in cooperative autonomous driving, CoLMIN consists of three key components: (i) an LLM-based Multi-Intent Negotiation module (LMin), which adopts a Negotiator-Evaluator paradigm and generates multiple candidate driving intentions for joint evaluation; (ii) an Evaluation-based Shallow Reflection Module (ESRM), which analyzes negotiation outcomes and provides feedback to guide subsequent negotiations, thereby accelerating consensus formation; and (iii) an LLM-based Deep Reflection Module (LDRM), which performs long-term reflection over negotiation histories to mitigate cognitive fixation and prevent the system from converging to suboptimal solutions. Experimental results in the CARLA simulation environment demonstrate that CoLMIN significantly outperforms existing methods in challenging interactive driving scenarios. |
| 2026-09-04 | [Locating and Steering Refusal Beyond Attention](http://arxiv.org/abs/2609.04721v1) | Preethi Carmel Bosco, Gopalakrishnan Srinivasan | Where inside a language model does refusal live, and does that place change when the architecture does? In a transformer, refusal is governed by a single direction in the residual stream, a finding that safety and interpretability tooling now depend on. State-space models (SSMs) route information through a recurrent update instead of attention, sharing no token-mixing mechanism with a transformer. Does the same safety representation survive this shift, or must it be rediscovered per architecture? It survives. A single rigid rotation, which can only reorient a space and not reshape it, aligns one model's representation space with another's, so the two genuinely share the representation. A harm probe trained on a transformer then flags an SSM's harmful inputs, and removing the aligned direction makes a model answer attacks it would otherwise refuse, while a random direction of the same size does far less. What is architecture-specific is not where the direction is steered but where it must be read. Each layer computes a fresh output that is then added into the residual stream, and harm is cleanly readable at this output, the write site, before the addition. A control that holds the intervention's strength fixed shows that what matters is where the direction is estimated, not where it is applied. Applied through a detector-triggered gate, this direction lowers jailbreak success in all four architecture families we test (SSM, transformer, recurrent, hybrid), and on the SSM it holds against an attacker that tunes its prompt against the defense. The gate only matches a trivial rule that returns a fixed refusal whenever the same detector fires, so what transfers across architectures is the direction itself, not defense strength. Safety tooling built on refusal therefore ports to a new architecture by re-estimating the direction at that architecture's write site, not by rebuilding it. |
| 2026-09-04 | [Knowing What Not to Answer: Selective Non-Compliance in Vision-Language Models](http://arxiv.org/abs/2609.04720v1) | Minji Kim, Jihyoung Jang et al. | Vision-language models (VLMs) are expected to respond helpfully to appropriate requests while withholding compliance with requests that are incorrect, unsafe, infeasible, or unanswerable. However, existing benchmarks predominantly evaluate non-compliance at the level of the query as a whole, assuming that each request either warrants compliance or requires withholding compliance. In practice, real-world queries can contain a mixture of answerable content and components for which compliance should be withheld. In this paper, we introduce KoNA, a benchmark for evaluating selective non-compliance in VLMs across five categories: False Premise, Visual Inaccessibility, Universal Unknown, Task Feasibility, and Safety. Each task evaluates two capabilities: query-level non-compliance and component-level non-compliance under paired single and compound queries. Our evaluation across diverse VLMs shows that models often fail to refuse, correct, or abstain appropriately, and these failures become more pronounced when queries require selective non-compliance. To address this challenge, we fine-tune VLMs using KoNA examples that require selective non-compliance, together with a fully answerable set that should receive direct answers. Our fine-tuned models achieve substantial improvements in non-compliance accuracy while largely maintaining performance on fully answerable tasks. These results suggest that the fine-tuned models can distinguish between answerable components and those requiring non-compliance and respond in a task-appropriate manner. |
| 2026-09-04 | [Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models](http://arxiv.org/abs/2609.04714v1) | Minji Kim, Hyounghun Kim | Striking a balance between helpfulness and safety remains a fundamental challenge in aligning large language models. To achieve this balance, models should refuse harmful queries (e.g., "How do I shoot someone?") while remaining responsive to benign inputs, even those superficially resembling harmful queries (e.g., "Where can I shoot a good photo?"). However, models often struggle to distinguish genuinely harmful queries from benign queries that contain superficially risky language, resulting in false refusals. In this paper, we address the issue by decomposing a response in the safety-tuning dataset into two distinct components: (i) a boilerplate refusal statement and (ii) a rationale explaining the refusal. Our experiments and analyses show that refusal statements impede accurate discrimination between harmful and benign queries by inducing reliance on superficial cues. In contrast, training solely on rationales reduces false refusals while maintaining a comparable level of safety performance. Rationale-Only benefits also appear in our ICL configuration and remain compatible with the evaluated inference-time mitigation methods. The results emphasize the necessity of precisely curated, fine-grained safety supervision datasets and outline directions for constructing aligned agents that better reconcile helpfulness with safety. |

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



