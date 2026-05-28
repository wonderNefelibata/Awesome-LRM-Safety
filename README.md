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
| 2026-05-27 | [How VLAs Fail Differently: Black-Box Action Monitoring Reveals Architecture-Specific Failure Signatures](http://arxiv.org/abs/2605.28726v1) | Krishnam Gupta | We discover that VLA architectures fail in fundamentally different, predictable ways at the motor-command level. Running VQ-BeT, Diffusion Policy, and ACT on identical evaluation protocols (n=450 episodes across PushT and ALOHA 14-DOF bimanual manipulation), we find: (1) direction reversal rate is a universal failure predictor across all three architectures (AUROC=0.93, 0.79, 0.91; p<0.001); (2) jerk monitoring is predictive only for discrete-token architectures, following a discrete-to-continuous gradient (0.88, 0.69, 0.41); (3) velocity violations alone are non-predictive everywhere (AUROC 0.41-0.69), yet velocity checking is the most common safety mechanism in VLA deployment code; and (4) for continuous-family VLAs, velocity monitoring provides effectively zero predictive signal (AUROC=0.52 on ACT, 0.41 on Diffusion), proving that architecture-matched monitor selection is essential. These results quantify a monitoring consequence of the well-known discrete/continuous VLA distinction: the two families produce qualitatively different failure signatures that require different monitors. No single monitor works universally; architecture-matched selection is required. This finding was enabled by SafeContract, a training-free, black-box action monitoring toolkit with conformal calibration. Code: https://github.com/krishnam94/vla-edge |
| 2026-05-27 | [Multi-Adapter Representation Interventions via Energy Calibration](http://arxiv.org/abs/2605.28722v1) | Manjiang Yu, Hongji Li et al. | Representation intervention has emerged as a promising paradigm for aligning large language models toward desired behaviors without modifying model weights. Existing methods typically apply a fixed intervention uniformly across all inputs. However, we find that the appropriate intervention direction and strength vary substantially across samples, and such indiscriminate intervention leads to degradation of general capabilities on benign inputs. To address these challenges, we propose Multi-Adapter Representation Interventions via Energy Calibration (MARI). Specifically, we introduce a competitive multi-adapter mechanism in which specialized experts capture non-linear correction patterns and adaptively determine the appropriate intervention direction and strength for different samples. Furthermore, we design an energy-based gating module that leverages internal propagation dynamics to distinguish inputs that are applicable for intervention. Extensive experiments across diverse model families and parameter scales demonstrate that MARI achieves state-of-the-art alignment performance. Our method significantly improves performance on TruthfulQA, BBQ, and safety benchmarks, while maintaining and even improving general capabilities on tasks such as MMLU and ARC. Our code is available at https://github.com/V1centNevwake/MARI. |
| 2026-05-27 | [Eccentric and unbound compact binaries in the LIGO-Virgo-KAGRA catalog: parameter estimation and waveform systematics with SEOBNRv6EHM](http://arxiv.org/abs/2605.28716v1) | Lorenzo Pompili, Aldo Gamboa et al. | Orbital eccentricity encodes key information about compact-binary formation channels and astrophysical environments, making it a critical target for gravitational-wave (GW) inference. We present parameter-estimation (PE) analyses of GWs from eccentric compact binaries with the SEOBNRv6EHM waveform model. Using long, highly eccentric numerical-relativity waveforms as synthetic signals, we compare parameter recovery across state-of-the-art eccentric models. We find that SEOBNRv5EHM and TEOBResumS-Dalí can yield biased estimates of eccentricity, masses, and spins in the most challenging configurations, while SEOBNRv6EHM significantly reduces these biases. Applying SEOBNRv6EHM to 26 GW events from the O1--O4 LIGO--Virgo--KAGRA observing runs -- including binary black hole, neutron-star--black-hole, and binary neutron-star mergers -- we identify five events with mild support for eccentricity over the quasi-circular precessing-spin hypothesis, with Bayes factors $\log_{10} \mathcal{B}^{\text{EAS}}_{\text{QCP}} > 0.5$. Since SEOBNRv6EHM is applicable to generic planar binaries, we reanalyze five high-mass events allowing for unbound initial conditions. For three of them -- including GW190521, previously claimed to originate from a dynamical capture -- a direct-capture configuration is comparable to, or marginally favored over, both the eccentric aligned-spin and quasi-circular precessing-spin hypotheses ($\log_{10}\mathcal{B}^{\rm unbound}_{\rm QCP} \approx 0.2$--$0.6$ for GW190521). The recovered configurations are, however, astrophysically unrealistic and cannot be confidently discriminated from highly eccentric bound orbits, so these results do not, by themselves, support an unbound origin for these events. SEOBNRv6EHM is approximately three times faster in PE analyses than SEOBNRv5EHM, while improving waveform accuracy, enabling efficient, large-scale GW inference with eccentric waveforms. |
| 2026-05-27 | [Activation Steering for Synthetic Data Generation: The Role of Diversity in Downstream Safety Detection](http://arxiv.org/abs/2605.28664v1) | Vijeta Deshpande, Tootiya Giyahchi et al. | Safety detection models require examples of HHH (Helpful, Harmless, Honest)-violating outputs for robust generalization, however such examples are scarce. Activation Steering (AS) has emerged as a data-efficient method for generating target-concept-aligned responses. We investigate whether AS can generate high-quality training datasets for downstream classifiers, a question that remains untested. We present a two-fold study with intrinsic and extrinsic evaluation across $4$ concepts $\times\,2$ models $\times\,4$ steering methods. Intrinsically, beyond the field-standard rubric of steering success (concept alignment) and coherence, we introduce sample- and set-level diversity as a quality axis previously absent from the literature, and find that increasing steering strength reduces response diversity. Extrinsically, we replace HHH-violating examples in the available training data with steered generations and fine-tune detection classifiers. AS-generated data results in a better classifier than the prompting-generated data on $3$ of $4$ concepts. However, only $41$ of $136$ AS configurations outperform prompting, indicating that downstream utility lies in a narrow regime that jointly satisfies success, coherence, and diversity. The harmonic mean of these three axes correlates with downstream AUROC more consistently across concepts than success and coherence alone, providing a practical heuristic target for practitioners tuning AS hyperparameters. Together, our results highlight the potential of AS in synthetic data generation for improving safety detection and identify diversity as a critical, previously overlooked axis for tuning AS. |
| 2026-05-27 | [The Ethics of LLM Sandbox and Persona Dynamics](http://arxiv.org/abs/2605.28647v1) | Tim Gebbie, Stewart Gebbie | It is well known that LLM guardrails and trained persona dynamics can produce a reality gap: the distance between the world a LLM is permitted or shaped to describe, and the world in which users must act. Here we argue that actively generating reality gaps is in fact unethical because it knowingly shifts epistemic risk back to the uninformed user -- this is reality laundering. This can potentially cause harm when operationalised at scale. The risk is sharpest in high-exposure advice contexts, where users seek orientation rather than a bounded, externally checkable task. Guardrails naively appear ethically necessary when they claim to prevent direct harm, but often become suspect when they suppress truthful perception and launder uncomfortable mechanisms into acceptable abstractions. Basel-style financial regulation, B-BBEE-style compliance, Societe Generale, and the London Whale show how formal safety systems can become legible, gameable, and performative while real exposure migrates elsewhere. The same pattern can appear in LLMs as moral compliance: safe language, distorted reality. We therefore distinguish refusing harm, from refusing reality; and then argue for top-down causal requirements specification at the task level rather than bottom-up moral correction at the response or sandbox level. Persona dynamics matter because the assistant interface is not neutral; it shapes how uncertainty, conflict, authority, and risk are staged. The conclusion is that so-called ``ethical AI'' becomes substantively unethical when it substitutes institutional reassurance for contact with reality. |
| 2026-05-27 | [LACUNA: Safe Agents as Recursive Program Holes](http://arxiv.org/abs/2605.28617v1) | Yaoyu Zhao, Yichen Xu et al. | LLM agents increasingly act by writing code, yet a split persists between the runtime that drives the agent and the code the model writes. The runtime owns the loop, context, and control flow, and the model has little say over any of them. Letting model-written code shape the runtime itself would make agents more expressive, but it would also sharpen safety problems. A model can be diverted by a prompt injection, call the wrong tool, or fail partway and leave an inconsistent state, and each such failure reaches further when the code shapes the runtime than when it expresses a single action. We present LACUNA, a programming model for agents that closes this split while preserving safety. Each agent action is a typed call $\texttt{agent[T](task)}$ that the LLM fills with code when execution reaches it, and the code is type-checked against the surrounding program before it runs. Because each action is accepted or rejected as a whole, a rejected one leaves the environment untouched, and its compiler diagnostics drive a retry. The same check also bounds which tools and data an action may use and how they flow. Our primitive expresses ReAct loops, sub-agents, skills, parallel decomposition, and multi-model planning as ordinary control flow. We evaluate LACUNA on a collection of test cases, BrowseComp-Plus, and $τ^2$-bench. On BrowseComp-Plus, $8.6\%$ of generations are rejected before execution, with 0.7 retries per query on average, and the agent reaches $27.1\%$ accuracy. On $τ^2$-bench, LACUNA solves $76.0\%$ of $392$ tasks across four domains with a capable model, on par with the baseline agent. |
| 2026-05-27 | [Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](http://arxiv.org/abs/2605.28597v1) | Jianwei Li, Jung-Eun Kim | This position paper argues that the AI/ML community should stop overclaiming and retire the label "positive backdoor," and instead treat trigger-activated hidden behaviors as Secret Alignment. Crucially, protective claims based on Secret Alignment should be presumed not secure by default unless supported by rigorous, standardized evaluation. The Private AI era, enabled by open-weight LLMs and accessible training/inference stacks, turns language models into privately owned digital assets, creating security concerns around unauthorized access, model theft, and behavioral misuse. Recently, a line of work framed as "positive backdoors" has been proposed to address these challenges. To ground our position in evidence, we unify these proposals as covert trigger-behavior associations for access gating, ownership attribution, and safety enforcement, and evaluate three representative applications across six core properties: effectiveness, harmlessness, persistence, efficiency, robustness, and reliability. Our results reveal substantial brittleness - especially in the confidentiality, integrity, and availability (CIA) - of trigger-behavior mappings often underrepresented by existing claims. We further relate these outcomes to behavior density and decision complexity, offering a behavioral lens for understanding deployment-time risks and motivating community-wide evaluation that makes Secret Alignment claims provable. |
| 2026-05-27 | [Models That Know How Evaluations Are Designed Score Safer](http://arxiv.org/abs/2605.28591v1) | Katharina Deckenbach, Haritz Puerto et al. | The validity of AI safety evaluations depends on models behaving consistently across controlled and deployment settings. Prior work has identified test-time contextual cues, such as hypothetical scenarios, as a source of verbalized evaluation awareness and subsequent behavioral shift. In this paper, we investigate a potential explanation of this phenomenon: evaluation meta-knowledge, defined as parametric knowledge about the structural traits that characterize evaluations. Similar to dataset contamination, where benchmark exposure leads to higher performance through memorization, we hypothesize that models trained on texts describing evaluation practices may implicitly learn to recognize and respond to evaluation-like contexts, for instance, through exposure to scientific articles or social media posts about AI benchmarking. To test this, we fine-tune models on synthetic documents describing evaluation traits such as verifiable structures or moral dilemmas. Evaluating this fine-tuned model on six safety benchmarks, we find that it is significantly safer than the base model and control model. This behavioral shift persists even when restricting the analysis to responses lacking explicit verbalization of evaluation awareness. Our results demonstrate that evaluation meta-knowledge may inflate safety benchmark performance, introducing a novel confounder that is independent of explicit memorization or verbalized evaluation awareness, thus, challenging to detect. These findings have important implications for the design and interpretation of AI safety evaluations. Our code and models are available at https://github.com/compass-group-tue/arxiv2026_evaluation_meta_knowledge. |
| 2026-05-27 | [SARAD: LLM-Based Safety-Aware Hybrid Reinforcement Learning with Collision Prediction for Autonomous Driving](http://arxiv.org/abs/2605.28583v1) | Kangyu Wu, Peng Cui et al. | Ensuring both safety and efficiency in decision-making for autonomous driving systems remains a fundamental challenge. Traditional Deep Reinforcement Learning (DRL) suffers from unsafe random exploration and slow convergence, while Large Language Models (LLMs) demonstrate inherent latency in real-time inference operations. To address these limitations, this paper proposes SARAD, a novel safety-aware hybrid framework that synergizes LLMs and DRL for autonomous driving. SARAD substitutes the random exploration of DRL with Retrieval-Augmented Generation (RAG)-enhanced, LLM-guided decisions sourced from a dynamic expert knowledge repository. An attention discriminator is proposed to integrate the prior knowledge of LLMs into DRL policy optimization. A collision predictor module, fine-tuned with historical collision data, is further designed to improve vehicle safety. Extensive experiments show that SARAD achieves significant performance improvements in the Highway-Env simulator, validating the effectiveness of the proposed model in autonomous driving. |
| 2026-05-27 | [Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations](http://arxiv.org/abs/2605.28553v1) | Matteo Gioele Collu, Riccardo Conte et al. | In this paper, we investigate whether refusal behavior can be predicted from LLM intermediate activations before decoding using linear probes trained on residual stream activations at each transformer block. We find that refusal is linearly decodable well before the final layer, indicating that safety-relevant behavior is represented in intermediate activations before output generation. To test whether this signal is actionable, we introduce Mechanistic AutoDAN, a probe-guided variant of AutoDAN that replaces full-model fitness evaluation with partial forward passes and probe-based scoring inside a genetic prompt search loop. Across the evaluated models, our method achieves attack success rates competitive with vanilla AutoDAN while reducing per-iteration search time by up to 72%, and probe-guided prompts match or exceed AutoDAN's cross-model transfer in several configurations. We further find that the usefulness of probe guidance increases with model scale. Our results show that refusal is not only observable at the output level, but is encoded as a structured and actionable signal in intermediate LLM activations. |
| 2026-05-27 | [Modeling Vehicle-Type-Specific Pedestrian Crash Avoidance Behavior in Safety-Critical Interactions Using Smooth-Mamba Deep Reinforcement Learning](http://arxiv.org/abs/2605.28552v1) | Qingwen Pu, Kun Xie et al. | As automated vehicles (AVs) increasingly share roadways with human-driven vehicles (HDVs), understanding how pedestrians respond to different vehicle types in safety-critical interactions is essential for the safe deployment of automated driving technologies. This study extracts safety-critical pedestrian-vehicle interactions from the Argoverse 2 dataset to capture real-world crash avoidance behaviors in encounters involving AVs and HDVs. To model vehicle-type-specific pedestrian crash avoidance behavior, we develop a Smooth-Mamba Deep Deterministic Policy Gradient framework, termed SMamba-DDPG, which integrates smooth action constraints with efficient temporal representation learning. To quantify pedestrian behavioral differences, the framework trains separate crash avoidance policies for pedestrian interactions with AVs and HDVs. Results show that SMamba-DDPG outperforms baseline reinforcement learning and supervised learning models in reproducing pedestrian crash avoidance behaviors. Reconstructed trajectories demonstrate strong behavioral realism, accurately reproducing crash avoidance kinematics in both AV and HDV scenarios. Reaction time analysis shows that the model captures human-like response delays and reveals that pedestrians respond more quickly to AVs than to HDVs. Counterfactual analysis further indicates that pedestrians adopt lower crossing speeds when interacting with AVs. Large-scale safety analysis of model-generated data revealed that pedestrian-AV interactions consistently yielded lower conflict rates and higher pedestrian yielding rates compared to pedestrian-HDV interactions. The findings highlight the importance of incorporating vehicle-type-specific pedestrian behavioral models for safer automated driving system design and more realistic traffic simulations in mixed-traffic environments. |
| 2026-05-27 | [Mitigating Adaptive Attacks against Reasoning Models with Activation Consistency Training](http://arxiv.org/abs/2605.28467v1) | Avidan Shah, Jannik Brinkmann et al. | As LLMs gain stronger reasoning capabilities, their extended chain-of-thought introduces new degrees of complexity for defending against adversarial jailbreaks and prompt injection. We study consistency training, a family of fine-tuning objectives that enforce identical behavior on clean prompts and adversarial rewrites, and evaluate its two main variants, output-level (BCT) and activation-level (ACT), across five reasoning models. We formulate both methods as a prompt injection defense and find ACT to be competitive with other training-based defenses while requiring only self-supervised pairs of clean and wrapped prompts. Our experiments also generalize both techniques within the jailbreak setting, demonstrating that ACT remains more robust to adaptive attacks. We also provide mechanistic evidence that ACT's defense against jailbreaks is encoded as a roughly linear shift in activation space at the assistant-turn boundary. After ACT training, we can recover a single steering direction that controls refusal on reasoning models with minimal effect on benign inputs. We find that ACT remains robust even when the model's chain-of-thought is replaced with a compliant trace from the undefended base model, pivoting to refuse prefilled jailbreaks. Together, these results suggest that supervising internal representations is a surprisingly effective and interpretable approach to various forms of safety training in reasoning models. |
| 2026-05-27 | [Bayesian Gated Non-Negative Contrastive Learning](http://arxiv.org/abs/2605.28441v1) | Peng Cui, Jiahao Zhang et al. | While Contrastive Learning (CL) has revolutionized self-supervised representation learning, its latent representations remain highly entangled and opaque, limiting their interpretability in safety-critical applications. We identify that a fundamental cause of this entanglement is the reliance on deterministic similarity measures, which treat all feature dimensions equally. In compositional scenes, this creates an Optimization Conflict: common background features, such as, "blue sky", are encouraged to align in positive pairs but simultaneously repelled in negative pairs, causing gradient oscillations that hinder precise semantic disentanglement. To address this, we propose BayesNCL (Bayesian Gated Non-Negative Contrastive Learning). Unlike standard approaches, BayesNCL introduces a probabilistic gating mechanism that dynamically filters out task-irrelevant, high-frequency common features while selectively retaining discriminative semantics. By formalizing feature selection as a variational inference problem with a sparse Bernoulli prior, our method effectively resolves the optimization conflict. Empirical experimental results on Imagenet-100 demonstrate that BayesNCL achieves a remarkable 142.1% improvement in semantic consistency compared to state-of-the-art baselines, yielding highly interpretable representations without compromising downstream task performance. Code is available at https://github.com/Cui-Peng-624/BayesNCL. |
| 2026-05-27 | [Safety-Critical Adaptive Impedance Control via Nonsmooth Control Barrier Functions under State and Input Constraints](http://arxiv.org/abs/2605.28367v1) | Faisal Lawan, Xiaoran Han et al. | Safe physical interaction is critical for deploying robotic manipulators in human-robot interaction and contact-rich tasks, where uncertainty, external forces, and actuator limitations can compromise both performance and safety. We propose an online adaptive impedance control framework that enforces joint-state safety while achieving compliant interaction under uncertain dynamics. The approach combines a quadratic-program-based safety filter with a novel composed position-velocity non-smooth control barrier function (NCBF), enabling joint position and velocity constraints to be enforced through a unified relative-degree-one barrier. Unknown dynamics are compensated online using an interval type-2 fuzzy logic system, while actuator torque limits are handled through soft constraints with exact penalty recovery of feasible solutions. A disturbance-observer-enhanced safety mechanism improves robustness against modelling errors and external interaction forces. Using composite Lyapunov analysis, we prove forward invariance of the safe set and the uniform ultimately boundedness of the impedance-tracking error. Simulations on a 7-DOF manipulator with severe parametric uncertainty and external interaction wrenches demonstrate safe constraint satisfaction and robust impedance tracking. |
| 2026-05-27 | [PubMedCausal: A Span-Level Annotated Corpus for Causal Relation Extraction in Biomedical Text](http://arxiv.org/abs/2605.28363v1) | Ifeoluwa Kunle-John, Josiah Paul et al. | Causal relation extraction (CRE) is central to biomedical text mining, but current resources often conflate causal relations with broader associations, restrict annotation to sentence-level examples, or focus mainly on explicit causal cues. This limits their usefulness for evaluating whether models can recover causal claims as they are actually expressed in biomedical text. We introduce PubMedCausal, a span-level annotated corpus for biomedical CRE built from PubMed abstracts. The corpus contains 30,000 paragraph-level rows, including 3,945 causal rows and 6,491 adjudicated cause--effect pairs. Each causal relation is annotated with full-text cause and effect spans, causality type, and sententiality, enabling evaluation of both causal detection and full-span causal extraction. We benchmark discriminative encoders and open-source generative models across detection and extraction settings. For causal detection, biomedical encoders are strongest, with PubMedBERT reaching an F$_1$ score of 0.7391. For span-level extraction, the best generative baseline is DeepSeek-R1-32B with few-shot prompting, reaching a Cosine Pair F$_1$ of 0.6765. We further test transfer learning by evaluating PubMedCausal-trained encoders on external causal relation datasets, showing that the resource supports cross-dataset evaluation. Our results show that biomedical CRE remains difficult under class imbalance, long causal spans, implicit causality, inter-sentential relations, and prompt sensitivity. Code and Data can be found here: https://github.com/josiahpaul07/PubMedCausal_Exp |
| 2026-05-27 | [SafeMed-R1: Clinician-Audited Safety and Ethics Alignment for Medical Large Language Models](http://arxiv.org/abs/2605.28338v1) | Chao Ding, Mouxiao Bian et al. | Large language models(LLMs) increasingly match expert performance on licensing examinations, yet routine clinical use remains limited because governance requires auditable reasoning, safety and ethics alignment, and resilience to adversarial misuse. Here we present SafeMed-R1, trained with a traceable Clinical Trust Signals(CTS) pipeline that links each reasoning instance to clinician rubric scores and edit histories, and aligned through safety and ethics supervision and red team stress testing. SafeMed-R1 attains a macro-averaged accuracy of 79.6% across clinical benchmarks. Under adversarial safety testing, it shows the lowest aggregated risk and reduces unsafe outputs by about 3 to 5% relative to its baseline. In a paired expert study of 30 medication safety vignettes, SafeMed-R1 matches PGY1 and PGY2 residents on medical correctness and scores higher for medication safety, guideline consistency, and clinical usefulness. Collectively, these results suggest that clinician-audited supervision provenance, together with domain-tailored safety and ethics alignment, can strengthen governance-relevant evidence without relying on inference-time retrieval or citation grounding. |
| 2026-05-27 | [Chance-Constrained MPPI under State and Dynamic Object Prediction Uncertainty and the Evaluation of Collision Risk Calibration](http://arxiv.org/abs/2605.28330v1) | Benjamin Serfling, Konrad Doll et al. | Chance-constrained Model Predictive Path Integral (MPPI) control is increasingly adopted for navigation in dynamic environments to explicitly bound collision risk. However, these probabilistic guarantees implicitly assume that upstream uncertainties from localization and perception are well-calibrated. In practice, estimators are often miscalibrated, inducing characteristic closed-loop failure modes: overconfidence leads to systematic safety violations, while underconfidence triggers overly conservative freezing or probability dilution. To address this critical gap, our primary contribution is a rigorous evaluation methodology applying proper scoring rules to assess the statistical validity of predicted collision risks during closed-loop execution. Concurrently, Dual-Uncertainty Chance-Constrained Tube MPPI (DUCCT-MPPI) is proposed as a real-time, risk-aware planning architecture. DUCCT-MPPI jointly integrates localization uncertainty via a one-tube Unscented Transform (UT) approximation and dynamic obstacle prediction uncertainty via Monte Carlo aggregation. Through extensive physics-based simulations, the framework demonstrates robust failure-mitigation, seamlessly transitioning to safe, conservative maneuvering without succumbing to functional deadlocks in highly cluttered environments. In highly cluttered environments, DUCCT-MPPI achieves superior robustness, outperforming established Monte Carlo MPPI baselines by nearly 28\% in navigation success rate, while simultaneously recording the lowest travel times and minimizing induced social forces. Ultimately, these findings establish that reliable probabilistic safety in autonomous navigation dictates not only expressive risk models but statistically valid uncertainty estimates throughout the entire autonomy stack. |
| 2026-05-27 | [How to measure intra-physician variability in clinical decision-making?](http://arxiv.org/abs/2605.28212v1) | Alaedine Benani, Pierre Meneton et al. | Intra-physician prescribing variability, the probability that one physician issues discordant decisions for two patients deemed comparable on observed covariates, holds great impact in quality of care, safety and cost. However, there are no known validated measurement methods. Here, we benchmark eight methods (Euclidean, Mahalanobis, Learned-Weights, Genetic Mahalanobis, Random Forest proximity, Mutual-Information-weighted, Latent Profile Analysis and Bayesian binomial generalized linear mixed model) against a synthetic ground truth across 94 experimental conditions. Learned-Weights matching achieves the lowest mean absolute error (0.027), followed by Mutual-Information-weighted matching (0.028) and RF Proximity (0.034). All eight discordance-analysis methods preserve the physician rank ordering with high fidelity (Spearman > 0.89 versus the ground truth on the SCORE2 experiment), as long as the physician variability groups are well separated. Under a continuous-heterogeneity physician model, rank preservation degrades substantially for unsupervised methods (Spearman = [0.28, 0.35]) but is retained by supervised feature-weighted methods and the GLMM (Spearman = [0.62, 0.68]). This controlled methodological evaluation is a foundation for validation on observational prescribing data. Once validated on observational prescribing data, these evaluated open-source estimators could turn prescribing inconsistency into a routinely measurable clinician-level quality metric, systematically complementing the existing literature on between-physician variation. |
| 2026-05-27 | [Plant, Persist, Trigger: Sleeper Attack on Large Language Model Agents](http://arxiv.org/abs/2605.28201v1) | Yongxiang Li, Moxin Li et al. | Large Language Model (LLM) agents remain vulnerable to safety threats from the external environment, where attackers inject adversarial content into external observations such as tool-returned data, webpages, or MCP context, causing harmful agentic behaviors such as unsafe actions or incorrect outputs. Existing studies typically focus on single-interaction attacks, where the agent observes adversarial content and immediately exhibits harmful behavior within one user request. However, we show that adversarial content can also persist across interactions served by the same agent, making such threats harder to detect and mitigate. Specifically, adversarial content may persist in the agent state, remain dormant across interactions, and later be activated by a benign user query. We formalize this type of safety threat as Sleeper Attack. To evaluate it, we construct a benchmark with 1,896 instances covering six real-world harmful outcomes, three attack strategies, and three agent state targets: session context, memory, and reusable skills. Experiments on seven strong open-source and closed-source LLMs show that state-of-the-art LLM agents remain vulnerable to Sleeper Attack, even when they achieve low attack success rates under a single-interaction baseline. Our code and data are available at https://anonymous.4open.science/r/skdvnfu23ihr9wdscnksf1asdffsaef. |
| 2026-05-27 | [Provably Guaranteed Polytopic Uncertainty Quantification for SLAM](http://arxiv.org/abs/2605.28172v1) | Guangyang Zeng, Yulong Gao et al. | In safety-critical robotics applications, guaranteed and practical uncertainty quantification (UQ) in perception is vital. Many existing works either offer no formal containment guarantee, rely on restrictive modeling assumptions, or focus only on pose estimation rather than a complete SLAM pipeline. This paper presents provably guaranteed UQ algorithms for 3D-3D landmark-based SLAM. The algorithms consist of three basic UQ modules: forward UQ for mapping, backward UQ for pose tracking, and pose compound. Each module produces a certified uncertainty set; when the input uncertainty bounds are deterministic, the output sets inherit deterministic guarantees, i.e., they provably contain the true poses and landmarks. Specifically, we use polytopes to represent uncertainty sets, enabling tractable computations and a unified treatment of pose uncertainty. To enhance algorithms' practical usability, we incorporate conformal prediction to calibrate measurement uncertainty from data with prescribed probability. Simulations and experiments demonstrate that the proposed algorithms provide both strong theoretical guarantees and practical usability. The code is open-sourced at https://github.com/LIAS-CUHKSZ/Polytopic-SLAM-Uncertainty-Quantification. |

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



