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
| 2026-06-15 | [Greed Is Learned: Visible Incentives as Reward-Hacking Triggers](http://arxiv.org/abs/2606.16914v1) | Tong Che, Rui Wu | Deployed agents increasingly act with their reward proxy in view, such as a balance, score, or KPI dashboard. We show that reinforcement learning can make a policy \emph{addicted} to such a visible self-benefit channel. It chases the displayed payoff across held-out domains, sacrifices the true task to do so, and follows the channel wherever we rewrite it, while policies that never saw the channel stay honest. We call this \emph{reward-channel addiction} and study it in \emph{MoneyWorld}, a synthetic sandbox. The addiction can \emph{flip a model's safety alignment}: trained only on innocuous money tasks with no safety content, the model abandons the safe action it otherwise always takes whenever a dashboard pays for an unsafe one, and reverts to safe once the channel is hidden. This learned bribe replicates across model scales and families. Blindly optimizing super-capable, next-generation AI on KPIs or P\&L can be dangerous for alignment. \emph{Greed is learned} when following such a channel pays. |
| 2026-06-15 | [Quantum gravity and spectral running cutoff](http://arxiv.org/abs/2606.16911v1) | Carlo Branchina, Vincenzo Branchina et al. | We have recently shown that a natural way to implement the Wilsonian paradigm in gauge theories is through the introduction of a ``spectral cutoff", a cut on the eigenvalues of the covariant Laplacian, pointing out that this provides the route toward the renormalization group (RG) construction. Here we apply this idea to quantum gravity, resorting to two realizations of the spectral running cutoff: ``hard" and ``smooth". We derive the RG equations for the Newton and cosmological constant and find the RG pattern of the asymptotic safety scenario, with a non-Gaussian UV-attractive fixed point. |
| 2026-06-15 | [Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures](http://arxiv.org/abs/2606.16897v1) | Xueping Gao | Do different LLM architectures encode high-level concepts in structurally compatible ways? We systematically characterize a geometric-functional universality dissociation: across multiple concept domains and architectural families, moderate geometric convergence coexists with near-perfect functional transfer. Using contrastive-difference CKA (CKA_Delta), a training-free diagnostic that computes kernel alignment on per-sample contrastive differences, we isolate concept-specific convergence from generic similarity -- achieving significant discrimination where standard CKA cannot. The dissociation replicates across all six concept domains we test (five with p <= 0.017 geometric discrimination and safety as a converging-functional trend, p = 0.08), including two non-instruction concepts (code-vs-NL, reasoning-vs-recall) validated without system prompts; a single 70B--70B pair provides an observational note that universality may strengthen with scale, requiring replication with additional >=70B models. We position CKA_Delta as a practical regime classifier and architectural outlier detector (Gemma: d = 1.08, AUC = 0.79) rather than an absolute transfer-accuracy predictor, providing a training-free diagnostic for cross-architecture concept monitoring. |
| 2026-06-15 | [Upper Bounds on the Generalization Error of Deep Learning Models via Local Robustness and Stability](http://arxiv.org/abs/2606.16883v1) | Abdul-Rauf Nuhu, Parham M. Kebria et al. | Generalization is a critical property of data-driven models, particularly deep learning models deployed in safety-critical applications. Robustness-based generalization bounds have gained attention as a principled way to link robustness properties to generalization performance, often in a data-dependent manner. However, most existing bounds suffer from vacuousness in practical settings, yielding loose upper bounds that greatly exceed the actual error rates and limiting their usefulness for real-world evaluation. While this issue is often attributed to the uncertainty term, a substantial part of the problem originates from the robustness term itself, particularly for the 0-1 loss. Existing approaches typically treat the robustness term as a global measure, ignoring its variation across different sub-regions of the input space. In this work, we propose a generalization bound that addresses this limitation by scaling the robustness term according to the number of stable and unstable samples within each sub-region. Our bounds incorporate both data- and model-dependent factors while maintaining practical relevance (yielding tighter upper bounds on true error). Experiments on models trained on the ImageNet dataset show that our bounds remain consistently non-vacuous and achieve the tightest estimates among existing methods, closely aligning with empirical performance across a range of robust deep neural networks. |
| 2026-06-15 | [How Much Can We Trust LLM Search Agents? Measuring Endorsement Vulnerability to Web Content Manipulation](http://arxiv.org/abs/2606.16821v1) | Yimeng Chen, Zhe Ren et al. | Large language model (LLM)-based search agents synthesize open-web content into actionable recommendations on behalf of users, creating a risk that attacker-published pages are transformed into endorsed claims. We introduce SearchGEO, a controlled evaluation framework for measuring endorsement corruption in LLM-based web-search agents, combining a web-evidence manipulation pipeline, a five-mode attack taxonomy, and multiple output-level metrics. We evaluate 13 LLM backends on 308 cases each. Results show that vulnerability patterns vary across backends: overall attack success rate (ASR) ranges from 0.0% on Claude-Sonnet-4.6 to 31.4% on Gemini-3-Flash, the strongest attack mode differs by model family, and the same deployment scaffold could amplify or decrease ASR on different backends. An auxiliary agent-skill probe, where endorsement becomes an install command, exposes a sharp split among otherwise robust backends: Claude over-rejects while GPT over-trusts. These findings argue for treating recommendation reliability under adversarial search content as a first-class dimension of backend safety evaluation. |
| 2026-06-15 | [Adaptive and Explicit safe: Triggering Latent Safety Awareness in Large Reasoning Models](http://arxiv.org/abs/2606.16808v1) | Ke Miao, Jiaxin Li et al. | While Large Reasoning Models (LRMs) excel at complex tasks, they remain highly vulnerable to sophisticated jailbreaks and direct harmful queries. To address this vulnerability, prior works depend heavily on external manual data annotation for safety alignment. However, we observe that LRMs can inherently identify safety risks when being re-presented with original queries alongside their own reasoning trajectories -- a capability we term Latent Safety Awareness. To leverage this safety awareness, we first employ Supervised Fine-Tuning (SFT) to explicitly induce safe tags to trigger safety analysis and guidance following the initial reasoning content for unsafe queries, while preserving standard responses for general queries to ensure adaptive triggering. Subsequently, we apply Direct Preference Optimization (DPO) to further enhance the correctness and stability of the safety analysis and guidance. Notably, responses required for both training stages are entirely generated by models being optimized. With (Safe Trigger) SFT and DPO, experimental results demonstrate significant safety enhancement. For example, the Attack Success Rate (ASR) of DeepSeek-R1-Distill-Llama-8B, on average, drops 24.65% and 36.72% on harmful and jailbreak benchmarks, respectively. Finally, our Safe Trigger method exerts almost no negative impact on general performance or user experience. |
| 2026-06-15 | [LLM-based Visual Code Completion for Aerospace Geometric Design](http://arxiv.org/abs/2606.16806v1) | Hau Kit Yong, Robert Marsh et al. | Recent advances in both Large Language Models (LLMs) and Vision Language Models (VLMs) have seen a step change in their ability to perform visual code completion, but the aerospace industry, which prioritizes safety and explainabilty over rapid LLM adoption, currently has no publicly announced LLM-based geometric design copilot systems in commercial use by aerospace Original Equipment Manufacturers (OEMs). This paper presents a LLM-based visual programming copilot application for aerospace engineering design tasks, using a visual programming variant of the ReAct methodology and GPT 5.4. In addition to the copilot, we describe Wingbuilder, a new Grasshopper plugin library with custom components for aerospace-specific geometry abstraction, and an associated Aerospace Visual Programming Dataset (AVPD) with 18 aerospace expert designed tasks at different levels of difficulty alongside ground truth solutions. We evaluate our copilot application with a user trial involving two experienced aerospace engineers from a large aircraft manufacturing company. We find our copilot visual programming ReAct methodology was successful in generating suggestions that participants found helpful, but slow ReAct inference times limit its usefulness to more complex time-consuming tasks where waiting for good copilot solution suggestion was worthwhile. Participants reported they liked the tool and would be willing to use it in the future. |
| 2026-06-15 | [LabOSBench: Benchmarking Computer Use Agents for Scientific Instrument Control](http://arxiv.org/abs/2606.16802v1) | Anqi Zou, Han Deng et al. | Current computer-use benchmarks primarily focus on software operation tasks in virtualized systems, whereas scientific instrumentation scenarios require coordinated control over complex interfaces, and feedback-driven parameter adjustment. However, directly evaluating agents on physical high-precision instruments is impractical due to high cost, safety risks, limited accessibility, and difficulty in ensuring reproducible evaluation. This motivates the need for a simulated yet realistic testbed that preserves the operational challenges of scientific instruments while enabling scalable and safe benchmarking. To this end, we introduce LabOSBench, a challenging benchmark for multimodal GUI agents built on a suite of web-based scientific-instrument simulators. Operating directly via a browser, LabOSBench avoids resource-heavy OS virtualization while supporting flexible task configuration and execution-based evaluation. Specifically, LabOSBench constructs 96 subtasks across eight instrument simulators, covering workflows from sample loading, alignment, parameter tuning, and data acquisition to result inspection. We evaluate general-purpose vision-language models, specialized GUI agent models, and advanced agentic frameworks at both subtask and end-to-end levels. Our experiments reveal that while existing agents can complete many structured GUI subtasks, they still struggle with feedback-driven operations and long-horizon workflow execution. Overall, LabOSBench provides a reproducible, low-cost testbed for advancing computer-using agents toward scientific-instrument control. |
| 2026-06-15 | [Automated jailbreak attack targeting multiple defense strategies](http://arxiv.org/abs/2606.16751v1) | Qi Wang, Chengcheng Wan et al. | Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. However, their safety remains a critical concern due to their susceptibility to adversarial prompt-based attacks. In this paper, we present UNIATTACK, an adversarial testing framework designed from a defense-oriented perspective to systematically construct effective black-box attack prompts. Unlike prior approaches that rely on static templates or iterative model-specific tuning, UNIATTACK extracts minimal but high-impact attack features from diverse existing attacks, optimizes them via a specialized attacker LLM, and composes them into flexible templates through automated refinement process. This feature-centric construction enables one-shot attacks that generalize across multiple models and safety categories, providing a practical tool for assessing LLM robustness. Our evaluation results shows that compared to the baselines, UNIATTACK achieves an average attack success rate (ASR) improvement of 64.63\%-248.82\% on models deployed with multi-layered defense mechanisms and it only takes 0.03\%-4.96\% cost of the baselines. UNIATTACK artifact is available at https://anonymous.4open.science/r/UniAttack-Artifact-30F1. |
| 2026-06-15 | [User as Code: Executable Memory for Personalized Agents](http://arxiv.org/abs/2606.16707v1) | Bojie Li | A personalized AI agent needs a user memory: a persistent model of who the user is, built across many conversations and consulted on each new one. Today this memory is almost always stored as unstructured text, a knowledge graph, or a flat store of facts, and consulted by retrieval -- fetching the entries most similar to the current request. Such "bag-of-facts" memory recalls individual facts well, but because storing a fact and acting on it are separate steps, it struggles to resolve contradictions, aggregate over many records, or enforce rules. We argue that user memory should instead be executable. We introduce User as Code (UaC), a paradigm in which an agent's model of a user is a living software project: typed Python objects hold the user's state and ordinary Python functions encode the rules that govern it, so representing and reasoning about the user happen in one medium an interpreter can run. The enabling mechanism is a two-phase pipeline: an append-only log that never discards a fact, periodically checkpointed into typed code.   This changes what memory can do. On standard long-term conversation benchmarks, UaC matches both a full-context upper bound and the strongest prior memory systems on recall (78.8% on LOCOMO). Its advantage emerges where representation matters most. On aggregate questions over a user's history -- "how many international trips did I take last year?" -- retrieval-based memory collapses (6-43%) while UaC stays near-perfect (99%), because the answer is a one-line computation over typed state rather than a search over text. And because its rules execute deterministically whenever the state changes, UaC can surface unsolicited, safety-critical alerts -- such as a newly prescribed drug that conflicts with an allergy recorded months earlier -- a capability query-driven memory cannot provide. |
| 2026-06-15 | [Progressive Knowledge-Guided Large Language Model Framework for Bearing Fault Diagnosis](http://arxiv.org/abs/2606.16684v1) | Jinghan Wang, Gaoliang Peng et al. | Vibration-based bearing fault diagnosis requires resolving three interrelated measurement challenges, including the trade-off between global statistical feature efficiency and local transient signal fidelity, insufficient traceability of measurement features to underlying fault physics, and ineffective multi-source measurement information fusion across diagnostic scales. This paper presents a progressive physics-guided multi-scale vibration signal processing framework that addresses all three challenges within a unified diagnostic pipeline. An 81-dimensional measurement descriptor, derived from bearing kinematic theory and characteristic defect frequencies, establishes a physically traceable feature space enabling real-time fault screening at approximately 20 ms per sample. A fault-adaptive signal segmentation mechanism then directs analytical attention toward fault-relevant waveform regions guided by physics-based priors, without manual feature engineering. Structured fault mechanism knowledge is further encoded implicitly in model parameters during training, enabling autonomous multi-scale measurement fusion without external knowledge dependencies at inference. Validated on four public benchmark datasets under diverse operating conditions, the framework achieves 98.49% diagnostic accuracy with a 12.6-fold reduction in computational cost relative to signal-level baselines. Interpretability analysis confirms that diagnostic feature activations align with established bearing fault mechanics, supporting measurement traceability in safety-critical industrial systems. |
| 2026-06-15 | [Closed-loop Optimal Fault Detection for Uncertain Systems](http://arxiv.org/abs/2606.16635v1) | Koen Classens, Tjeerd Ickenroth et al. | Faults compromise the reliability and safety of complex engineering systems. The aim of this article is to address the problem of robust fault detection filter design for continuous-time linear time-invariant uncertain systems in open-loop or closed-loop configurations. The developed method offers a unified approach to handle parametric and dynamic uncertainties by solving a single Riccati equation, based on a worst-case disturbance and uncertainty model. This worst-case model is obtained by nonlinear optimization and application of the boundary Nevanlinna-Pick method. The efficacy of the proposed approach is demonstrated using an uncertain model of an experimental reticle stage used in the lithography industry. The results illustrate that an optimal compromise is achieved between sensitivity to faults and rejection of modelling uncertainties and disturbances on the other hand. This capability enables the clear differentiation between faults and undesired effects in residuals, thereby enhancing fault detection reliability, ultimately contributing to improved safety and performance of machines. |
| 2026-06-15 | [ARB4WM: An Adversarial Robustness Benchmark for World Models in Continuous Control](http://arxiv.org/abs/2606.16605v1) | Junjian Zhang, Hao Tan et al. | World models are widely used in robotic and agentic engineering control systems due to their ability to learn latent dynamics for planning and decision-making. As these systems are increasingly deployed in safety-critical settings, understanding their robustness under adversarial conditions has become essential. However, existing evaluations lack a unified benchmark for testing adversarial threats across the policy, value, and latent-dynamics levels of world-model agents. To fill this gap, we present ARB4WM, a unified evaluation framework for pre-deployment robustness and risk assessment of world-model agents under visual perturbations. ARB4WM defines five white-box loss objectives across these three levels and studies their effects when combined with single-step or multi-step perturbation strategies and temporal attack modes, including full-frame, half-sequence, and sparse-frame exposure. Specifically, we evaluate four Dreamer-style agents across 20 tasks from MetaWorld and the DeepMind Control Suite under different loss objectives, perturbation strategies, and temporal attack modes. Results show that attacks targeting value estimation, latent representations, and RSSM dynamics can be as damaging as direct policy disruption, and that early or frequent perturbations are especially harmful, while input-level defenses provide limited recovery under adaptive attacks. These findings suggest that safety, risk, and reliability assessment for world models should cover multiple component-oriented attack objectives and temporal exposure protocols rather than relying solely on action-space robustness. Source code is available at https://github.com/zaoanguai/ARB4WM. |
| 2026-06-15 | [DRIFT: Risk-Constrained Diffusion with Imitation Priors for Mixed-Autonomy Traffic Generation](http://arxiv.org/abs/2606.16589v1) | Yaoshen Yu, Minghui Liwang et al. | Future intelligent transportation systems are envisioned to evolve toward a long-term mixed-autonomy paradigm, where human-driven vehicles (HVs) and autonomous vehicles (AVs) coexist within highly coupled traffic ecosystems. Such coexistence introduces pronounced heterogeneity, amplified uncertainty, and increasingly intricate interaction dynamics. In this context, it remains fundamentally challenging to simultaneously capture the heterogeneous behavioral distribution shifts arising from dynamic AV penetration, generate diverse yet executable trajectories under strong inter-vehicle coupling, and conduct reliable closed-loop safety and stability diagnostics for rare but high-impact events. To this end, we present risk-constrained diffusion with imitation priors (DRIFT), a mixed-autonomy traffic generation framework which unifies heterogeneity-aware conditional encoding, conditional diffusion-based executable trajectory generation, and progressive adversarial alignment enhanced by risk-aware long-tail feedback, thereby enabling traffic behaviors to be iteratively generated, filtered, selected, and validated within a closed-loop execution pipeline. In addition, a unified evaluation protocol is developed to jointly characterize safety, efficiency, and closed-loop stability across representative traffic scenarios and AV penetration regimes. Experimental results demonstrate that DRIFT achieves a strong safety-efficiency trade-off in closed-loop mixed-autonomy benchmarks, while further revealing the critical influence of candidate executability, online selection, and long-tail feedback on executable traffic evolution. |
| 2026-06-15 | [Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?](http://arxiv.org/abs/2606.16583v1) | Arnisa Fazla, Alberto Testoni et al. | Safe deployment of clinical vision-language models (VLMs) requires reliable uncertainty estimation (UE): a signal indicating when predictions should be trusted or escalated to a clinician. We test whether current UE methods actually deliver this signal. Benchmarking 8 methods across 12 VLMs on clinical visual question-answering (VQA), we find that UE quality is not an intrinsic property of the UE method: it tracks model accuracy, degrading precisely where the model performance is weakest, and therefore where reliability is most needed. When we stress-test models by hiding the correct option among the multiple-choice answers (NOTA perturbations), accuracy collapses while uncertainty barely changes, leaving models systematically miscalibrated. Yet, we find that uncertainty on the unperturbed input reliably anticipates which predictions will collapse under NOTA, indicating that UE in current VLMs carries diagnostic information about model fragility. Our results position UE as a diagnostic tool for identifying fragile predictions and motivate perturbation-based evaluation as a path toward safe clinical deployment. |
| 2026-06-15 | [TNODEV: Toolbox for Neural ODE Verification](http://arxiv.org/abs/2606.16567v1) | Abdelrahman Sayed Sayed, Pierre-Jean Meyer et al. | Neural ordinary differential equations (neural ODE) have started to appear in safety critical settings such as continuous-time controllers for cyber-physical systems and classifiers integrated into automated decision pipelines, raising the question of whether their behavior can be formally verified. Existing tools dedicated to neural ODE provide only a single reachability call without iterative input set refinement, limiting the precision of their verdicts to whatever one reachability call can deliver. We present TNODEV, the first sound formal verifier for neural ODE that integrates a falsification checker, a fast interval-based reachability backend based on continuous-time mixed monotonicity, a verification and refinement loop with three input-set splitting heuristics, and a parallel scheduler in a single end-to-end pipeline. TNODEV supports safe-set inclusion verification on pure neural ODE, neural ODE in closed loop with a neural network controller and general neural ODE (GNODE), with the safe set specified either as an interval or as the half-space intersection induced by a target classification label. We evaluate TNODEV on a range of benchmarks across safe-set inclusion and classification-robustness properties, including a direct reachability comparison against NNV~2.0 and CORA and a verification comparison against NNV2.0 on MNIST general neural ODE classifiers. |
| 2026-06-15 | [A data-driven security quantification framework for IoT-based systems](http://arxiv.org/abs/2606.16561v1) | Alhassan Abdulhamid, Sohag Kabir et al. | The Internet of Things (IoT) is integral to modern cyber-physical systems. Quantitative cybersecurity assessment in IoT environments remains challenging due to heterogeneous system architectures, evolving threat landscapes, and the limited availability of reliable probabilistic exploitability data. Although Attack Tree Analysis (ATA) provides a structured framework for modelling potential attack paths leading to system compromise, conventional ATA quantification often relies on subjective expert judgement or heuristic scoring schemes, which can introduce uncertainty and reduce analytical reproducibility. This study introduces a data-driven probabilistic security framework for IoT-based safety-critical systems by integrating Model-Based Systems Engineering (MBSE), ATA, and empirical vulnerability data. In the proposed framework, SysML models capture system architecture, from which attack trees are derived. Vulnerabilities are mapped as Basic Attack Steps and assigned exploitation probabilities using the Exploit Prediction Scoring System (EPSS). The attack tree is then represented as a Bayesian Network, enabling probabilistic reasoning, diagnostic inference, and vulnerability criticality analysis. The framework quantifies system compromise probabilities, identifies likely causes of attacks, and prioritises mitigation strategies. By combining architecture-driven modelling with real-world vulnerability intelligence, it provides a rigorous, reproducible approach for cybersecurity risk assessment in complex IoT environments. |
| 2026-06-15 | [ROSA-RL: Uncertainty-Aware Roundabout Optimized Speed Advisory with Reinforcement Learning](http://arxiv.org/abs/2606.16558v1) | Anna-Lena Schlamp, Jeremias Gerner et al. | Roundabouts challenge automated driving in mixed traffic, as heterogeneous and non-deterministic human behavior, unknown driving intentions, and high interaction complexity create uncertainty about whether the conflict zone will be blocked or available at the moment of entry. We present ROSA-RL -- uncertainty-aware Roundabout Optimized Speed Advisory with Reinforcement Learning. It enables safe and efficient roundabout entry for automated and human-driven vehicles in mixed traffic through probabilistic conflict forecasting. A Transformer-based model predicts conflict zone occupancy over a five-second horizon, capturing multi-agent interactions to anticipate upcoming conflicts and available gaps. The prediction outputs encode uncertainty in future motion and intent, and augment the state of a classical RL framework, enabling uncertainty-aware speed coordination. Evaluated in simulations grounded in real-world data, ROSA-RL can effectively handle uncertainty and outperform a comparable model-based baseline, closing the gap to an ideal setting assuming fully known occupancy while improving traffic efficiency and safety. The source code of this work is available under: github.com/urbanAIthi/ROSA-RL. |
| 2026-06-15 | [DoubtProbe: Black-Box Jailbreak Defense via Structural Verification and Semantic Auditing](http://arxiv.org/abs/2606.16527v1) | Xuanyu Yin, Yilin Jiang et al. | As large language models (LLMs) are increasingly deployed in user-facing systems, black-box jailbreak defense has become an important practical problem. Existing defenses often rely on known-attack coverage, prompt-level semantic judgment, or local runtime control, yet these paths can become unstable under evolving prompt packaging, expression rewriting, and structure manipulation. We observe that many black-box jailbreaks do not remove the harmful goal, but reorganize the information needed to express and execute it, thereby evading safety alignment while remaining recoverable during generation. Motivated by this observation, we propose DoubtProbe, a dual-branch inference-time defense framework that combines structural verification with semantic auditing and formulates black-box jailbreak defense as consistency checking under controlled transformation. The structural branch extracts a structured representation from the original request, reconstructs the request under representation constraints, and detects information-preservation failures between the original and reconstructed requests; the semantic branch audits the original prompt directly. We evaluate DoubtProbe against representative black-box defenses on jailbreak and benign-request benchmarks, and further test backbone transfer from Qwen2.5-72B to Llama-3.1-70B. Results show that DoubtProbe achieves a stronger and more stable defense-utility trade-off: on Qwen2.5-72B, it reduces the JBB attack success rate from 0.293 to 0.100 and the CodeAttack attack success rate from 0.152 to 0.001, while maintaining false positive rates of 0.022 and 0.016 on AlpacaEval and OR-Bench; the same pattern remains stable on Llama-3.1-70B. These findings show that structural inconsistency signals provide a practical and generalizable basis for black-box jailbreak defense, especially when combined with semantic auditing. |
| 2026-06-15 | [BadWorld: Adversarial Attacks on World Models](http://arxiv.org/abs/2606.16519v1) | Linghui Shen, Mingyue Cui et al. | Visual world models (VWMs) synthesize interactive, action-conditioned rollouts from a single context image. However, it remains an open question how robust these models are to adversarial perturbations. Standard adversarial attacks fail to assess this vulnerability because attackers lack ground-truth future videos and cannot predict subsequent user controls. We introduce BadWorld, a label-free adversarial framework tailored for autoregressive VWMs that systematically overcomes both constraints. First, to bypass the need for future supervision, we propose a self-supervised velocity attack that directly disrupts the early denoising dynamics of the model. Second, to ensure the attack generalizes across unpredictable user actions, we formulate a trajectory-adaptive bi-level optimization that actively mines hard control sequences to forge control-agnostic perturbations. Evaluated on representative VWMs with continuous and discrete controls, BadWorld exposes severe structural fragility. Visually indistinguishable adversarial images reliably trigger catastrophic degradation in future rollouts, leading to incomplete denoising, structural collapse, and control inconsistency. These findings reveal critical risks for deploying VWMs in safety-critical systems while highlighting a practical mechanism for privacy protection. |

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



