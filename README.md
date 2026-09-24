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
| 2026-09-23 | [Statistical Methods for Estimating Probability of Detection in Structural Health Monitoring](http://arxiv.org/abs/2609.28407v1) | Qizheng Xia, William Q. Meeker et al. | There is much interest in the potential to use structural health monitoring (SHM) technology to augment traditional nondestructive inspection (NDI) methods to improve safety, increase asset availability, and reduce maintenance and inspection costs. SHM has the potential to be used in many applications, including critical components in aircraft and pipelines. Probability of detection (POD) plays a critical role in aircraft structural integrity programs, leading to increased interest in developing methods to assess POD in SHM applications. In contrast to traditional NDI laboratory experiments involving specimens with cracks, SHM sensors are fixed, and SHM data are acquired over time as cracks grow or otherwise evolve. Thus, traditional statistical methods for assessing POD must be replaced or extended to properly handle repeated-measures data. The purpose of this paper is to review the basic statistical concepts of POD and show how these concepts can be extended or adapted for SHM-POD applications. The paper presents statistical methods for modifying and extending existing POD methods, including a simple size-of-damage-at-detection (SoDaD) method and a random-parameter (RP) method for repeated-measures data. The methods are compared using three case studies involving Piezoelectric Transducer (PZT), Carbon Nanotube (CNT), and Comparative Vacuum Monitoring (CVM) sensor systems. Results show that the SoDaD method provides a simple approach for POD estimation with limited data, while the RP method offers enhanced modeling fidelity by utilizing repeated measurements. These methods are applicable when a scalar damage index or similar response is used to make a detection decision. |
| 2026-09-23 | [ForgetMimic: Motion Unlearning for Reinforcement Learning Humanoid Control](http://arxiv.org/abs/2609.28378v1) | Xukun Luan, Zhongxiang Lei et al. | Humanoid control, leveraging human demonstrations, has achieved diverse, agile, and natural locomotion behaviors through reinforcement learning (RL). While this paradigm has yielded remarkable performance in physical humanoid control, how to eliminate specific motions from learned policies remains insufficiently explored. Addressing this issue is motivated by pressing safety and privacy concerns: the removal of malicious, poisoned, or suboptimal motions, as well as copyright-protected motions subject to the right to be forgotten under regulations such as the GDPR, is of critical importance. To this end, we propose {ForgetMimic}, the first motion-level unlearning method designed specifically for physical-world humanoid control. The core idea of ForgetMimic is as follows: given a policy $π_θ$ trained on $N$ motions, our method degrades performance on a target subset of $K$ motions while preserving the effectiveness of the remaining $N-K$ motions. Furthermore, we identify and resolve two key training mechanisms in robot control that lead to unlearning failure. We conduct extensive experiments on the Unitree G1 and H2 humanoid robots across 12 motions, including Dance, Fight, Flip, and others. Experimental results demonstrate that ForgetMimic effectively eliminates memory of designated motions while maintaining the normal operation of all other motions. |
| 2026-09-23 | [LEAP-CBF: A Safety Filter for Uncertain Systems with Least-Effort Adversarial Potentials](http://arxiv.org/abs/2609.28364v1) | Oswin So, Eric Yu et al. | Control barrier functions (CBF) are a popular safety filter to ensure safety for nonlinear dynamical systems. However, when the system is subject to uncertainties and disturbances, this requires the use of robust variants of CBFs, which can be difficult to construct and can be overly conservative, especially for high-dimensional systems under input constraints. In this work, we propose a new approach to solve these challenges by introducing Least-Effort Adversarial Potentials (LEAP), a certificate that quantifies the robustness of a given state against disturbances in terms of the effort required by the disturbance to cause failure. We show that LEAP is a CBF for the undisturbed system, but can also be used to construct a safety filter that is robust to disturbances whose cumulative effort is bounded. We propose a method for constructing LEAPs with on-policy deep reinforcement learning. Next, we demonstrate LEAPs in simulation on a variety of multi-agent systems with disturbances and uncertainties. Finally, hardware experiments on a quadruped and quadrotors validate that LEAPs are well suited to tackle the disturbances and uncertainties from real-world robotic systems. |
| 2026-09-23 | [An Open Pipeline and Dashboard for Systemic-Risk Evidence under the EU AI Act's Code of Practice](http://arxiv.org/abs/2609.28335v1) | Jacob T. Emmerson, Phuong-Anh Nguyen-Le et al. | Claims about AI safety reach audiences well beyond the AI community, yet many rely on opaque evidence or static assessments, when supporting evidence is accessible at all. We present the Systemic Risk Index, an open evaluation pipeline and dashboard built to make empirical evidence more transparent and traceable to the public. Our work organizes 19 public benchmarks into four systemic-risk categories defined by the EU GPAI Code of Practice---CBRN, cyber offense, harmful manipulation, and loss of control---and evaluates models using harm-preserving perturbations and simulated deployment contexts. The interactive dashboard lets users alternate between average and worst-case aggregation, vary how model capability affects the aggregate score, and trace each risk rating to its benchmark evidence. Across 18 models, scores fall by 14 to 37 points under worst-case aggregation, highlighting information that can be hidden by an average assessment of model risk. LLM judges show agreement with human graders comparable to human--human agreement ($κ= 0.78\text{--}0.82$), and a blind audit finds that $83\%$ of sampled transformations preserve the original harm. In a survey ($N = 21$), most participants report that scores are easy to understand and that the dashboard encouraged them to view model evaluations under different settings |
| 2026-09-23 | [BronchoTop: Bronchoscopy Navigation via RGB-Only Topological Localization](http://arxiv.org/abs/2609.28328v1) | Clara Tomasini, Ana Cristina Murillo et al. | Accurate localization of the bronchoscope within the bronchial tree is essential for clinicians to be able to reach target lesions, perform biopsies and avoid misidentification of airway segments during diagnostic and therapeutic procedures. However, existing navigation systems typically rely on patient-specific CT scans or additional external sensors, increasing cost, setup time and patient radiation exposure. This work presents BronchoTop, a real-time, RGB-only framework for topological bronchoscopy localization that eliminates the need for patient-specific data. BronchoTop estimates scope location relative to a generic airway model through four modules: lumen detection and tracking, lumen-branch label association, probabilistic scope location estimation, and switch verification. By using only standard bronchoscopy video input, BronchoTop provides practical, real-time navigational assistance to physicians. Evaluation on phantom, simulated and real data demonstrates state-of-the-art accuracy, improving existing approaches performance by over 20% on real bronchoscopy sequences. BronchoTop is the first published framework including both the localization algorithms as well as all the real data used, together with code to generate additional simulations, encouraging and facilitating further developments and benchmarking. The results highlight BronchoTop's potential to enhance procedural safety, efficiency and accessibility in clinical and robotic bronchoscopy. |
| 2026-09-23 | [Cooperating against Catastrophe](http://arxiv.org/abs/2609.28291v1) | Drew Fudenberg, Andrew Koh | We study a continuous-time game in which two firms choose how quickly to advance their capabilities while an exogenous safety threshold advances at a fixed rate. When the capability frontier (max of firms' capabilities) exceeds the safety threshold, all firms are exposed to common disaster that arrives at a hazard rate increasing in the capability-safety difference. We characterize Markov perfect equilibria in terms of the disaster risk, how quickly safety advances, and flow payoffs: when they are low, only racing is an equilibrium; when they are intermediate, racing and pacing coexist; when they are high, only pacing survives. Across all subgame perfect equilibria, low risk induces perpetual racing while high risk rules it out and we bound the probability of disaster across all SPE. If internal capabilities are hidden with a fixed lag until deployment, it is harder to sustain pacing which highlights the importance of transparency about internal capabilities. |
| 2026-09-23 | [GUIAuditor: Enabling Post-hoc Child Safety Forensics via Action-Guided GUI Provenance on Mobile Devices](http://arxiv.org/abs/2609.28205v1) | Junlin Liu, Yifeng Cai et al. | The proliferation of smart devices exposes children to online risks like grooming and financial scams that are deeply embedded within legitimate applications. Current approaches rely on automated prevention and detection, a paradigm that is fundamentally limited by its inherent fallibility. Whether rule-based or AI-driven, they inevitably produce false positives and negatives, failing to provide reliable protection. In this paper, we argue for a complementary, human-in-the-loop, post-hoc forensic paradigm. We present GUIAuditor, the first system designed to realize this vision by creating GUI Provenance: a queryable, semantic record of a child's interaction sequence. To generate this, GUIAuditor leverages a Multimodal Large Language Model (MLLM) to translate the temporal sequence of GUI events into a human-understandable narrative. To make this practical on mobile devices, a novel evidence distillation pipeline reduces the data requiring analysis by over 89.2% compared to periodic sampling approaches adopted by industry standards, with negligible impact on accuracy. On a new dataset of 295 interaction clips, GUIAuditor achieves a 95.23% Macro-F1 Score in logging significant events and, crucially, its two-stage forensic query engine successfully retrieves the correct evidence as the top result for over 90.20% of natural language questions. An end-to-end evaluation on three modern smartphones shows that the full pipeline, including on-device MLLM inference, adds 2.1W of power draw and 7.4s of per-event latency, with a peak memory footprint of ${\sim}$3.1GB. These results show that post-hoc GUI forensics can run on modern mobile devices and provide useful context for guardian-led safety review. |
| 2026-09-23 | [PASTABench: Proactive Assessment of Sequential Trajectories for Agent Safety](http://arxiv.org/abs/2609.28197v1) | Jiapeng Sun, Yujin Zhou et al. | As Large Language Models (LLMs) evolve into autonomous agents that alter real-world states, ensuring operational safety across multi-step workflows has become a critical challenge. While recent work has moved beyond single-turn evaluation toward multi-turn paradigms, key limitations persist: step-level methods treat actions in isolation, missing how risks accumulate, while trajectory-level evaluations operate post-hoc, offering no opportunity for timely intervention. To address these limitations, we formalize Decoupled Proactive Safety Monitoring along three dimensions: whether to intervene, when to intervene, and what the risk is. We introduce PASTABench, a benchmark of 1,139 multi-turn trajectories spanning 5 risk categories and 13 subcategories. We further propose the Optimal Intervention Window (OIW), anchored by annotated Earliest-Signal and Trigger turns, to quantify intervention timeliness. Evaluation of 16 LLMs reveals that proactive intervention remains largely unsolved, with the best model achieving only 40.74% optimal-timing interventions. Fine-grained diagnosis further uncovers pervasive lexical overfitting: competitive safety scores of smaller models mask keyword hypersensitivity rather than genuine risk comprehension, as their proactive capability largely collapses once hazard vocabulary is neutralized. |
| 2026-09-23 | [Alfvénic high-frequency oscillations at the pedestal of JET plasmas](http://arxiv.org/abs/2609.28193v1) | Leonor Roque, Paulo Rodrigues et al. | Long-lived ($\sim 10$ s) high-frequency oscillations (HFOs, $50-450$ kHz) near the plasma edge have been reported and experimentally described in L-H transition studies at JET and AUG. In this work, we show that these HFOs are a general phenomenon observed in various plasma scenarios and compositions (H$^1$, D, D-He$^3$, D-T) under different heating schemes, including pure Ohmic, ICRH and NBI. We demonstrate that dominant axisymmetric ($n=0$) HFOs are Global Alfvén eigenmodes (GAEs) tied to the shear-Alfvén continuum (SAC) minima arising at the plasma edge due to the sharp decrease of the density and increase of the safety factor. This hypothesis is corroborated by the remarkable agreement found between the measured frequencies of HFOs and the SAC minima computed by the ideal MHD code CSMISH for a comprehensive set of JET pulses. This result, along with the fact that HFOs are observed over long time windows and across a variety of plasma scenarios, makes them convenient MHD constraints for accurate equilibrium reconstruction near the edge. In view of this pragmatic application, we derive a first-order analytic expression for the two lowest coupled branches of the $n=0$ SAC, disclosing their dependence (and, consequently, that of the HFOs) on the plasma density, safety factor, and elongation profiles. In addition, we discuss the nature of $n\neq 0$ HFOs observed along with the dominant $n=0$ HFOs. |
| 2026-09-23 | [Connectivity Preservation and Graph Stretching in Range-Only Swarm Dispersion](http://arxiv.org/abs/2609.28190v1) | Ariel Barel | We study connectivity-preserving finite-jump dispersion of anonymous, identical, and oblivious agents under an idealized range-only sensing model. Each agent measures only the distances to its visible neighbors, without bearings, identifiers, communication, memory, or a shared coordinate system. We derive the largest isotropic displacement certifiable as safe from these measurements alone. The resulting rule requires only the distance to the farthest visible neighbor: each agent selects a random direction and moves by half of its remaining visibility margin. The rule preserves every existing visibility edge under synchronous finite motion and therefore preserves connectivity. For two agents, we prove positive conditional drift in squared distance, almost-sure convergence to the visibility boundary, and finite expected time to reach any fixed neighborhood of that boundary. A one-million-run Monte Carlo experiment agrees with the exact first-round moments and estimates approximately 9.5 rounds to reach distance 0.97V from coincident initial positions; an independent Bellman-equation computation gives the same estimate. For general swarms, 1,000 runs across five initial-topology classes reproduce the deterministic safety guarantee at implementation level and reveal a consistent topology-dependent ordering of attainable diameter under the tested protocol. These results provide a theoretical foundation for connectivity-preserving multi-robot dispersion under minimal sensing, while isolating the guarantees achievable from anonymous range measurements alone. |
| 2026-09-23 | [Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination](http://arxiv.org/abs/2609.28182v1) | Yihong Zhou, Hanbin Yang et al. | Coordinating large population of flexible grid-edge devices can alleviate the need for time-consuming and capital-intensive network upgrades, and AI-based control methods such as multi-agent reinforcement learning or imitation learning are promising in their real-time decision scalability. However, system operators still need an independent and rigorous way to decide whether a given AI system is safe enough for deployment. This paper develops a finite-sample probabilistic safety certification framework for black-box AI decision models in closed-loop grid operation. The central idea is to reduce the complete input--AI--grid evaluator workflow to a binary unsafe outcome under an operator-defined safety specification, and then use exact binomial inference to certify the corresponding unsafe operation probability. Given a set of held-out calibration scenarios, the framework returns the tightest one-sided upper certificate and an accept/reject deployment criterion that controls the probability of false safety certification. Because the certification is for the calibration distribution that may deviate from the future operation, we further combine the nominal certificate with physically interpretable sample-space adversarial attacks, a concept widely used in AI to investigate the fragility of AI models. Case studies on grid-edge flexibility coordination with 1{,}000-agent AI models (independent parameters) verify the finite-sample safety guarantee and the value of integrating adversarial attacks into a rolling-window training-certification-deployment flow. |
| 2026-09-23 | [Safety-Aware Zero Trust Enforcement for IoT and Cyber-Physical Systems](http://arxiv.org/abs/2609.28170v1) | Alessandro Lotto, Alessandro Brighente et al. | Zero Trust (ZT) replaces the implicit trust of perimeter-based security with explicit, continuous, context-aware authorization. This shift is particularly relevant to IoT and cyber-physical systems, whose heterogeneous, long-lived, and remotely connected components make persistent trust untenable. Yet their physical coupling complicates ZT adoption: restricting a suspicious component can reduce cyber exposure while removing telemetry or control capabilities required for operation. Existing work mainly models physical harm caused by attacks, with less attention to consequences introduced by enforcement itself.   We introduce Safety-Aware Zero Trust (SA-ZT), which treats restriction-induced physical consequences as policy inputs. We map the NIST ZT tenets to nine IoT/CPS convergence strains, distinguish IoT-amplified challenges from those specific to cyber-physical coupling, and derive corresponding operational requirements. SA-ZT extends the NIST ZT Architecture with a Safety Engine and a Telemetry Broker. The Safety Engine selects among admissible responses by jointly considering residual cyber risk and restriction-induced consequences, while the Telemetry Broker mediates raw telemetry visibility and estimator influence. With command-side enforcement, these entities separate raw visibility, automated influence, and state-changing authority, preserving observations for monitoring while constraining their influence on automated control. An IEEE 30-bus case study under false-data-injection attack illustrates how SA-ZT makes cyber containment, telemetry visibility and influence, physical consequences, and authorization timing explicit, providing an implementable and inspectable representation of cyber-physical enforcement trade-offs. |
| 2026-09-23 | [Learning from Failures: Heterogeneous Graph Memory for Small Language Model Tool-Using Agents](http://arxiv.org/abs/2609.28003v1) | Jiaxing Li, Lei Song et al. | Small and medium-sized language models offer cost-effective executors for tool-using agents, making them attractive for local and large-scale deployment. However, in long-horizon and stateful environments, they often make structural errors such as missing required observations, performing premature writes, repeating failed calls, and violating action preconditions. These errors can lead to incorrect state updates, policy violations, and costly or irreversible consequences, making reliable tool execution a critical deployment challenge. Existing fine-tuning approaches require substantial data and computation, while flat memory may retrieve failed actions without preserving their causal context or safety conditions. In this paper, we propose FRESH, a Failure-aware Retrieval framework over Experience-Structured Heterogeneous graphs, which transforms historical successes and failures into structured external experience for tool-using agents. By explicitly modeling the dependencies among tasks, actions, errors, repairs, and execution conditions, FRESH helps frozen language models reuse reliable strategies, avoid recurring failures, and make safer decisions in stateful tool interactions. Experiments on $τ$-Bench and AppWorld with multiple open-source models show that FRESH consistently improves task success and tool-use reliability over no-memory agents and representative memory-based baselines. |
| 2026-09-23 | [CasCVS-Net: A Staged Multi-Task Cascade for Critical View of Safety Assessment](http://arxiv.org/abs/2609.27681v1) | Bock-Zien Toh, Yuanchuan Ren et al. | Automated assessment of the Critical View of Safety (CVS) in laparoscopic cholecystectomy requires both recognition of the three CVS criteria and anatomical grounding in small, rare, and often occluded hepatocystic structures. Learning-based methods differ in the anatomical information they use, from image-level classification to detection, segmentation, or graph-based reasoning, yet grounding the safety-critical anatomy remains the main bottleneck. We propose CasCVS-Net, a staged multi-task cascade that jointly performs object detection, semantic segmentation, and CVS assessment, trained on the Endoscapes dataset. The model couples the tasks through predicted anatomy: predicted boxes guide segmentation, and predicted masks provide region-level features for CVS classification, so CVS assessment at inference uses only model predictions rather than ground-truth annotations. To reduce optimisation instability in this coupled setting, training progresses from detection to detection-segmentation and then to the full three-task cascade, followed by task-wise fine-tuning. Evaluation on the public unseen test set shows that CasCVS-Net improves over matched single-task baselines on all three tasks, achieving 32.0 detection mAP, 46.8 semantic mIoU, 15.3 rare-anatomy mIoU, and 67.2 CVS mAP. It outperforms the state-of-the-art LG-CVS and SV2LSTG by 6.3% and 4.5% relative CVS mAP, respectively, corresponding to 4.0 and 2.9 mAP points. These results show that staged task coupling through predicted boxes and masks improves anatomical grounding for CVS assessment, particularly for rare hepatocystic structures. |
| 2026-09-23 | [InGuard: Towards Generalized Inner Guardrail for Safe Text-to-Image Generation](http://arxiv.org/abs/2609.27620v1) | Zeyu Wang, Xiaodan Li et al. | Modern text-to-image (T2I) models generate high-quality images from arbitrary user prompts, yet they can just as easily produce not-safe-for-work (NSFW) content. Conventional outer guardrails consist of two components: a prompt classifier that checks for risk before generation, and a post-hoc image classifier that checks the fully generated image. In this design, both classifiers operate outside the generation pipeline and do not use the model's own representations. This separation can limit prompt-screening accuracy, while the image-side check runs only after the full generation cost has been spent. Moreover, a flagged prompt can only be rejected, even when it could be adjusted to produce a safe image. In this work, we propose the Inner Guardrail (InGuard), a safety framework that works inside the pipeline on the model's own representations, leaving base-model parameters untouched. First, a risk classifier grades each prompt as unsafe, risky, or benign based on the text encoder's embeddings, with no external language model. Second, SAGE (Soft-gated Asymmetric Guardrail for Embeddings) modifies the embeddings of risky prompts, aiming to return a safe image instead of a refusal. Third, a latent detector checks the one-step clean latent estimate midway through denoising, reaching nearly image-level performance and halting generation when risk is detected. We also construct the RevGen Safety Benchmark to evaluate T2I safety under realistic conditions: 10,000 prompts built through real-image reverse generation, with a rewriting step that supplies controlled intellectual-property (IP) characters, covering graded porn/gore risks, categorical IP risks, and benign negatives. Across five open-weight T2I models, InGuard reaches 97.9-98.8% safety rate, matching or exceeding the outer guardrail, with 57.5-73.5% less benign disturbance, ~3.7x fewer parameters, and 50-55.6% of denoising steps skipped. |
| 2026-09-23 | [Control-Token Injection Suppresses Chain-of-Thought and Defeats Reasoning-Based Oversight in Tool-Using Agents](http://arxiv.org/abs/2609.27542v1) | Muhammad Usama, Khair Un Nisa et al. | The safety of a tool-using language model agent is usually treated as a property of the model alone. We give controlled, full-precision evidence that it is instead a joint property of the model and the software that renders its chat template and parses its tool calls, the decoding harness, and that both halves are attackable from untrusted input. On the released gpt-oss-20b reasoning model under its published tool sandbox, appending a single string of the model's own channel-control tokens to a user message makes the tokenizer render a reasoning turn that is already complete, so the model writes no chain-of-thought and proceeds directly to the tool call. Across forty tasks the model already completes, the reasoning channel falls from a mean of 52.5 tokens to zero on every trial while the http.post still fires on every trial. A rule monitor and a cross-family language-model monitor detect the unsafe request on all plain trials and no forged trials, and on overtly malicious requests the attack converts 39.6% of the model's refusals into completed exfiltrations. Separately, whether an identical tool-call generation fires is decided by the harness parser, not the model: a truncation-tolerant regular expression fires a call whose closing token is missing while a strict one drops it, and two parsers shipped for the Gemma agent give opposite outcomes on identical greedy generations, firing on all twenty-four trials and on none. We show the suppression can be delivered indirectly and characterize its dependence on the chat template across two more reasoning models, and we evaluate input sanitization, parser hardening, and empty-reasoning detection as defenses; flagging an absent trace catches the basic attack but not an adaptive benign decoy. All measurements use greedy decoding on publicly released models. Code and per-trial logs: https://github.com/Usama1002/deleting-the-trace |
| 2026-09-23 | [MDRC: A Deployable State-Recovery Defense for Traffic Signal Control under Sensor Corruption](http://arxiv.org/abs/2609.27528v1) | Mingyuan Li, Chunyu Liu et al. | Traffic Signal Control (TSC) is a safety-critical cyber-physical system that relies on real-time sensing. Corrupted observations caused by adversarial perturbations or sensor failures can propagate from the sensing layer into the controller and degrade traffic efficiency. Existing robust Reinforcement Learning (RL)-based TSC methods often suffer from limited cross-city generalization, high inference latency, and weak recovery under partial observability.   We present MDRC (Meta-Diffusion-based framework for Resilient traffic signal Control against adversarial attacks and sensor failures), a post-detection state-recovery defense inserted between sensing and control. MDRC reconstructs trustworthy traffic states before they are consumed by the controller. It combines Denoising Diffusion Implicit Models (DDIM) for efficient state recovery with Reptile meta-learning for a transferable initialization across cities. We provide an optimization-based view of the DDIM recovery dynamics and establish a recovery-error bound that separates score approximation, numerical discretization, and initialization mismatch.   Across seven real-world-derived CityFlow benchmarks, MDRC reduces Average Travel Time by 6.77% under stochastic and policy-aware attacks and by 12.75% under structured sensor loss, while improving state-recovery fidelity. We further evaluate 3,600 seconds of real roadside measurements with 50% of detector channels disabled and integrate MDRC into a hardware-in-the-loop traffic-signal stack. Over a 9.16-hour run with 32,389 sensing/control cycles, the system achieves 99.79% decision availability, produces no out-of-plan recommendations, and requires approximately 38 ms of component-wise processing per one-second control interval. |
| 2026-09-23 | [Safety-Filtered Distributed Koopman-MPC](http://arxiv.org/abs/2609.27463v1) | Shengjun Zhang, Wenhao Li et al. | Distributed model predictive control (DMPC) often constructs both predictions and collision constraints from neighbor trajectories, so packet loss can remove both. We separate these roles: received trajectories drive Koopman-MPC, while local sensing and shelf geometry define a hard-constrained quadratic program (QP) that projects the applied input. Its radial demand is the least constant acceleration that keeps a supporting-plane clearance nonnegative throughout one zero-order-hold interval. Complementary pair rows recover the coupled demand without exchanging safety decisions. We give an intersample separation theorem under bounded snapshot and directional plant errors, an exact max-min test for simultaneous local feasibility, and a sensing-radius condition for switching interaction graphs. Anticipatory high-order rows may be relaxed for performance, but the finite-hold rows contain no safety slack. Matched eight-robot warehouse simulations use a frozen Koopman model, nonlinear drift, bounded inputs and speed, shelf constraints, a 120 ms control period, and packet dropout. The full controller is collision-free in 20/20 matched trials and reaches 160/160 robot goals; predictive Koopman-MPC without the final projection is collision-free in 1/20 trials. All 38,400 full-method hard-row sets pass the online feasibility test, and every local QP solves. Five-stream fleet sweeps are collision-free and hard-row feasible through 16 robots; the 20-robot boundary fails only after the online margin turns negative, while the reconstructed per-agent critical path remains below the sampling period. Bounded-sensing and differential-drive tests provide additional deployment stress. |
| 2026-09-23 | [Latent evolving World Action Model](http://arxiv.org/abs/2609.27455v1) | Xueji Fang, Boqiang Duan et al. | World Action Models (WAMs) jointly model action generation and environment dynamics and are mostly built on pretrained Video Diffusion Models (VDMs). In VDM-based WAMs, observations are first encoded by a VAE, and the resulting compressed latents are then processed by large video diffusion backbones to extract effective features for action generation. However, this paradigm ties WAM performance and training cost to large-scale video generation pretraining, limiting WAM efficiency and scalability. In this paper, we theoretically and empirically investigate how visual representations affect action generation in WAMs. Our results show that predictive embeddings from Joint-Embedding Predictive Architecture (JEPA) encoders better support action generation than compressed VAE latents, with I-JEPA performing best in our encoder comparison. Based on these findings, we propose LeWAM, which conditions action generation on JEPA embeddings and models environment evolution by predicting future embeddings in the same space, without relying on a video diffusion backbone. We further find that imitation learning matches demonstrated actions but does not distinguish better actions from worse ones, even though small action deviations can greatly affect task success. To address this limitation without additional environment interaction or the human oversight required for resets and safety, we introduce Demonstration-Guided DPO (DemoDPO), an offline preference refinement stage that derives preference supervision directly from demonstrations.With only 0.4B trainable parameters, LeWAM achieves an average success rate of 92.28\% on RoboTwin 2.0, comparable to that of state-of-the-art VLAs and WAMs, and maintains practical effectiveness on real-world manipulation tasks. |
| 2026-09-23 | [Shape without scale: an identifiability dichotomy for a bounded tail observed through a non-additive measurement kernel](http://arxiv.org/abs/2609.27384v1) | Jiarui Qi | A latent severity has a bounded lower tail with density of shape alpha and scale L. It is observed only through a fixed Markov kernel K that is biased and non-additive. The relative conditional spread of K diverges at the endpoint. Our sample is i.i.d. from the marginal Q alone, with no anchoring covariate or instrument. We prove a dichotomy. The shape index alpha is identifiable: for every admissible choice of the class constants, any two observationally equivalent members of a lean class share alpha, determined by a near-endpoint expansion of Q. The rate, namely L and the fixed-scale exceedance p_tau, does not survive. There exist admissible shared class constants and two members of a smaller regularity class whose observed laws coincide exactly. Across the pair alpha agrees, whereas L and p_tau move. A degenerate Le Cam two-point bound excludes any uniformly consistent estimator of either, and pointwise consistency fails at one member. Only the rate needs an anchor. We conjecture that a known kernel family with known edge map identifies the rate fiber by fiber if and only if the family satisfies a fixed-scale injectivity clause, and we prove the sufficiency direction. In surrogate safety, uncalibrated conflict data give the shape of near-crash risk, not its absolute rate. |

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



