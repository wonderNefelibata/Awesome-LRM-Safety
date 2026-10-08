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
| 2026-10-07 | [Agentic RSR: Real-to-Sim-to-Real through Scene Reconstruction and Execution-Grounded Robot Policies](http://arxiv.org/abs/2610.10479v1) | Yihan Li, Yating Feng et al. | A simulation of a real robot workspace must preserve task-relevant interactions, while policies developed in it must operate on observations available to the real robot. Yet scene reconstruction and policy development are often treated separately. We present Agentic Real-to-Sim-to-Real (Agentic RSR), a framework that links scene reconstruction, policy development, and real-robot execution through the same manipulation task. Given a workspace video, a task description, and a known robot model, an agent recovers metric scale, iteratively refines the scene using visual feedback, and checks task-relevant interactions in MuJoCo. A coding agent then develops an executable policy, progressing from privileged object poses to visual observations and randomized simulation. The policy can interleave multiple observations and actions within one invocation, while the agent uses execution feedback to continue, retry, or revise its approach. A shared task-level interface carries the policy and accumulated experience to the real robot, where fresh observations and safety checks guide execution. Across 18 reconstructed scenes involving two robots, the mean four-view Depth MAE against reference depth estimates is 0.1057 m, the mean Lab $ΔE_{76}$ is 11.04, and the mean grayscale SSIM is 0.6990. In real-robot experiments, the aggregate task success rate reaches 80% of the simulation task success rate, indicating substantial retention of simulated performance on hardware. Code and reconstructed scene data will be made publicly available. |
| 2026-10-07 | [Safe Meta-Policy Design with Risk Control](http://arxiv.org/abs/2610.10393v1) | Wenbin Zhou, Michael Lingzhi Li et al. | Models can be retrained as new data arrive, but deploying every new version risks replacing a good policy with a worse one. We study how to plan policy updates (i.e., meta-policy) before future candidates are trained, balancing the benefits of improvement against the risk of performance regression. Our offline meta-policy maximizes expected cumulative value subject to a budget on the expected number of updates that perform worse than the policies they replace. We estimate the value and risk of possible switches from historical learning trajectories, represent an update schedule as a path in a directed acyclic graph, and select a schedule using dynamic programming. A leading-order analysis identifies the signal-to-noise ratio of policy improvement as a key driver of update frequency, waiting times, and risk allocation: clearer improvements support earlier, more frequent updates, while noisier improvements call for longer waits or greater risk expenditure. Their asymptotic rates also reveal a diminishing marginal cost of achieving greater safety over time. Experiments on synthetic and clinical trial data illustrate the performance--risk tradeoff and compare our method with alternative baselines. |
| 2026-10-07 | [Explicit Geometric Chain-of-Thought for Vision-Language-Action in Autonomous Driving](http://arxiv.org/abs/2610.10390v1) | Xingtai Gui, Yucheng Zhou et al. | Vision-language-action~(VLA) models have emerged as a promising paradigm for autonomous driving. However, existing VLA models still suffer from a fundamental mismatch: driving actions require precise 3D geometric cues, while visual-language understanding and reasoning are largely conducted in a 2D semantic space. In this paper, we propose GeoCoTDrive, an explicit geometric chain-of-thought framework that grounds geometry in a planning-oriented manner. GeoCoTDrive follows a think with 2D first, drive with dedicated 3D priors paradigm. It first grounds 2D regions corresponding to decision-critical cues, and then retrieves localized 3D priors by sampling features from a geometric foundation model within the grounded regions. These localized geometric features are interleaved into the autoregressive context to support the trajectory generation. To supervise this process, we introduce planning-relevant grounding, a new region-level grounding task that focuses on local spatial cues directly affecting ego planning decisions, and construct the PlanningGrounding dataset to endow VLAs with planning-oriented grounding capability. Experiments across multiple end-to-end autonomous driving benchmarks show that GeoCoTDrive consistently improves safety-critical planning performance, demonstrating the effectiveness of the explicit geometric chain-of-thought process for VLA-based planning. |
| 2026-10-07 | [Open-MMUnlearning: Unifying Methods and Evaluation for MLLM Unlearning](http://arxiv.org/abs/2610.10358v1) | Junkai Chen, Yuhao He et al. | As multimodal large language models (MLLMs) become more capable and widely deployed, concerns about privacy and safety have become increasingly pressing. Machine unlearning offers one approach to addressing these concerns by removing designated information from trained models while preserving unrelated capabilities. However, fragmented implementations and evaluation protocols, incomplete robustness testing, and limited understanding of metric reliability make progress in MLLM unlearning difficult to assess systematically. We introduce Open-MMUnlearning, an open-source, extensible framework that integrates target-model preparation, multimodal data processing, unlearning, and evaluation through shared interfaces and structured configurations. The framework supports five benchmarks spanning privacy, safety, and copyright, eight MLLMs from four model families, and twelve unlearning methods. Its evaluation suite jointly assesses forgetting effectiveness, retained utility, and robustness to model interventions, adversarial inputs, and membership inference attacks. Using a common evaluation protocol, we compare ten representative unlearning methods. In this comparison, GD and MIP-Editor tie for the highest overall score: GD achieves the highest Forget Quality, while MIP-Editor preserves more Model Utility. We further introduce a metric meta-evaluation protocol that tests faithfulness using models with controlled exposure to target knowledge and robustness under quantization and relearning. Among the thirteen evaluated metrics, BLEU achieves the highest aggregate reliability score. KS-Test attains the highest faithfulness AUC but performs less well on robustness. Together, the framework and these findings support reproducible comparison of MLLM unlearning methods and systematic assessment of evaluation reliability. |
| 2026-10-07 | [The addicted predator-prey model: How opioid use disorder shapes productivity and growth-cycle dynamics](http://arxiv.org/abs/2610.10356v1) | Nara Chung, Marwil Davila Fernandez | This paper extends Goodwin's (1967) predator-prey growth-cycle model to incorporate the negative impact of Opioid Use Disorder (OUD) on labor productivity. Using U.S. state-level data for 1998--2019, we first document that deteriorations in the labor share are followed by rising drug-induced mortality. We then build this link into the model through a novel discrete-choice mechanism in which the probability of OUD decreases with the wage share, so when real wages fail to keep pace with productivity growth, opioid use rises. This behavioral health channel feeds back into the economy through a productivity damage function. Applying the existence part of the Andronov-Hopf bifurcation theorem, we show that the resulting three-dimensional nonlinear system admits a limit cycle, as confirmed by our numerical simulations. Along the cycle, rising OUD paradoxically raises the employment rate, as falling productivity requires more labor per unit of output; across steady states, however, higher OUD prevalence is associated with a lower employment rate, consistent with evidence that opioid use disorder reduces labor force participation. Moreover, our numerical experiments show that OUD sensitivity to income widens fluctuations, while productivity sensitivity to OUD counterintuitively compresses them through a self-correcting feedback that speeds the recovery of the wage share. Policies that weaken the link between income and OUD, such as stronger social safety nets, stabilize the resulting dynamics. |
| 2026-10-07 | [The Handover Problem: Governing Autonomy Transitions in Human-AI Collaboration](http://arxiv.org/abs/2610.10352v1) | Vicente Pelechano, Antoni Mestre et al. | Human-machine systems rarely operate at a fixed level of AI autonomy. As operators and AI systems collaborate over time, control must shift: the AI can take on more responsibility when collaboration is stable, maintain its current role when evidence is ambiguous, or return control to the human when conditions deteriorate. Existing work on adaptive automation, supervisory control, trust in automation, and deskilling explains parts of this problem, but provides no auditable, multi-signal criterion for governing when autonomy should change across multi-cycle workflows.   We formalise this challenge as the Handover Problem: deciding, at each operational cycle, whether to escalate, maintain, or revert AI autonomy while keeping the process reversible, recoverable, and auditable. We introduce the Handover Readiness Score (HRS), a transparent composite measure that integrates four signal dimensions: operator readiness, human-AI trust, learning stability, and operational performance. It is combined with a hysteresis-based transition policy that requires sustained positive evidence before increasing autonomy but reverts promptly when conditions worsen.   Across software engineering and manufacturing domains, the HRS and hard safety guards address complementary failure regimes: guards enforce immediate corrective action when a single indicator breaches a critical threshold, while the HRS detects the slow, multi-signal erosion of operator readiness that no individual guard can observe. The framework establishes autonomy handover as a governance problem requiring explicit, composite, and auditable criteria. This provides a conceptual and formal foundation that adaptive automation research has not previously provided. |
| 2026-10-07 | [SLDR: Defending Against Malicious Fine-tuning via Selective Layers Recovery and Dynamic Routing](http://arxiv.org/abs/2610.10345v1) | Hui Zhang, Yachao Yuan et al. | Fine-tuning-as-a-service enables users to adapt aligned large language models (LLMs) to specialized tasks, but malicious fine-tuning can erode refusal behavior while preserving task performance on legitimate inputs. We revisit recent layer-wise safety diagnostics and find that safety sensitivity is signed: scaling different layers can strengthen refusal, weaken it, or have little effect. Motivated by this observation, we propose SLDR, a post-fine-tuning defense based on Selective Layers Recovery and Dynamic Routing. SLDR trains a LoRA recovery adapter only on the layers with the maximum and minimum sensitivity scores in the signed spectrum, and uses representation-based dynamic routing inference to activate the adapter only for malicious queries. Across four model architectures, five downstream tasks, and four harmful benchmarks, SLDR substantially reduces harmful outputs while preserving downstream utility. On Llama3.1/SST2, SLDR reduces the average harmful score from 11.54 to 0.08 while maintaining downstream accuracy, and the harmful score remains near zero under poisoning ratios up to 0.9. The code is available at https://github.com/Stardust457/SLDR. |
| 2026-10-07 | [Estimating Uncoded Crash Factors with Tabular Foundation and System One Models: Kumo Tabular and Jev](http://arxiv.org/abs/2610.10321v1) | Amir Rafe, Subasish Das | Road safety programs count the coded fields of police crash records, while the officer's narrative, which often records factors the fields omit, is rarely read. A safety office thus cannot tell how much its counts miss or where to review. This study develops and evaluates a system that joins both views of the 5,601,890 Texas crashes from 2017 to 2025 into population estimates with stated validity. An in-context tabular foundation model, Kumo Tabular, reads the coded record of every crash, a calibrated System One model, Jev, reads the narratives of two probability samples, and human judgments recalibrate its probabilities. A multiwave predict-then-debias estimator joins the three tiers, and a second human tier drawn with recorded probabilities checks the estimates by design. For hydroplaning, medical episodes, fatigue, animals, and phone use, the narrative documents more injury crashes than the coded field, 15,074 against 7,340 for phone use, and the human check agrees with all fifteen estimates within its margin. A re-read list ranked by Kumo Tabular finds confirmed discordance 7 to 58 times as often as random reading. At the planning cost of human coding, one further round of human judgments would cut the root mean square relative half-width from 22.0 to 16.2 percent, against 21.2 for reading every narrative. Two calibrated readers of different views, joined by a sampling design, give a safety office counts, a discordance map, a validated re-read list, and a reading budget, with Kumo Tabular reading the table at 15 times the speed of TabPFN 3.5. |
| 2026-10-07 | [AI Safety Considerations for Agents With Limited Time to Act](http://arxiv.org/abs/2610.10285v1) | Leo Zeitler, Jack Richings et al. | In the wake of the increasingly public discussion about AI alignment, recent work has tried to propose specific AI architectures that behave safely. However, the proposed arguments that seemingly demonstrate proved alignment mostly neglect the environment the agent needs to act in. We discuss theoretical bounds for agent-agnostic safety guarantees in environments that can only be partially observed and within which an action is required within limited time. We introduce two realistic scenarios, one with an infinite state space and one with signal mixture. In these scenarios, we prove that even a perfect agent cannot guarantee safe behaviour. It will be argued that for any proof of AI safety or alignment, the environment and associated safe actions need to be specifically considered together with the agent. |
| 2026-10-07 | [PatchBench: Measuring Collateral Damage in Activation Patching](http://arxiv.org/abs/2610.10276v1) | Alexi Canesse, Mathis Le Bail et al. | An LLM safety patch can pass a benchmark while still being a poor repair. This risk is especially acute for jailbreak repairs, where the goal is to correct a specific unsafe behaviour without changing unrelated behaviours. A patch may block exact evaluation prompts yet fail on close harmful variants, or suppress harmful behaviour by over-refusing benign prompts that share its wording or structure. Existing protocols primarily test whether models can be broken, while aggregate metrics (attack success, refusal rates, global capability) cannot distinguish selective repairs from broader local suppression. To address this gap, we introduce PatchBench, a benchmark of empirically observed model-specific jailbreak failures inducing actionable harmful answers. Starting from 27,870 prompts from 37 public datasets, we curate 15,314 English prompts and query 8 open-source instruction-tuned models. Combining WildGuard filtering, pairwise Elo ranking, and manual verification, we retain a curated bank of 400 high-confidence jailbreak failures. We further introduce PatchBench-Local, an evaluation protocol testing whether a patch is behaviourally precise. For each harmful source prompt, PatchBench-Local generates three families of local neighbours: harmful variants preserving malicious intent, benign prompts with matched structure, and benign prompts reusing key harmful terms. It evaluates harmful-neighbour correction and benign-neighbour preservation, distinguishing selective repair from broader local suppression. Evaluating four activation steering methods with PatchBench-Local and MMLU shows that global capability can remain nearly unchanged while local benign regressions are severe, confirming aggregate metrics miss important collateral damage. PatchBench-Local provides a more precise basis for developing and comparing jailbreak repair methods. |
| 2026-10-07 | [Neutral Is Not Free: Evaluating Downside Risk in Neutral Launches](http://arxiv.org/abs/2610.10223v1) | Pablo Alcain, Jason Kang et al. | Evaluating "neutral launches" (e.g., infrastructure upgrades) using traditional confidence interval overlap is flawed: it is dangerously permissive with scarce data and excessively restrictive with abundant data. To resolve this, this paper introduces Expected Bayesian Loss (EBL), a continuous metric that quantifies both the probability and expected severity of metric degradation. Computable directly from standard frequentist estimates, EBL explicitly penalizes empirical noise and high-variance experiments. Validated against expert decisions, EBL provides experimentation platforms with a rigorous, tunable guardrail that aligns statistical safety with institutional risk appetite. |
| 2026-10-07 | [A Probabilistic Perspective on Wasserstein-Based Evidential Uncertainty for Out-of-Distribution Segmentation](http://arxiv.org/abs/2610.10116v1) | Arnold Brosch, Abdelrahman Eldesokey et al. | Semantic segmentation networks operate on a fixed set of classes and therefore fail when out-of-distribution (OOD) objects appear during deployment, a critical limitation for safety-critical applications such as autonomous driving. Reliably identifying OOD objects requires well-calibrated epistemic uncertainty, yet common softmax-based confidence scores remain overconfident, while Bayesian alternatives such as Monte Carlo dropout or deep ensembles require costly repeated forward passes. Evidential Deep Learning (EDL) offers an efficient alternative by modeling class probabilities as a Dirichlet distribution learned from a single deterministic forward pass. Existing EDL formulations rely on Euclidean objectives that push predictions towards the simplex vertices, encouraging overconfidence rather than preserving uncertainty for unfamiliar inputs. We instead employ Wasserstein-based objectives, which respect the geometry of the probability simplex, and study the influence of the Wasserstein order on segmentation accuracy and OOD detection within a unified evidential framework. We evaluate this framework on a convolutional (DeepLabV3+) and a transformer-based (SegFormer) architecture on the SegmentMeIfYouCan benchmark, including LostAndFound, RoadObstacle21, RoadAnomaly21, and Fishyscapes. Our results show the optimal Wasserstein order is architecture-dependent: second-order objectives dominate on the convolutional backbone, third-order objectives on the transformer backbone, and our framework surpasses comparable baselines on most metrics, with a single deterministic forward pass. |
| 2026-10-07 | [Comprehension Audits to Mitigate Risks from Automated AI Research](http://arxiv.org/abs/2610.10064v1) | Ronald J. Bodkin, Bahrad A. Sokhansanj et al. | AI is already writing a majority of code for frontier AI labs. This creates a safety risk if there is insufficient human oversight. Existing work proposes minimum comprehension thresholds and unaided checks to mitigate this. To our knowledge, however, there is currently no published frontier-AI assurance regime that requires demonstrated evidence that the responsible humans understand what they are building as a precommitted condition for continuing development or usage. We propose comprehension audits, a novel development-process assurance mechanism in which the responsible people explain R&D contributions to auditors to demonstrate understanding. With independent administration and graded reports, they provide a gate: development of a contribution stops based on a failure to demonstrate human understanding until remediated, with escalating consequences for repeated failures. Our analysis of leading open-source AI projects finds increased output of code with reduced human review commentary rates per line of code, with far lower rates for automated fleet accounts. We advocate for labs to conduct them with embedded independent auditors. |
| 2026-10-07 | [ReSAFT: An Efficient Stuck-at Fault-Tolerant Scheme for ReRAM-based Process-in-Memory Accelerators](http://arxiv.org/abs/2610.09999v1) | Aniseh Dorostkar, Hamed Farbeh et al. | Analog ReRAM-based process-in-memory (PIM) accelerators provide high parallelism and energy efficiency for deep convolutional neural networks (CNNs) inference. However, their susceptibility to permanent faults, such as stuck-at high (SaH) and stuck-at low (SaL) resistance states, poses a major challenge by permanently corrupting the CNN weights mapped to conductance values of ReRAM cells and degrading inference accuracy, which leads to system unreliability in safety-critical applications. In this paper, we propose a fault-tolerant scheme for analog ReRAM-based PIM accelerators to tackle stuck-at faults (SAFs) with minimal redundancy overhead to recover classification accuracy degradation. The proposed scheme contains a redundancy-based hardware solution alongside fault-aware mapping method for ensuring reliable analog computation in ReRAM crossbar. We analyze the impact of varying number of redundant rows and columns on accuracy and design metrics. Subsequently, a multi-objective optimization (MOO) problem is formulated and solved to efficiently determine the number of redundant rows and columns, considering trade-offs among various design metrics. Furthermore, a fault-aware weight mapping is proposed for dual-crossbar structures to further compensate for the accuracy degradation caused by SAFs. Simulation results show that, for the SimpleNet model using the MNIST dataset, the inference accuracy is recovered by approximately 22.39%, on average, across four configurations of optimal solutions, each offering a trade-off between reliability and area, energy consumption, and latency overheads. The mean-time-tofailure (MTTF) improves by about 61x on average compared to the baseline. These selected configurations also reduce energy and area overheads by 32%, on average, in comparison to row-only and column-only configurations. |
| 2026-10-07 | [From Expected Harmfulness to Likelihood: A Probabilistic Reformulation of Jailbreaking LLM Agents](http://arxiv.org/abs/2610.09973v1) | Juanyang Xu, Zheng Wang et al. | When the harmfulness of an LLM agent's output can be quantified, a natural jailbreaking objective is to maximize expected harmfulness over admissible input modifications. An alternative approach constructs or selects harmful target outputs and modifies the input to increase their likelihood. We establish a precise connection between these two approaches through a probabilistic reformulation. Specifically, we show that the gradient of the logarithm of expected harmfulness with respect to the input equals the expected input gradient of the model's log-likelihood under a harmfulness reweighted output distribution. This identity provides a unified interpretation of expected harmfulness and target likelihood optimization. Building on this connection, we propose OPUR, a sampling distribution designed to generate highly harmful target outputs and use the resulting samples to guide likelihood-based input optimization. Experiments demonstrate the effectiveness of the resulting method in jailbreaking LLM agents. |
| 2026-10-07 | [Purifying Backdoored Large Vision-Language Models by Removing Hijacked Directions](http://arxiv.org/abs/2610.09941v1) | Bojun Yang, Haochen Zhou et al. | Large vision-language models (LVLMs) are increasingly deployed in safety-critical applications, yet they remain vulnerable to backdoor attacks. Defending against such attacks remains costly, as existing methods require either extensive retraining on clean data or per-query intervention at inference time. To address this limitation, we propose OrthoPurify, a more efficient method to purify backdoored model weights via one-step orthogonal projection. Specifically, through structural analysis of backdoor weight updates, we find that the backdoor is encoded by diverting a small number of weight update directions from task adaptation to backdoor shortcut encoding, a phenomenon we term direction hijacking. However, identifying these hijacked directions requires a benign reference model, which is typically inaccessible to the defender. We show that a pseudo-benign model, obtained by fine-tuning the pretrained weights on only a small set of clean samples, provides a sufficient approximation, as the dominant update directions stabilize within the first few gradient steps. OrthoPurify uses this pseudo-benign reference to isolate the hijacked directions and removes them through a single projection on the weight update. Extensive experiments show that OrthoPurify reduces the attack success rate to near zero while preserving the original performance across diverse benchmarks, without retraining the backdoored model or introducing inference-time overhead. Our code is publicly available at https://github.com/womeimingzi/OrthoPurify. |
| 2026-10-07 | [A Scoping Review and Experimental Study on Reinforcement Learning from Human Feedback for Human-Robot Collaboration](http://arxiv.org/abs/2610.09891v1) | Alexandra Coroiu, Andrea Vogt et al. | Human-Robot Collaboration (HRC) can facilitate mass customisation in Industry 4.0, with Reinforcement Learning from Human Feedback (RLHF) representing a promising approach for developing safe AI-based robots. Practical challenges remain regarding safety during AI development, human feedback quality, and bidirectional human-robot adaptation. We conducted a scoping review of RLHF in HRC systems, mapping methods that address these challenges. Following PRISMA guidelines, we screened 199 records and included 20 peer-reviewed publications (2020-2025) spanning multiple HRC domains. To our knowledge, this is the first review focused on the bidirectional, closed-loop design of RLHF. Our review found multiple feedback modalities enabling data collection in various feedback formats. Collected data can be integrated at different stages of AI training, resulting in a multi-step development process. Pilot experiments are commonly used to evaluate HRC systems based on both human and robot metrics. To empirically test a key gap identified in the review, we conducted a between-subjects VR experiment comparing system- and user-initiated feedback on robot proxemic behaviour for safe navigation. Using Bayesian models, we analysed the relation between the collected feedback and safety metrics: psychological safety (post-experiment questionnaire) and physical safety (inverse time-to-collision). Results show that user-initiated feedback captures perceived safety better than system-initiated feedback, indicating that feedback timing directly affects feedback quality. Our review and experiment findings show that RLHF relies on appropriate feedback methods to ensure AI safety in HRC, and future RLHF research should prioritise realistic HRC experiments evaluating the effects of feedback collection methods on relevant human and robot metrics. |
| 2026-10-07 | [Formal Runtime Verification for Tool-Using LLM Agents: An Offline Same-Benchmark Study on AgentDojo and STAC](http://arxiv.org/abs/2610.09793v1) | Nikolaos Kekatos, Stylianos Basagiannis et al. | Guardrails for tool-using LLM agents are usually application-specific rules, which makes multi-step, data-dependent safety policies hard to specify, audit and reuse. As a declarative alternative, we evaluate metric first-order temporal logic (MFOTL), replaying the recorded trajectories that AgentDojo, STAC and R-Judge already ship through the unmodified MonPoly monitor, offline and without running an agent. On these corpora, five generic obligations flag 71.8% of STAC attack chains and 70.1% of successful AgentDojo attacks, but also fire on 29.3% of benign runs. This imprecision stems from the corpora rather than the logic: they rarely record approvals and never record timestamps, so history-dependent obligations reduce to detecting risky action types. Where the trace does carry relational context, provenance-aware policies discriminate better; that context, however, is itself attackable, and one planted line defeats a naive provenance check on 94-99% of the runs it would otherwise flag. Binding provenance to the lookup that produced it closes this evasion at no cost in detection or benign firing. Taken together, these results show that formal temporal monitoring adds value exactly when the trace exposes trustworthy history. We therefore quantify how far current benchmarks are from that point and propose a twelve-field enforcement-ready trace schema. |
| 2026-10-07 | [Beyond Policy Support: Interaction Constrained Offline Reinforcement Learning for Autonomous Driving](http://arxiv.org/abs/2610.09763v1) | Mahmoud Selim, Cristina Cipriani et al. | Offline reinforcement learning enables reward-driven policy improvement from fixed datasets without requiring online exploration, making it particularly attractive in safety-critical domains. A central challenge, however, is distribution shift: policy optimization may favor actions that are weakly supported by the offline data, rendering value estimates unreliable. Existing approaches primarily control this shift in the policy's own action space. In interactive environments such as autonomous driving, this can be insufficient: a candidate ego trajectory may remain well supported under the marginal behavior distribution while being poorly supported jointly with the surrounding-agent behavior observed in the logged interaction. We refer to this degradation in interaction support as \emph{interaction distribution shift} (IDS), and introduce \emph{Interaction-Constrained Drive Policy} (ICDP), an offline reinforcement learning framework that explicitly controls interaction-level distribution shift. Starting from the joint data distribution over ego and surrounding-agent futures, we show that joint-support degradation decomposes exactly into an ego-support component and a residual interaction-support component. We recover the latter through contrastive density-ratio estimation, isolating interaction compatibility without explicit joint-density modeling, surrounding-agent prediction, or rollouts in reactive simulators or learned world models during policy optimization. Closed-loop evaluations on nuPlan, Interplan and real-world truck experiments show that ICDP suppresses high-value yet interaction-unsupported trajectory selections and improves performance in interaction-critical driving scenarios. Project webpage: https://mahmoud-selim.github.io/ICDP/ |
| 2026-10-07 | [Black-Box Adversarial Patch Attacks on VLAs via Ancestor VLM Exploitation](http://arxiv.org/abs/2610.09708v1) | Xiaoyi Pang, Haoyue Feng et al. | Vision-Language-Action models (VLAs) are increasingly deployed in safety-critical physical environments, yet their adversarial robustness remains poorly understood. Existing attacks typically assume white-box access or rely on surrogate VLAs, which rarely holds in real-world deployments. Our key insight is that most VLAs are adapted from a publicly released pretrained vision-language model (VLM), inheriting two capabilities essential for action generation: visual perception and instruction-conditioned grounding. Therefore, this paper explores a previously unaddressed question: can an adversary attack deployed VLAs using only their ancestor VLMs? To this end, we propose three adversarial patch attacks that disrupt the inherited capabilities: a vision disruption attack that corrupts the projected visual tokens through relative and absolute terms, an instruction-grounded semantic evidence suppression attack that removes the visual evidence required for instruction-grounded concepts, and a joint attack that unifies both objectives under a two-phase curriculum. Experiments across different VLA families on both simulation and static real-world images show that patches optimized on the ancestor VLM cause substantial degradations in VLA task success rates, demonstrating that VLAs inherit adversarial vulnerabilities alongside their foundational capabilities. This effect is not uniform: it is strongest on tasks that require precise instruction-grounded localization, and nearly vanishes on policies whose adaptation rewrites the shared visual-semantic representation or whose action head iteratively smooths perturbations away. By characterizing the boundary conditions of vulnerability inheritance and providing analysis of why the inheritance effect holds or fails, we advance the understanding of safety for VLA-involved systems. |

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



