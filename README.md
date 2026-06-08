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
| 2026-06-05 | [Re-imagining ISO 26262 in the Age of Autonomous Vehicles: Enhancing Controllability through Transferability and Predictability](http://arxiv.org/abs/2606.07437v1) | Chaitanya Shinde, Hadi Hajieghrary et al. | The ISO 26262 standard defines functional safety for road vehicles through risk assessments based on Severity, Exposure, and Controllability, grounded in a human-driven vehicle paradigm. In the context of autonomous vehicles (AVs), the absence of a human driver necessitates revisiting these principles. This paper decomposes the Controllability placeholder into two auditable evidence dimensions of ISO 26262 by introducing two measurable sub-concepts: Transferability and Predictability. Transferability extends Controllability to capture AV systems' ability to hand off control to dedicated fallback safety mechanisms, while Predictability captures how easily external agents can anticipate AV behavior. Predictability is formally defined from human-robot interaction-inspired principles, and a mathematical framework is provided to quantify it. A designed-versus-achievable gap is introduced to distinguish architectural fallback claims from scene-conditioned achievable fallback capability. The proposed metrics align with ISO 26262 and ISO/PAS 21448 (SOTIF), rendering fallback and interaction claims falsifiable and traceable across ODD slices. These dimensions complement rather than replace existing standards, and the enhancements preserve the structure of ISO 26262 while extending its applicability to driverless automated systems operating at SAE Levels 4 and 5. |
| 2026-06-05 | [A Comprehensive Anatomy of Human and DeepSeek-R1 LLM Mathematical Reasoning](http://arxiv.org/abs/2606.07410v1) | Yuxiang Chen, Jun Wang | The emergence of "Aha moments" in large language models, particularly DeepSeek-R1-0120, has raised the question of whether these systems genuinely reason or merely imitate the appearance of reasoning. We conduct a comprehensive empirical comparison between model and human reasoning across all 30 problems from AIME 2025, exhaustively annotating 10,247 reasoning steps into five functional categories: Analysis, Inference, Branch, Backtrace, and Reflection. We find a clear structural difference. Human solutions maintain a compact alternation between analysis and deduction, whereas DeepSeek-R1 frequently revisits intermediate results, performs shallow and often unnecessary verification, and loops through local checks without meaningful logical progress. We describe this as topological mimicry: reproducing the surface form of reasoning without its functional role. Despite this, we identify two signals of genuine reasoning. First, successful traces exhibit stable use of branching and backtracking, while failed traces either underuse or overuse exploratory actions. Second, reflection is only effective when placed within deductive inference; reflections trapped in analysis loops focus on local numerical details while missing global logical errors. These findings suggest that current long-CoT models may be rewarded more for the appearance of reasoning than for genuine deductive progress. We discuss directions for improving evaluation and training, including measuring cross-trace stability, penalising "spinning-wheel" traces, encouraging deeper logical correction, and reallocating inference-time compute toward deduction and backtracking. Overall, reasoning quality depends not simply on how much reflection occurs, but on whether reflection appears consistently and at the appropriate logical scale. |
| 2026-06-05 | [LLM-Guided Evolution for Medical Decision Pipelines](http://arxiv.org/abs/2606.07342v1) | Ivan Sviridov, Artem Oskin et al. | Adapting large language models (LLMs) to clinical workflows often requires costly fine-tuning or manual prompt and pipeline engineering. We study LLM-guided MAP-Elites evolution as an inference-time alternative for discovering medical decision strategies and provide an implementation repository at https://github.com/univanxx/llm_guided_evo_medical. We formulate urgency triage, interactive consultation, and medical image classification as evolutionary searches over executable artifacts optimized by task-specific fitness functions.   Across all three settings, evolution improves over manually designed baselines under practical constraints. In triage, evolved programs increase Semigran accuracy from $77.3\%$ to $87.1\%$ and emergency recall from $0.60$ to $0.97$, while improving safety-weighted held-out MIMIC-ESI performance. In interactive consultation, evolved policies improve the accuracy--cost frontier across Llama-3, Qwen-3.5, and Gemma-4 and transfer to held-out iCRAFTMD. In PneumoniaMNIST, prompt-only evolution improves frozen MedGemma VLMs while preserving strict JSON outputs. Qualitative analysis shows that the gains come from interpretable program-level mechanisms, calibrated triage boundaries, targeted evidence acquisition, selective commitment, and finding-oriented visual decision rules, rather than superficial prompt rewording alone. |
| 2026-06-05 | [Defending Jailbreak Attacks on Large Language Models via Manifold Trajectory Kinetics](http://arxiv.org/abs/2606.07335v1) | Hangtao Zhang, Yucheng Zhao et al. | Jailbreak prompts can bypass alignment guardrails in large language models (LLMs) and elicit unsafe outputs, making reliable deployment-time detection critical. Prior detection approaches largely rely on a fixed metric space, e.g., raw inputs, gradients, or hidden features, in which benign and jailbreak prompts are linearly separable. We show this assumption breaks under (i) pseudo-malicious prompts that are benign by intent but contain safety-related keywords, and (ii) adaptive attacks that explicitly optimize against the deployed detector. To overcome this limitation, we shift our focus from identifying a universal metric space to analyzing the more robust neighborhood structure of the underlying data manifold. We present Manifold Trajectory Kinetics (MTK), which treats an LLM as a kinetic system transforming inputs into outputs and detects jailbreaks by tracking how a prompt's neighborhood structure evolves across layers. Benign prompts remain close to benign neighborhoods throughout inference, whereas jailbreak prompts exhibit a characteristic trajectory that begins near malicious seeds and later strategically shifts toward benign neighborhoods to evade refusal.Across four LLMs and ten jailbreak attacks, MTK achieves strong robustness to both failure modes: on pseudo-malicious prompts, it attains a jailbreak true positive rate of 95% at a false positive rate of 5% on benign prompts and 2% on pseudo-malicious prompts, and under adaptive attacks, it maintains a true positive rate of 85%. We further demonstrate the superior performance of MTK for jailbreak detection in vision-language models. Our code is available at https://github.com/Rookie143/mtk. |
| 2026-06-05 | [Hierarchical Certified Semantic Commitment for Byzantine-Resilient LLM-Agent Collaboration](http://arxiv.org/abs/2606.07316v1) | Haoran Xu, Lei Zhang et al. | Byzantine collaboration among large-language-model agents requires a finality-control primitive: given delivered stochastic, structured natural-language proposals, the protocol must decide whether the round supports a commit, what kind of commit, or a typed safe abort. Naive aggregation hides this choice behind a single verdict; classical Byzantine fault tolerance hides it behind byte-identity that LLM proposals do not satisfy. We introduce Hierarchical Certified Semantic Commitment (H-CSC), a BFT-inspired protocol that converts embedding-derived finality signals over verdict-conditioned proposal groups into one of three typed outcomes: a semantic_commit (a 2f+1 within-verdict semantic core backs the verdict, emitting a parameter-bound digest over the quantised aggregate), a verdict_commit (strong verdict margin but dispersed semantic rationale, emitting a verdict-level certificate without claiming a semantic aggregate), or an explicit abort with a typed reason. The contribution is typed finality, not raw commit accuracy. On a controlled semantic-poisoning diagnostic (BCS_v1, 120 episodes), H-CSC commits with low angular deviation on BFT-feasible buckets (0.31 to 2.04 degrees) and aborts 100% of beyond-BFT rounds (n<3f+1) as intended. On a real LLM-agent claim-verification benchmark (MVR-50, 50 tasks) under paired static and rushing Byzantine attacks, H-CSC commits 0.90/0.92 with honest-reference-invalid rates of 0.02/0.00, statistically matching a strong certificate-emitting verdict-only baseline. Unlike that baseline, H-CSC also emits an embedding-backed semantic_commit digest on 74%/72% of rounds, supplying typed provenance. A strict-semantic ablation commits only 0.54/0.48, showing the verdict-level fallback is necessary for coverage (+0.36/+0.44) at the same <=0.04 safety floor; a 100-task cross-model check across four LLMs preserves invalid_hmaj within 0.00 to 0.03. |
| 2026-06-05 | [When Large Language Models Fail in Healthcare: Evaluating Sensitivity to Prompt Variations](http://arxiv.org/abs/2606.07237v1) | Mahdi Alkaeed | Large Language Models (LLMs) are increasingly used in healthcare for tasks such as clinical question answering, diagnosis support, and report summarization. Despite their promise, these models remain highly sensitive to subtle prompt perturbations, both lexical and syntactic, posing serious risks in safety-critical clinical applications. In this study, we conduct a systematic sensitivity analysis to evaluate the robustness of both general-purpose (e.g., GPT-3.5, Llama3) and medical-specific LLMs (e.g., ClinicalBERT, BioLlama3, BioBERT) using the MedMCQA benchmark. We categorize perturbations into natural and adversarial types and examine their effect on model consistency, accuracy, and reliability in clinical reasoning tasks. Our findings reveal that medical LLMs are not intrinsically safe. Even minor variations in phrasing can alter clinical advice, and targeted adversarial prompts can provoke harmful outputs. In high-stakes settings like healthcare, such unpredictability is unacceptable-models that change diagnoses due to reworded inputs or hallucinate medications when slightly rephrased cannot be reliably trusted by clinicians. While models tend to show resilience to simple lexical substitutions or paraphrasing, they often break down under syntactic reordering or misleading contextual cues. This fragility is evident across both general-purpose and domain-specific LLMs. Notably, adversarial manipulations can lead to clinically dangerous outputs, such as recommending incorrect dosages or omitting critical findings. |
| 2026-06-05 | [An Abstract Architecture for Explainable Autonomy in Hazardous Environments](http://arxiv.org/abs/2606.07211v1) | Matt Luckcuck, Hazel M Taylor et al. | Autonomous robotic systems are being proposed for use in hazardous environments, often to reduce the risks to human workers. In the immediate future, it is likely that human workers will continue to use and direct these autonomous robots, much like other computerised tools but with more sophisticated decision-making. Therefore, one important area on which to focus engineering effort is ensuring that these users trust the system. Recent literature suggests that explainability is closely related to how trustworthy a system is. Like safety and security properties, explainability should be designed into a system, instead of being added afterwards. This paper presents an abstract architecture that supports an autonomous system explaining its behaviour (explainable autonomy), providing a design template for implementing explainable autonomous systems. We present a worked example of how our architecture could be applied in the civil nuclear industry, where both workers and regulators need to trust the system's decision-making capabilities. |
| 2026-06-05 | [Shield-Loco: Shielding Locomotion Policies with Predictive Safety Filtering](http://arxiv.org/abs/2606.07193v1) | Aditya Shirwatkar, Sebastian Sanokowski et al. | Reinforcement learning (RL) policies enable dynamic legged locomotion but lack mechanisms to avoid violations of safety constraints that are absent during training. Large-scale offline safe learning is impractical for covering all edge cases. Existing safety frameworks either rely on reduced-order models that cannot reason about whole-body behaviors or require conservative recovery controllers that degrade task performance. We propose a predictive safety filter that post-hoc filters the nominal contact locations fed to the RL policy. When a collision is predicted, a sampling-based optimizer asynchronously searches for safer contact sequences using a full-physics model, while a learned value function bootstraps long-horizon returns. Our three algorithmic components (geometric projection of sampled contacts, momentum-augmented updates, and replica-exchange) make the optimization tractable in a discontinuous contact landscape. We validate the filter on a quadruped robot in dense, cluttered environments, both in simulation and in the real world, showing substantial reductions in safety violations with minimal deviation from the nominal input. |
| 2026-06-05 | [RISE: A Rust Library for Inverted Index Search Engines](http://arxiv.org/abs/2606.07187v1) | Angelo Savino, Rossano Venturini | Inverted indexes are a crucial data structure for efficient information retrieval in large text corpora. They enable fast full-text search by mapping each term to the documents in which it appears, on top of which efficient algorithms quickly retrieve the documents relevant to a user query. We present RISE, a novel inverted index library implemented in Rust, designed to deliver high performance and efficiency for information retrieval tasks. RISE leverages Rust's safety and performance to provide a robust solution for building and querying inverted indexes, while offering accessible extensibility through its expressive trait system. While developing RISE, we revisited the inverted-index literature, thereby reproducing numerous prior works using this new test bench. We evaluated RISE against existing libraries, demonstrating competitive query performance across various datasets and workloads, with speedups of up to 2x over the current state of the art. Our results indicate that RISE is a promising tool for researchers and practitioners in the field of information retrieval. |
| 2026-06-05 | [A Causal Probabilistic Framework for Perception-Informed Closed-Loop Simulation of Autonomous Driving](http://arxiv.org/abs/2606.07186v1) | Zhennan Fei, Rickard Johansson et al. | Software-in-the-loop (SIL) simulation is a cornerstone for the validation of modern automotive safety functions. However, many current frameworks utilize ideal sensing, which bypasses the functional insufficiencies of perception algorithms, leading to over-optimistic safety assessments. This paper proposes a perception-informed SIL testing methodology that bridges the gap between ground-truth simulation and real-world perception behavior. We present a framework for incorporating causal probabilistic models into standardized, scenario-based simulation toolchains, applicable to both Advanced Driver Assistance Systems (ADAS) and Autonomous Driving Systems (ADS). Our approach enables the systematic injection of realistic perception errors, such as loss of detection, sizing inaccuracies, and positioning offsets, derived from physical triggering conditions like fog, rain, and object-merging scenarios. By evaluating these ``faults'' within a standardized simulation environment, we demonstrate that perception-informed testing reveals latent operational risks that ideal SIL environments fail to capture, providing a scalable pathway for SOTIF (ISO 21448) validation. |
| 2026-06-05 | [Think Fast: Estimating No-CoT Task-Completion Time Horizons of Frontier AI Models](http://arxiv.org/abs/2606.07157v1) | Dewi Gould, Francis Rhys Ward et al. | Many efforts to ensure frontier AI models are safe rely on monitoring their chain-of-thought (CoT) reasoning. If models become able to perform sufficiently complex reasoning internally, without explicit thinking tokens, this would undermine such oversight. We measure how well frontier models reason without CoT across a suite of over 30,000 questions spanning 43 benchmarks in domains including math, coding, puzzles, causality, theory-of-mind, and strategic reasoning. To compare models against humans, we estimate the $50\%$-task-completion time horizon (TH): the human time required for tasks a model completes with $50\%$ success rate. We complement this with a $50\%$ reasoning token horizon: the minimum number of o3-mini reasoning tokens needed for tasks a model solves with $50\%$ success rate. We find that the no-CoT $50\%$ TH of frontier models has been doubling roughly every year over the past six years, with GPT-5.5's TH reaching over 3 minutes and reasoning token horizon exceeding 1,500 tokens. Our median estimates predict that frontier no-CoT THs could exceed 7 minutes by 2028, and 25 minutes by 2030, though these projections carry substantial uncertainty. We recommend frontier developers track this explicitly. |
| 2026-06-05 | [REMEDI: A Benchmark for Retention and Unlearning Evaluation in Multi-label Clinical Disease Inference](http://arxiv.org/abs/2606.07141v1) | Anurag Sharma, Sai Teja Chunchu et al. | Language models trained for clinical disease inference are trained on patient data, which may include sensitive and private information, and data owners may request the removal of their data from a trained model due to privacy or copyright concerns. However, exactly unlearning patient-specific data is intractable, and retraining with minor data removal is resource-intensive. While there exists several machine unlearning methods that can be used, their utility is generally restricted to non-medical domains. Moreover, the existing benchmarks for evaluating such unlearning methods primarily utilize synthetically curated datasets, which are not truly representative of real-world systems. Hence, the effectiveness of these unlearning methods in the medical domain is largely unclear. To this end, we introduce REMEDI, an extensive benchmark for machine unlearning tailored to multi-label and multiclass clinical disease inference, where label correlations, longitudinal structure, and safety constraints make unlearning particularly challenging. Unlike the existing benchmarks, REMEDI considers: (1) a relevant application domain (medical), (2) comprehensive unlearning setups involving diverse sets of forget instances, (3) challenging unlearning scenarios including multi-label and multi-class classification tasks, and (4) evaluation metrics involving performance both in terms of utility and extent of unlearning achieved. REMEDI is developed using the MIMIC-III clinical database that contains comprehensive clinical data of patients. Experiments with existing unlearning methods indicate that there exists a trade-off between utility and unlearning performance. They are also largely unsuited to multi-label classification tasks. To facilitate reproducibility, we make our benchmark publicly available. |
| 2026-06-05 | [Residual-Controlled Multiplier Learning for Stochastic Constrained Decision-Making](http://arxiv.org/abs/2606.07088v1) | Kang Liu, Jianchen Hu et al. | Stochastic constrained decision-making requires optimizing performance objectives while enforcing statistical requirements such as safety or fairness. However, standard primal--dual methods struggle to update multipliers robustly under stochastic mini-batch feedback, as the noise of mini-batch gradients and constraint estimates can be directly accumulated into the multiplier memory. To address this issue, we propose Residual-Controlled Multiplier Learning (RCML), which reformulates multiplier updating as projected-pressure feedback. The central idea is to decompose the projected multiplier into an effective pressure signal for primal descent and a pressure-memory residual for finite-gain multiplier tracking. To handle heterogeneous and noisy observations, we further augment this residual-integral backbone with modular stochastic stabilization components. For the convex-affine backbone, we establish finite-gain convergence, derive a stochastic residual bound under mini-batch feedback, and show that the residual feedback law admits a local KKT-residual interpretation near regular KKT points of nonconvex problems. Experiments across optimization, allocation, and fair-ranking tasks show that RCML improves feasibility control and multiplier stability while maintaining competitive objective performance. Code is available here. |
| 2026-06-05 | [Extending Responsibility-Sensitive Safety for the Assessment of Offloaded Autonomous Driving Services](http://arxiv.org/abs/2606.07067v1) | Robin Dehler, Aryan Thakur et al. | Safety is a fundamental requirement in the development of autonomous driving (AD) systems. While function offloading has demonstrated significant benefits in terms of computational efficiency and energy consumption, its application to safety-critical AD functionality introduces new challenges. In particular, offloaded service compositions incur increased and variable response times due to wireless vehicle-to-everything (V2X) communication, which directly affects the vehicle's reaction time and thus its safety guarantees. In this paper, we address this challenge by extending the definitions of Responsibility-Sensitive Safety (RSS) to explicitly account for different response times of local and offloaded AD service compositions. Based on this extension, we propose an integration into function offloading, using the RSS safety constraints for offloading decision-making and fallback mechanisms. Offloaded service compositions are only permitted if the current traffic situation remains safe under the corresponding end-to-end response time. If this condition is violated, the system performs a controlled fallback to local execution. Furthermore, we introduce an enhanced fallback strategy that includes a warm-standby phase for offloaded services, enabling faster and safer transitions from offloaded to local services. The proposed approach is integrated into our AD stack and evaluated in both simulation and the real world. Experimental results demonstrate that the proposed method improves safety compared to state-of-the-art function offloading and safety frameworks, while preserving the benefits of distributed computation when safety conditions allow. |
| 2026-06-05 | [An Integrated Roadside Sensing and Communication Framework for Vulnerable Road User Safety at Signalized Intersections](http://arxiv.org/abs/2606.07016v1) | Parvez Anowar | Vulnerable road users (VRUs) account for approximately half of urban traffic deaths globally, with intersections concentrating a disproportionate share of these casualties. Recent reviews of sensing technology for VRU protection have cataloged dozens of single-sensor and dual-sensor deployments, yet none of the surveyed systems couples multi-modal sensing with edge-side near-miss analytics and bidirectional vehicle-to-everything (V2X) and pedestrian-to-everything (P2X) messaging in a single intersection cabinet. This paper presents an integrated framework for VRU protection at signalized intersections, combining LiDAR, radar, RGB camera, and thermal camera at the perception layer, edge-based prediction and surrogate-safety analytics at the computation layer, V2X and P2X messaging at the communication layer, and adaptive signal control at the actuation layer. The framework is grounded in an empirical case study using R-LiViT, the first publicly released roadside LiDAR-Visual-Thermal dataset, which provides 200 multi-modal sequences and 2,400 annotated RGB-T frames at three German intersections. Analysis of 53,319 detection annotations reveals that VRUs comprise approximately 49% of all road-user observations, that day-to-night density drops by 38% for pedestrians and 45% for vehicles while the night distribution shows a higher close-proximity share, that per-frame close-proximity event counts vary approximately 10-fold across the eight unique locations at three intersections, and that 83% of pedestrian bounding boxes are small in image space, indicating that VRUs are typically far from any single sensor. These findings support multi-modal sensing, edge-side analytics, and adaptive context-sensitive deployment rather than uniform single-sensor solutions. |
| 2026-06-05 | [Mission-Level Runtime Assurance Framework for Autonomous Driving](http://arxiv.org/abs/2606.06996v1) | Chieh Tsai, Salim Hariri | This paper studies runtime safety for autonomous driving when high-level driving commands become faulty or unreliable. Unlike conventional runtime-safety approaches that mainly focus on immediate vehicle safety, the proposed framework evaluates both driving safety and whether the vehicle can still successfully complete its mission before a command is executed. The framework extends highway-env with mission-level fault scenarios such as skipping required checkpoints, entering restricted areas, and generating future routes that can no longer complete the mission successfully. A runtime monitoring system is introduced to detect and reject unsafe or mission-infeasible commands before execution. For comparison, an adapted Simplex-Drive runtime-safety baseline with learning-based driving control, safety fallback control, and runtime controller switching is implemented using the public Simplex-Drive framework. Experimental results show that platform-level runtime safety alone cannot detect mission-level planning faults, while the proposed framework successfully rejects mission-infeasible commands and improves mission success under randomized fault conditions. |
| 2026-06-05 | [HAVE: Host Active Verification Engine for Closing the Contextual Reality Gap in Security Digital Twins](http://arxiv.org/abs/2606.06968v1) | Vincenzo Sammartino, Marco Pasquini | Security Digital Twins (SDTs) provide continuously updated virtual replicas of infrastructure for threat simulation, yet they rely on theoretical CVSS scores to assign lateral-movement probabilities -- creating the Contextual Reality Gap: risk is overestimated where unacknowledged mitigations neutralize exploits, and drastically underestimated where logic flaws bypass all memory-safety defenses. We present the Host Active Verification Engine (HAVE), an SDT extension that deploys a safety-constrained host agent to measure the empirical probability of compromise $\hat{p}$ via maximum-likelihood estimation over snapshot-isolated Bernoulli trials. A Wilson interval-width confidence weight $α_w$ propagates $\hat{p}$ into Monte Carlo simulations via a Bayesian blending rule formally related to the Beta-Binomial posterior. Evaluation across four vulnerability classes, three security tiers, and two production binaries shows HAVE reduces $P_{\text{reach}}$ by 38.2% in false-positive scenarios and increases it by 132.4% in false-negative scenarios, with a net +124.1% correction; post-HAVE estimates vary by only $1.12\times$ across calibration exponents $κ$, versus $4.6\times$ for CVSS-only baselines. |
| 2026-06-05 | [Workflow-to-Skill: Skill Creation via Routing-Workflow-Semantics-Attachments Decomposition](http://arxiv.org/abs/2606.06893v1) | Yuyang Zhang, Xinyuan Han et al. | Large language model agents increasingly rely on Skills to encode procedural knowledge, yet high-quality Skills remain costly to hand-write. This paper studies automatic Skill construction from heterogeneous interaction evidence, including demonstrations, agent trajectories, tool traces, and execution logs. We argue that trace-to-skill construction is not simple summarization tasks, because traces are fragmented, redundant, and may miss rare but safety-critical behaviors. To address this, we introduce RWSA, a workflow-oriented intermediate representation that decomposes Skills into Workflow structure, execution Semantics, and runtime Attachments, capturing task decomposition, control flow, verification, safety, rollback, and state management. Building on RWSA, we propose W2S, a framework that segments traces, induces local Skill drafts, aligns shared structures, reconciles branches, and compresses redundancy while preserving evidence and confidence annotations. Experiments on 70 Skills show that W2S improves behavioral replay consistency by 10.5% over summarization- and prompting-based baselines, highlighting the need to treat traces as executable runtime specifications rather than compressible text. |
| 2026-06-05 | [Unified Safe In-context Image Generation in Multimodal Diffusion Transformers via Restricting Unsafe Information Flows](http://arxiv.org/abs/2606.06875v1) | Xiang Yang, Feifei Li et al. | Diffusion transformers (DiTs) equipped with multimodal attention (MM-Attn) have become a dominant paradigm for image generation. However, preventing the generation of harmful content remains a critical challenge, particularly in image-to-image (I2I) editing tasks. Existing safety mechanisms are primarily designed for text-to-image (T2I) synthesis or U-Net-based architectures, which limits their effectiveness for unified safety mitigation in DiT-based frameworks. To bridge this gap, we propose Unified Visual Safety Regulator (UVR), a training-free safe generation framework that regulates unsafe semantics in generated images. UVR is grounded in an analysis of attention dynamics from the perspective of information flow in MM-Attn. We identify a task-independent start-up stage, during which unsafe semantics in output patches rapidly emerge and can be accurately localized, followed by task-specific semantic amplification and interference stages, where harmful signals are further propagated and entangled with benign content. Based on these observations, UVR mitigates unsafe generation through unified, targeted attention modulation and explicit restriction of harmful information flow over the identified unsafe output patches. Experiments across various concepts show that UVR achieves state-of-the-art safety performance by achieving 91% and 77% erase rate in image synthesis and editing tasks, while preserving visual quality and fidelity with minimal degradation. Code is available at https://github.com/deng12yx/UVR. |
| 2026-06-05 | [VideoSEG-O3: A Multi-turn Reinforcement Learning Framework for Reasoning Video Object Segmentation](http://arxiv.org/abs/2606.06819v1) | Ming Dai, Sen Yang et al. | Reasoning Video Object Segmentation (RVOS) demands a sophisticated integration of temporal dynamics, spatial details, and linguistic reasoning to achieve precise pixel-level localization. Existing methods are limited to reasoning over fixed initial inputs and lack the capacity to actively acquire further visual evidence, which is often essential for resolving complex references in long or intricate videos. To address this, we propose \textbf{VideoSEG-O3}, the first multi-turn reinforcement learning framework for RVOS that emulates the human \textit{``coarse-to-fine''} cognitive process. It employs a \textit{multi-turn temporal-spatial chain-of-thought} to capture fine-grained details by iteratively pinpointing critical intervals and keyframes. Additionally, to enable the policy to perceive segmentation quality beyond mere text probability of \texttt{[SEG]} during the RL stage, we introduce \textit{SEG-aware logit calibration}, which integrates pixel-wise segmentation feedback directly into the token-level logits. Furthermore, we design a \textit{decoupled thinking trace} to hierarchically decompose the reasoning process into temporal, spatial, and linguistic dimensions, and construct \textbf{VTS-CoT}, a specialized cold-start dataset featuring comprehensive reasoning trajectories. The code and models will be released at https://github.com/Dmmm1997/VideoSEG-O3. |

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



