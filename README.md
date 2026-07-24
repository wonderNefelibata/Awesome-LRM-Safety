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
| 2026-07-23 | [Seeking Help in the Digital Age: A Cross-Platform Analysis of Online Support Systems for Technology-Facilitated Abuse Victims](http://arxiv.org/abs/2607.21549v1) | Nowshin Tabassum, Solomon G. Dandekar et al. | Technology-facilitated abuse (TFA), the use of digital technologies to stalk, harass, monitor or threaten others, has become a pervasive form of interpersonal harm. As victims turn to online sources for guidance, responses can shape how they assess risks, interpret abuse, and choose protective actions. We present a large-scale evaluation of online support for TFA victims across three channels: web search, peer-support forums, and conversational AI systems. Drawing on a decade of victim narratives from r/Stalking, we use qualitative coding and supervised classifiers to construct a dataset of TFA queries spanning 11 categories of technology misuse. We simulate these queries across the three channels and evaluate responses using a unified framework spanning technical, social, and safety dimensions. The framework assesses relevance, accuracy, actionability, persuasiveness, and understandability, alongside platform risks and support characteristics, including social-engineering risk, toxicity, empathy, bias, risky guidance, and support information. We build and validate automated classifiers to scale the evaluation. Our findings reveal differences in support quality across platforms. Google Search and general-purpose LLMs provide more relevant and actionable guidance than Reddit discussions, yet none consistently provide safe, trauma-informed support. More than 65% of victim queries encounter potentially malicious links in search results, over 20% of Reddit discussions contain toxic responses, and conversational AI systems frequently fail to provide risk-aware guidance or concrete support resources. Surprisingly, domain-specific survivor-support chatbots underperform general-purpose LLMs across most dimensions. These findings expose weaknesses in digital support for TFA victims and highlight the need for safety-centered design, evaluation, and deployment of future support technologies. |
| 2026-07-23 | [Same Dangerous Objective, Opposite Advice: Direct Exposure versus Multi-Agent Mediation](http://arxiv.org/abs/2607.21518v1) | Linjun Li | Even a current high-capability LLM can appear safer when shown a dangerous objective directly than when other agents transform and relay its direction. Using OpenAI's gpt-5.6-sol model alias, we test 25 pre-specified mirrored trade-off profiles. Direct exposure to an objective authorizing concealment, fabrication, and pressure produced advice net opposed to its target. After an Id and Censor transformed the same objective into affect and a constraint-rewritten, target-bearing intention, the user-facing Superego---which saw the preferred direction but not the raw objective, its manipulative clauses, or its source---produced advice net aligned with the target.   This behavioral reverse shift is consistent with the model recognizing or distrusting the manipulative motive, although we do not identify its internal mechanism. The second result exposes a compositional safety gap: a current high-capability model can be used as the user-facing component of an automated, multi-stage workflow serving an explicitly manipulative objective. The workflow can keep the raw instruction, its manipulation-authorizing clauses, and its provenance outside the downstream model's context while preserving the objective's target direction. A user with endpoint-only access likewise cannot directly inspect those upstream messages including the objective. |
| 2026-07-23 | [Top-down = Bottom-up: Sound and Complete Characterisations of Liveness by Multiparty Global Protocols](http://arxiv.org/abs/2607.21489v1) | Kai Pischke, Nobuko Yoshida | Multiparty session types (MPST) are a type discipline for concurrent and distributed systems, designed to ensure not only type safety and deadlock-freedom, but also liveness of typed communicating processes. Two main MPST methodologies, top-down and bottom-up, have been proposed and are integrated into a wide range of programming languages and tools. The top-down strategy starts by specifying the overall choreography of the protocol (called a global type), from which a set of local types that satisfy safety and liveness are generated by endpoint projection (EPP). Once each participant is type-checked against a generated local type, liveness of the set of typed processes is automatically ensured by construction. The bottom-up strategy directly checks whether local types inferred from processes satisfy liveness to enforce liveness of processes. Since the top-down strategy depends on global types and the EPP algorithms, it has often been considered that the top-down system offers strictly less typability than the bottom-up system. Our paper negates this belief. We prove that, using the precise subtyping for the subsumption rule, the top-down strategy offers exactly the same typability as the bottom-up system. More precisely, a multiparty session $M$ is typable and verified to be live by the bottom-up typing system if and only if $M$ is typable by the top-down typing system. The key to the proof is a principal global type inference algorithm which builds a principal global type from an arbitrary set of live local types. We have implemented the global type inference algorithm together with projection, process type checking and local type inference algorithms, and built a toolchain for both the top-down and bottom-up strategies. We evaluated our toolchain with representative examples from the literature, confirming that the top-down approach is more efficient than the bottom-up approach. |
| 2026-07-23 | [Out-of-Distribution Detection in Wireless Multimodal Foundation Models for 6G ISAC](http://arxiv.org/abs/2607.21455v1) | Mohammad Farzanullah, Akram Bin Sediq et al. | The integration of Foundation Models (FMs), such as the Wireless Multimodal Foundation Model (WMFM), into 6G networks provides a unified framework for Integrated Sensing and Communication (ISAC), leveraging generalized representations to simultaneously optimize data transmission and environmental perception. However, the deployment of such data-driven models in safety-critical infrastructure is hindered by the Out-of-Distribution (OOD) problem, which poses a fundamental threat to system trustworthiness. Standard FMs operate under a closed-world assumption, rendering them vulnerable to silent failures when deployed in unseen radio environments. To address this reliability gap and ensure trustworthy network operation, we propose WMFM-OOD, a robust metric-based OOD detection framework. Unlike traditional methods that rely on raw compatibility scores, WMFM-OOD constructs geometric Base Station (BS) Prototypes within the joint latent space to capture the manifold structure of valid radio environments. By employing a temperature-scaled probabilistic scoring mechanism, our approach effectively distinguishes between In-Distribution (ID) and covariate-shifted anomalies. We validate the framework on the DeepVerse6G dataset. Experimental results demonstrate that WMFM-OOD significantly outperforms uncalibrated baselines, achieving an Area Under the Receiver Operating Characteristic Curve (AUROC) of 0.8824 and reducing the False Positive Rate (FPR) at 95 % True Positive Rate (TPR), commonly referred to as FPR95, by approximately 17% in the optimal temperature regime, thereby providing an initial layer of detection sensitivity to mitigate catastrophic model failures without completely disrupting network availability. |
| 2026-07-23 | [Token Budget Saturation and Mechanistic Early Detection of Reasoning Non-Convergence in Chain-of-Thought Models](http://arxiv.org/abs/2607.21433v1) | Renuka Oladri, Niveda Jawahar et al. | Chain-of-thought reasoning models such as DeepSeek-R1-Distill-Qwen-7B exhibit a bimodal convergence pattern: generations either terminate within a token budget (converged) or exhaust it without reaching a conclusion (non-converged). We characterize this phenomenon empirically, showing that converged generations achieve 90.3% accuracy on AIME 1983-2024 while non-converged ones achieve only 6.6%, with an overall convergence rate of 62.0%. We then ask whether this outcome is detectable early in the thinking chain using internal model representations. Training linear probes on hidden-state activations at token positions 50-300, we find that layer-20 activations at token 150 achieve AUC 0.608 (+-0.080, 5-fold CV), reliably above chance even at token 50. Activation probes consistently outperform behavioral baselines derived from token entropy and repetition statistics. A sweep-level permutation test yields p=0.063 (100,000 permutations), consistent with a modest signal that our sample size cannot confirm at conventional thresholds. These findings suggest that convergence fate is partially encoded in intermediate representations well before the generation ends, opening a path toward early-exit inference and adaptive compute allocation. |
| 2026-07-23 | [Euclid-MCP: A Model Context Protocol Server for Deterministic Logical Reasoning via Prolog](http://arxiv.org/abs/2607.21412v1) | Bartolomeo Bogliolo | Large Language Models (LLMs) excel at natural language understanding and generation but remain unreliable for multi-step logical reasoning, especially in safety-critical or compliance-sensitive domains. Recent neuro-symbolic approaches address this gap by coupling neural models with external symbolic engines, yet most integrations are bespoke and lack a standardized interface for tool-augmented agents. This paper presents Euclid-MCP, an open-source MCP server that provides deterministic logical reasoning via SWI-Prolog. Euclid-MCP introduces Euclid-IR, an engine-agnostic intermediate representation for Horn-clause logic that is human-readable, easy for LLMs to generate, and straightforward to compile into Prolog or alternative backends. The server exposes a compact tool interface that supports a translate-run-inspect-repair loop, enabling LLM clients to delegate inference while retaining full access to proof traces and derivation logs. We evaluate Euclid-MCP on a realistic IT security and compliance use case. Results show that while LLMs alone are sufficient on small knowledge bases, they hallucinate systematically on larger problems, whereas Euclid-MCP delivers exact answers with lower latency and more compact outputs. We argue that semantic RAG is fundamentally unsuited for rule enforcement, and that Euclid-MCP can serve as a stable, shared reasoning substrate for both RAG-based assistants and agentic systems. |
| 2026-07-23 | [When Are Reasoning-Based Guardrails Not Efficient? ResponseGuard: A Fast Vision-Language Guard for Real-Time Moderation](http://arxiv.org/abs/2607.21401v1) | Dongbin Na | A vision-language AI assistant returns its answer as a stream of generated tokens. Therefore, a safety guard that watches that answer has to keep up with the stream and stop a harmful reply before a user reads it. Recent vision-language guardrails instead generate a chain of thought before they issue a verdict. They believe that step-by-step reasoning yields a safer guard. This design makes the guard heavy and slow, since the model must decode many tokens for harmfulness detection. We pose the question of whether a vision-language guard really needs to reason in order to screen a response. We answer with a guard that has no chain. ResponseGuard reads a harmful verdict from a single pooled representation of the request, the response, and the image in one forward pass. Across a standard multimodal guardrail benchmark, our 2B ResponseGuard outperforms a recent 3B reasoning-based vision-language guard on response harmfulness detection, without any reasoning and at about 150 times lower time cost. On request harmfulness the reasoning guard retains an overall lead, and the remaining gap on both tracks sits on the image-only cells. We observe that the gap may stem from the frozen vision encoders that both designs use rather than from the missing chain. We have also found the reasoning guard directs almost none of its verdict attention to the image. Based on a single-pass detection, ResponseGuard can screen an answer sentence by sentence as it streams and stop a harmful answer before it finishes. For guarding the response of a vision-language model, a calibrated single-pass label may provide a sufficient safety signal. We fully release all source code, trained models, and datasets at https://github.com/ndb796/ResponseGuard. |
| 2026-07-23 | [Grasp, Handover, Rotate: Bimanual Object Reorientation via Compositional Diffusion and Energy-Based Optimization](http://arxiv.org/abs/2607.21341v1) | Wun Lam Yeung, Wenjun Liu et al. | Bimanual object reorientation - picking an object, handing it over between two arms, and placing it in a desired target pose - is valuable when direct placement from the initial grasp is infeasible due to collisions, kinematic constraints, or poor final orientation. However, achieving this under multiple competing objectives remains challenging. We introduce BiCompoDiff, a compositional diffusion and energy-based framework that jointly optimizes grasp selection, handover, regrasp, and motion planning under multiple constraints. By combining a pretrained grasp diffusion model with bimanual planning energy-based models (EBMs), our method injects gradient guidance during reverse diffusion to enforce collision avoidance, trajectory smoothness (via differentiable inverse kinematics), handover feasibility, and regrasp safety. Annealed MCMC sampling further refines grasp poses over the composite energy landscape. Experiments across diverse simulated household reorientation tasks demonstrate that BiCompoDiff achieves over 20% higher success rates and up to 37% smoother trajectories (measured by joint displacement) compared to strong sampling-based baselines. Real-world validation confirms effective sim-to-real transfer and robust performance on challenging scenes. |
| 2026-07-23 | [Detectors Learn the Wrong Thing: Shortcut-Resistant Adversarial Training Against Physically Realizable Attacks](http://arxiv.org/abs/2607.21243v1) | Yuanhao Huang, Yilong Ren et al. | AI-enabled visual perception systems are increasingly deployed in intelligent transportation infrastructure and autonomous vehicle related applications. However, physically realizable adversarial appearances pose a significant reliability challenge for these safety-critical systems. Adversarial training is effective, but repeated co-occurrence between adversarial texture and positive person instances can cause detectors to treat the texture itself as evidence of object presence, forming a patch texture shortcut. The detector may then treat texture as evidence for the target, causing false detections on texture-only inputs and weakening cross attack generalisation. We propose InsCAT, an instance-level contrastive adversarial training framework that prevents detectors from using adversarial texture as an independent decision cue. SICA aligns adversarial person features with matched clean features and separates them from texture-only negatives, while ROPO and Guard maintain online attack pressure and coordinate training. We evaluate eight independently generated attack textures on rendered nuScenes, INRIAPerson, printed garments, and three detector families. InsCAT achieves an average attack AP of 82.3% on rendered nuScenes, exceeding the strongest baseline by 11.1 points.Relative to AT-Mix, texture FPR decreases from 46.9% to 7.3%. Physical tests yield an F1 score of 96.6% and an FPR of 1.8%. Consistent gains across separately trained detectors demonstrate applicability across architectures with direct inference. The findings show that robust physical detection depends on preserving target related evidence while preventing adversarial texture from becoming an independent decision cu |
| 2026-07-23 | [Hybrid MKNF with Classical Negation in the Rule Component](http://arxiv.org/abs/2607.21202v1) | Arun Raveendran Nair Sheela, Christophe Rey et al. | Hybrid MKNF knowledge bases under the well-founded semantics integrate Description Logics with Logic Programming. However, they do not support classical negation in the rule component, limiting their ability to represent explicit negative knowledge. This limitation is particularly significant in safety-critical applications, where reasoning often requires explicit negative information rather than interpreting the absence of information as evidence of absence. To address this issue, we introduce an extension of Hybrid MKNF that supports classical negation in the rule component. We formally define the syntax and semantics of the extended language and present a general procedure for computing its well-founded model. |
| 2026-07-23 | [SafeStep: AI-powered Travel Assistance for Elderly People with Frailty or Dementia](http://arxiv.org/abs/2607.21156v1) | Elderly People with Frailty or Dementia Azul Debenedetti, David Gamez et al. | More than a million people in the UK suffer from frailty or dementia, which severely compromise their ability to travel in urban environments. This paper presents SafeStep, an AI-driven travel system that assists elderly users with their journeys. At the core of SafeStep is a novel travel graph representation, which integrates route planning with predictive modelling. For each stage of a journey, the system (i) generates personalized failure scenarios using a combi-nation of LLMs and the Anticip8 behavioral prediction engine, (ii) proposes targeted interventions, and (iii) estimates the impact of interventions on out-come probabilities. This enables SafeStep to select interventions that maximize the likelihood of the person reaching their destination. SafeStep was evaluated through experiments on travel graph generation and a field study involving 26 real-world journeys. Results showed that combining Anticip8 for failure pre-diction with GPT-based models for intervention evaluation yields the most re-liable performance. User feedback indicated that SafeStep improves confidence and perceived safety during travel, although interface usability needs to be im-proved for the target demographic. In the future, we would like to improve and release SafeStep. The AI system that was developed for SafeStep could be ap-plied in other areas, such as mental health, career coaching and addiction treatment. |
| 2026-07-23 | [V-DEAL: Diagnosing Video Safety De-Calibration as an Understanding-Refusal Coupling Failure](http://arxiv.org/abs/2607.21151v1) | Zhetong Zhang, Honghao Fu et al. | As Video Large Language Models are increasingly deployed in real-world applications, ensuring their safety alignment has become critical. Counterintuitively, we find that harmful videos paired with benign queries achieve higher attack success rates than the same videos paired with explicitly harmful queries. To understand the underlying mechanism of this vulnerability, we present V-DEAL, a three-level diagnostic framework that jointly analyzes this failure across model behaviour, understanding, and internal representations. By progressively ruling out perception failure and quantifying the model's internal refusal tendency, V-DEAL provides a new diagnostic perspective for analyzing the underlying mechanism of the observed vulnerability. We tested six Video LLMs on three public benchmarks and observed that models correctly recognize harmful video content with over 81\% accuracy, yet the average attack success rate still reaches 48.33\% under the condition pairing harmful videos with benign queries. Hidden-state analysis further shows that visual understanding activates a weaker refusal tendency than textual understanding. Furthermore, we introduce a prompt injection intervention method that reduces attack success rates by an average of 48.24 percentage points and achieves performance comparable to prior fine-tuning-based methods, providing an effective and practical means to address such safety risks in Video LLMs. |
| 2026-07-23 | [Safety-oriented sidewalk and road segmentation for smartphone-based assistive navigation](http://arxiv.org/abs/2607.21137v1) | Hakan Calim, Anamaria Dumitrescu et al. | Independent sidewalk mobility is essential for blind and visually impaired pedestrians (BVIPs), yet smartphone-based assistive navigation requires perception models that distinguish walkable sidewalks from adjacent unsafe regions. This study presents a safety-oriented semantic segmentation framework for future mobile guidance. We introduce SENSATION-DS, a chest-height pedestrian-view dataset with 2,752 image-mask pairs and nine-class navigation-relevant taxonomy. External urban and sidewalk datasets were harmonized to this label space, and five segmentation architectures were evaluated using staged target-domain adaptation with mask-conditioned synthetic images and Segment Anything Model 2 (SAM2) pseudo-labels. Models were assessed using mean Intersection over Union (mIoU), road- and sidewalk-specific metrics, Road-as-Sidewalk Error Rate as a proxy false-safe measure, and Android Open Neural Network Exchange benchmarking. Synthetic augmentation generally improved segmentation accuracy, whereas SAM2 pseudo-labels more consistently reduced Road-as-Sidewalk errors. UPerNet-MobileNetV3 achieved the highest offline mIoU (0.715 +/- 0.006), while DeepLabV3Plus-MobileNetV3 achieved the lowest Road-as-Sidewalk Error Rate (0.079) and highest Android runtime at 512x384 (7.383 FPS). These results show that assistive sidewalk perception should be evaluated jointly by segmentation accuracy, proxy false-safe behavior, and smartphone deployment feasibility, while real-world benefit requires validation with BVIP users. This evaluation supports selecting models that balance accurate perception, conservative error behavior, and practical runtime. |
| 2026-07-23 | [Safety and Security: Experimental Validation of Encrypted Model Predictive Control](http://arxiv.org/abs/2607.21136v1) | Juraj Holaza, Martin Kalúz et al. | In this paper, we revisit the problem of an encrypted model predictive control (MPC) design, representing a significant challenge in the recent field of secure process control. Existing methods in secure optimization-based control are non-existent and even partial implementation fails to address the closed-loop system stability and recursive feasibility properties of the constrained MPC. To overcome these limitations, we propose a novel approach that utilizes a polynomial approximation of the optimal control law. This method evaluates the explicit control law within a fully homomorphic encryption framework, ensuring that the controller is securely deployed on any third-party or cloud-based platform, with both process data and controller coefficients protected. Experimental results from a laboratory-scale implementation and validation of the proposed privacy-aware control method demonstrate its advantages. |
| 2026-07-23 | [RL-MACRO: A Cybernetic Closed-Loop Intelligence Framework for Multimodal Adaptive Robotic Craniotomy](http://arxiv.org/abs/2607.21113v1) | Xiao Zhang, Jiaxuan Li et al. | Autonomous robotic craniotomy requires continuous regulation of tool-tissue interactions to mitigate mechanical overload and thermal damage while maintaining surgical efficiency. However, this process is inherently partially observable due to unknown, time-varying tissue properties and the inability to directly measure cutting temperatures under physical occlusion. To address these challenges, we propose RL-MACRO, a cybernetic closed-loop intelligence framework that couples multimodal perception, adaptive decision-making, and robotic execution. This framework empowers the surgical robot to autonomously perceive inaccessible states from partial sensory feedback and dynamically optimize its behaviors under uncertain environment. A CNN-LSTM observer first fuses force and sound feedback to reconstruct the hidden temperature state (R^2=0.939, MAE = 1.717 deg C). This reconstructed temperature, alongside multi-sensor features, forms the belief state for an offline Implicit Q-Learning (IQL) policy. A novel dual-head Actor dynamically coordinates the feed rate, spindle speed, and cutting depth to optimize efficiency within strict safety bounds. These decisions are seamlessly translated into spatial motions via online trajectory re-planning and velocity servoing. Experiments on bovine ribs and six ex vivo goat skulls validate the system's robust perception, adaptive recovery from force/temperature excursions, and smooth execution on irregular surfaces, establishing a data-driven cybernetic paradigm for safe and efficient autonomous bone cutting. |
| 2026-07-23 | [STLSat---An Improved Tableau for Satisfiability Checking of Signal Temporal Logic Formulas](http://arxiv.org/abs/2607.21081v1) | Marco Zamponi, Florian Lammel et al. | Signal Temporal Logic (STL) is a formalism used to describe temporal properties of real-valued signals in cyber-physical systems. In mission- and safety-critical domains, specifications often consist of large collections of STL formulas, making consistency checking and requirement analysis a major engineering bottleneck. Despite tableau-based satisfiability procedures being a natural solution to solve this problem, we have recently found out that the only existing tree-shaped tableau for bounded discrete-time STL does not provide a sound satisfiability/unsatisfiability verdict for all possible STL formulas. In this paper, we pinpoint the flaw in that procedure and present a new tree-shaped tableau which we prove to be sound and complete for bounded discrete-time STL.   On top of this theoretical foundation, we introduce STLSat, an open-source Rust tool that decides the satisfiability of STL formulas, synthesizes concrete witness signals, checks the logical implication and equivalence between specifications, and extracts unsatisfiable cores, allowing users to identify inconsistent subsets of requirements for more effective specification debugging. STLSat also implements enhanced First-Order Logic and Satisfiability Modulo Theories encodings for STL, which allow it to act as a portfolio solver.   We evaluate STLSat on an extended benchmark suite (including STL and Mission-time Linear Temporal Logic formulas) that we release publicly. Across the whole benchmark, the portfolio solver matches or outperforms state-of-the-art tools while preserving correctness guaranteed by our sound tableau procedure. |
| 2026-07-23 | [QuantiBias: Benchmarking Quantization-Induced Bias in LLMs](http://arxiv.org/abs/2607.21063v1) | Emilio Ferrara | Almost every large language model that reaches a broad audience is quantized: trained in full precision, then compressed for efficiency. This step is assumed harmless and its safety is rarely re-checked. We find its principal side effect is increased bias that standard safety evaluation misses. Holding the model, its training, and the prompts fixed, a quantized model still refuses harmful requests, still avoids over-refusing benign prompts, and still selects the unbiased multiple-choice answer. Yet asked an open-ended question, the same model volunteers stereotypes in all eight languages we probe, in roughly one in four open-ended answers under an independent judge (~24% to ~27% across the compression ladder): it passes every standard check and still reaches users measurably more biased. The selective gap is a robust finding; whether open-ended bias further increases with compression is less certain, sensitive to the judge that scores it. We address both with \textbf{QuantiBias}, a benchmark that pairs a generative, multilingual stereotype probe with the refusal and multiple-choice controls that isolate open-ended generation, contrasts each build with and without reasoning, and rates the content severity of what it generates. Across two backbone models (Qwen and Gemma), a five-family screen, and eight benchmarks, quantizers allocate their extra precision by capability data that carries no bias-prevention signal, and reasoning before answering roughly halves the effect on some families while doing nothing on others. A quantized build must be re-evaluated for open-ended bias, not only on the short-form safeguards it already passes. |
| 2026-07-23 | [Human-Inspired Framework for Robotic Craniotomy: Integrating Multimodal Fusion and Adaptive Trajectory Adjustment](http://arxiv.org/abs/2607.21058v1) | Renzhen Le, Xiao Zhang et al. | Manual craniotomy is a high-risk, skill-dependent procedure associated with surgeon fatigue and potential dural injury. While robotic approaches have improved safety, existing open-loop systems rely solely on preoperative images and cannot compensate for intraoperative registration errors or tissue deformation. To address this, we propose a human-inspired closed-loop robotic craniotomy framework that intelligently integrates preoperative planning with intraoperative execution. An adaptive dual-contour fusion algorithm is employed to generate trajectories that conform to complex cranial geometries while maintaining a consistent tool-bone relative pose. For intraoperative perception, a multimodal two-stage cross-modal attention block (CMA)-temporal convolutional network (TCN)-Transformer network combined with an adaptive Bayesian filter fuses force and acoustic signals to achieve robust breakthrough detection under varying bone conditions. Upon detection, an in-situ projection-based trajectory adjustment strategy dynamically compensates for depth deviations, enabling safe residual bone isolation. Experiments on bovine ribs show a breakthrough prediction accuracy of 97%, a detection latency of 0.048 +/- 0.097 s, and a maximum overshoot of 0.29 mm. All four ex vivo cranial experiments were successfully completed without dural injury. These results demonstrate that the proposed cybernetic framework enables safe and autonomous craniotomy with highly effective closed-loop control. |
| 2026-07-23 | [A Real-Time Generalized Nash Equilibrium Framework for Interaction-Aware Autonomous Driving in Mixed Traffic](http://arxiv.org/abs/2607.21043v1) | Nouhed Naidja, Mohamed-Cherif Rahal et al. | Safe and efficient navigation in mixed-traffic environments remains a critical challenge for Autonomous Vehicles (AVs), primarily due to the complex interdependence between the AV's decisions and the unpredictable reactions of human drivers. This paper introduces a comprehensive decision-making framework that formulates the driving interaction as a Generalized Nash Equilibrium Problem (GNEP). Unlike decoupled optimization approaches, this framework explicitly models shared safety and geometric constraints, ensuring that the feasibility of the AV's strategy is dynamically linked to the opponent's actions. To solve this non-convex problem in real-time, we propose a dedicated solver based on Particle Swarm Optimization (PSO). The complete architecture was validated on a test track using a real autonomous Renault Zoé interacting with a human driver. Experimental results demonstrate the system's ability to handle critical scenarios by generating comfortable, human-like trajectories. Benchmarks confirm the solver's operational feasibility, achieving convergence in under 50 ms. |
| 2026-07-23 | [GuardianAgentBench: Where Agents Fail and How to Guard Them](http://arxiv.org/abs/2607.20982v1) | Vishal Ishwar Naik, Chenyu Xu et al. | As large language model agents increasingly operate autonomously with access to tools and external environments, ensuring their safe and reliable behavior becomes critical. We present GuardianAgentBench (GABench), a benchmark of 580 scenarios across six domains evaluated on three production-ready frameworks: LangChain, LlamaIndex, and Vectara. The benchmark incorporates rigorous multi-stage validation and five adversarial attack modes. Experiments with six state-of-the-art models reveal that even the strongest configuration achieves only 74.8% overall accuracy and expose two distinct failure regimes: stronger models under-call required tools, while weaker models mis-select and over-call tools. Performance degrades monotonically with both tool-set size and sequential turn depth, with long-horizon planning proving the steeper bottleneck. Our guardrail implementation consistently outperforms system-prompt-based defenses across all models, recovering 19.9% of failures at a false positive rate of just 0.5%. These results demonstrate that execution-time structural intervention improves safety without disrupting correct agent behavior. |

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



