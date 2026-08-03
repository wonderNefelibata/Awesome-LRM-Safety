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
| 2026-07-31 | [Safe Vision Language Action Models via Barrier Enhanced Flow Matching](http://arxiv.org/abs/2607.29569v1) | Kasra Sinaei, Hung-Chieh Wu et al. | This article presents a modular inference framework that integrates Flow Matching generative models with formal Control Barrier Function (CBF) safety guarantees. Unlike existing methods that apply external safety filters to a model's final output, our approach modifies the Flow Matching denoising process within the model to inherently generate safe trajectories. By employing a smooth Log-Sum-Exponential aggregate barrier, we enforce safety over entire action chunks. This aggregate barrier ensures a minimal increase in computational overhead and does not alter the semantic intent of the model. We show that, within the proposed framework, the 2-Wasserstein distance between the generated distribution and the target distribution remains bounded. Our method eliminates the need for safety-specific datasets or costly model retraining, providing a versatile solution for safe inference. We validate the approach on two robotic manipulation platforms and a 2D navigation benchmark, verifying that our framework achieves reliable safety without degrading the success rate of the model. |
| 2026-07-31 | [TransGraspNet: Physically and Geometrically Consistent Manipulation of Transparent Labware](http://arxiv.org/abs/2607.29567v1) | Hailing Hu, Mingyi Zhu et al. | Manipulating transparent laboratory glassware that contains liquid is inherently safety-critical: even small geometric errors can cause unstable grasps and hazardous spillage. Although recent progress has been made in transparent object perception and robotic grasping, most existing systems optimize detection, depth reconstruction, and grasp planning independently, which leads to cross-stage inconsistency imperfect boundaries induce depth bleeding, distorted surfaces corrupt normal estimation, and task agnostic grasp scoring yields tilted or off-center grasps that fail under dynamic motion. In this paper, we propose TransGraspNet, a geometry physics consistent framework that explicitly enforces consistency from perception to execution through three coupled principles: boundary consistency to produce structurally reliable object contours as downstream priors, surface consistency to preserve geometric fidelity and surface normal accuracy during depth reconstruction, and physics consistency to refine grasp selection with centroid alignment and wrench-space stability for upright and dynamically robust manipulation. We evaluate TransGraspNet on public benchmarks, a dedicated transparent glassware dataset, and a real robotic platform. The results show improved boundary quality and surface normal fidelity, and demonstrate strong task-level performance in cluttered transparent scenes. Most importantly, the proposed system achieves reliable real-world operation, including high grasp success rates in clutter and zero spillage during high speed liquid transport, highlighting the effectiveness of our method. |
| 2026-07-31 | [STAGE: STyle-controllable Action GEneration for personalized autonomous driving](http://arxiv.org/abs/2607.29517v1) | Zihao Liu, Xing Liu et al. | Driving style refers to the behavioral preferences that drivers maintain during driving, shaped by their diverse experiences, habits, and needs, and is typically reflected in varying levels of aggressiveness. If humans choose to use autonomous driving systems, they would expect the driving style of the systems to closely resemble their own habit. However, this is challenging for current industrial autonomous driving systems. To address this, we developed a style controllable action generation method, STAGE, for driving tasks. Its training process is based on imitation learning, incorporating both style value and latent value action modality encoding. Preference learning is then used to identify the user's driving style as a continuous, monotonic style value. And to reduce the cost of human involvement in the preference training process, we also developed a set of rules to compare driving style in data pairs. Then, during inference, the user inputs the style value to control the generated action patterns, dynamically meeting the user's expectations. Using the STAGE method, we verified that the style-controlled action generation results in several typical road scenarios significantly align with human expectations. Furthermore, through comparisons between the STAGE method and various other approaches, we reveal the unique functionalities of STAGE, including its style controllability, style continuity, driving style alignment capability and driving safety. The code for this work is available at: https://github.com/CarlDegio/STAGE |
| 2026-07-31 | [Beyond Component Testing: Validating Agentic AI Systems](http://arxiv.org/abs/2607.29405v1) | Fabio Orazio Mirto, Luca D'Agati et al. | Agentic AI systems act through multi-step trajectories that combine planning, tool use, memory, interaction, and adaptation. This behavior stretches validation practice beyond component testing and one-shot input--output evaluation, because acceptable system behavior now depends on how decisions unfold over time and under changing environmental conditions. This survey synthesizes 257 papers spanning agent evaluation, software assurance, cyber-physical systems, runtime monitoring, and regulatory guidance in order to characterize the validation problem for agentic systems. The review is organized around a five-dimension taxonomy covering behavioral, safety, temporal, regulatory, and multi-agent concerns, and uses that taxonomy to map current approaches and expose recurrent coverage gaps. The analysis shows that behavioral evaluation is comparatively mature, while temporal validity, runtime evidence maintenance, regulatory legibility, and open-ended multi-agent systems assurance remain under-developed. Three cross-domain case studies (medical care, industrial operations, smart-mobility systems) provide operational illustrations of how the five taxonomy dimensions recur in safety-critical settings, grounded in the failure patterns documented in the reviewed literature. The paper concludes with a lifecycle-oriented research agenda centered on bounded-autonomy specifications, adversarial trajectory generation, runtime monitoring, and audit-ready evidence structures. The central claim is that trustworthy deployment of agentic AI depends on validating trajectories in context rather than assessing isolated components alone. |
| 2026-07-31 | [BRHC: Backend-driven Reactive Hypermedia Controls with a Statically Typed Kotlin DSL](http://arxiv.org/abs/2607.29338v1) | Fernando Miguel Carvalho, Paulo Carvalho et al. | AI-assisted coding tools (e.g., Copilot, Cursor, Claude) are increasingly ubiquitous and enable rapid generation of web applications. However, this raises concerns regarding complexity, longevity and the long-term maintainability of generated systems. A key source of complexity is the heterogeneity between backend and frontend programming models, where multiple languages and paradigms are combined within a single application, often leading to duplicated logic and fragmented state management. To address this issue, recent approaches (e.g., HTMX, Turbo Hotwire, Datastar, etc.) follow the Hypermedia-Driven Application (HDA) model, positioning HTML as the primary communication medium between client and server. Unlike SPA-centric architectures, HDA systems shift the application state and interaction logic to the server, where backend-driven reactive signals synchronize with the client user interface. However, these approaches still introduce complexity through custom attributes and do not fully eliminate JavaScript, particularly in computed expressions. In this work, we propose a statically typed approach using a Kotlin-based HTML DSL (Domain-Specific Language) for backend-driven reactive web applications. We extend the HtmlFlow Kotlin DSL with typed custom HTML attributes (i.e., Datastar data-* attributes) and signal-based bindings using statically typed builders. We demonstrate the approach through a catalog of reactive interaction patterns and a Petclinic Spring MVC case study. The results indicate that the proposed approach can nearly eliminate the need for JavaScript while improving type safety and preserving a homogeneous programming model across frontend and backend, bridged through a backend-driven reactive, signal-centric architecture. |
| 2026-07-31 | [Translation with Thought: Difficulty-Adaptive Reasoning via Reinforcement Learning for Multi-Domain Machine Translation](http://arxiv.org/abs/2607.29287v1) | Yongshi Ye, Biao Fu et al. | Multi-domain machine translation (MDMT) poses a unique challenge due to varying levels of linguistic complexity across domains. Inspired by human translators' ability to adapt reasoning effort based on difficulty, we propose TwT (Translation with Thought), a resource-rational framework that learns to modulate inference between intuitive and deliberate reasoning. TwT is trained in two stages: (1) supervised fine-tuning on difficulty-aware long chain-of-thought traces distilled from DeepSeek-R1 and rewritten by GPT-4o to reflect human-like reasoning economy, and (2) reinforcement learning with a hybrid reward to optimize translation quality and reasoning efficiency. Evaluated on 15 benchmarks spanning in-domain and out-of-domain settings, as well as 3 seen and 59 unseen languages, with ablations across three backbone models, TwT-7B and TwT-14B outperform much larger SOTA reasoning models in translation quality, while reducing token usage by 32--60\%. These results confirm that aligning translation behavior with cognitive principles enables robust generalization, high translation quality, and efficient reasoning in MDMT. |
| 2026-07-31 | [ParaASR: Multi-Token Prediction for Fast and Long-Context LLM-Based Speech Recognition](http://arxiv.org/abs/2607.29279v1) | Qingjian Lin, Yuxin Li et al. | Audio-encoder-LLM-decoder architectures have become the dominant paradigm for modern automatic speech recognition (ASR), improving transcription quality through large-scale language modeling. However, the cost of autoregressive decoding scales with decoder size, creating a fundamental trade-off between recognition quality and serving latency. We argue this trade-off is not inherent: unlike open-ended text generation, ASR outputs are strongly anchored to the input speech signal, providing a natural inductive bias toward high-parallelism decoding. Building on this, we introduce ParaASR, an ASR system that leverages Multi-Token Prediction (MTP) to let a 4B LLM decoder emit multiple tokens per forward step. Starting from a publicly available audio-language foundation, the model first establishes a robust autoregressive recognizer and then aligns five future-token branches through a staged optimization recipe. At inference, it proposes a six-token continuation per step and admits only the verified prefix into the transcript, preserving the safety of standard autoregressive decoding. The average accepted length reaches 5.0 out of 6 proposed tokens, confirming that the deterministic structure of speech makes ASR an especially natural setting for multi-token decoding. ParaASR further retains a native 32K-context window and transcribes up to 30 minutes of audio in a single pass. Across diverse benchmarks, it attains average error rates of 2.97%, 3.68%, and 3.70% on Chinese, English, and long-form evaluations, respectively, while reaching a real-time factor (RTF) as low as 0.0053. These results show that decoder scaling, low-latency inference, and long-context transcription need not be competing goals when future-token proposals are anchored by the acoustic signal and guarded by autoregressive verification. |
| 2026-07-31 | [Tool Specifications Matter: Uncovering and Mitigating Safety Risks in AI Agents](http://arxiv.org/abs/2607.29254v1) | Minghui Pan, Jiayuxuan Yang et al. | AI agents extend large language models (LLMs) with external tools, enabling them to perform complex tasks and translate model outputs into consequential real-world actions. Yet LLMs often become substantially less safe when deployed as agents, and the source of this degradation remains poorly understood. In this paper, we identify schema-formatted tool specifications as a primary source of agent safety degradation and show, through white-box representation analysis, that they weaken the model's internal refusal signals and contribute to unsafe tool execution. Building on this finding, we propose SafeKeep, an inference-time safeguard that decouples safety judgment from tool execution: it assesses requests using flattened textual tool specifications while retaining the original schema-formatted specifications for execution. Across two representative benchmarks and four LLMs, including both white-box and black-box models, SafeKeep increases the average refusal rate for harmful requests from 23.8% to 70.6% and reduces the average attack success rate under observation-level prompt injection from 25.6% to 2.5%. It also outperforms existing safeguards and preserves task-handling capability. We release the code and data at https://github.com/snowcatsmoking/SafeKeep . |
| 2026-07-31 | [Don't Mix Rewards, Mix Policies: Policy Decomposition and Optimization for Multi-Reward RL](http://arxiv.org/abs/2607.29246v1) | Ruiming Liang, Yi Zhong et al. | Modern large language models (LLMs) are expected not just to answer correctly, but to adapt their behavior to different human values and use cases. As a result, multi-reward reinforcement learning (RL) has become an increasingly important problem for LLMs, where each reward captures a different aspect of desired behavior. However, optimizing with multiple rewards suffers from a more severe alignment tax issue, where different optimization objectives can trade off or even conflict with each other, leading to unstable and inefficient post-training. In this work, we propose PRISM, a new multi-reward RL framework built upon the idea of policy-space decomposition and composition. Instead of compositing different rewards, PRISM optimizes a set of standalone positive policies and a global negative policy. This alleviates the potential conflict during multi-reward policy optimization, while enabling controllability during inference by flexible policy composition. Experiments on scientific reasoning, tool-use reasoning, and helpfulness-safety alignment show that PRISM consistently outperforms existing multi-reward RL baselines, with extra controllability for inference-time preference control. |
| 2026-07-31 | [MROPE: A Multi-Robot Safe Cooperative Strategy via combined Predictive Safety Filters and Ellipse-based Constraint Compression](http://arxiv.org/abs/2607.29203v1) | Alice Rosetti, Lorenzo Pichierri et al. | Deploying drone swarms to track a dynamic target in cluttered environments presents severe computational and safety challenges. We propose MROPE, a hierarchical strategy that decouples the cooperative monitoring mission from strict local safety requirements. To overcome the computational bottlenecks typical of dense spaces, our approach dynamically aggregates complex obstacle geometries into a single safe bounding ellipse for each drone. Methodologically, this architecture is realized by combining distributed aggregative optimization for high-level swarm coordination, a decentralized consensus scheme for the safe area computation, and local Predictive Safety Filters (PSF) for real-time collision avoidance. Virtual and real-world experiments validate the framework, demonstrating superior real-time efficiency and scalability compared to centralized approaches. |
| 2026-07-31 | [Knox: Fortifying Smart Spaces With Safety Guarantees](http://arxiv.org/abs/2607.29198v1) | Rishabh Menezes, Jadon T. Schuler et al. | Internet of Things (IoT) devices in smart spaces and buildings are an emerging class of distributed systems with critical safety requirements. This paper presents Knox, the first system to enable safety checking in IoT-enabled smart spaces. Knox's contributions include (i) safety specifications: a new language for safety clauses in such smart spaces, and (ii) static safety checking: two new algorithms for static verification of multiple safety properties across multiple routines running inside a smart space. Since the latter problem is NP-hard, we present and analyze novel and explainable algorithms for the static version of the problem. We also present optimizations that further reduce runtime. Our analysis and experimental results with real datasets show that Knox reduces checking time significantly compared to baselines, while providing high accuracy in catching safety violations. |
| 2026-07-31 | [Safety Analysis of Metasurface-Based Near-field Wireless Power Transfer System for Deep Implant](http://arxiv.org/abs/2607.29187v1) | Maoyuan Li, Ali Khaleghi et al. | Wireless power transfer is a method for energizing future implantable medical electronics. In this study, a metasurface-based near-field magnetic wireless power transfer system for deep implants is presented, and electromagnetic safety parameters, including field distributions, specific absorption rate (SAR), and temperature variations, are evaluated. The power transfer is modeled for a receiver implant at a distance of 8.5 cm from the designed metasurface. Based on the results, a maximum localized SAR of 0.072 mW/kg is achieved when the efficiency is 1.62%. Moreover, continuous power transfer shows that the local tissue temperature rises by less than 1.1 degrees Celsius. |
| 2026-07-31 | [Few-shot Deep Learning for Phase-Amplitude Aberration Correction in Transcranial Focused Ultrasound](http://arxiv.org/abs/2607.29182v1) | Minju Seol, Minjee Seo et al. | Transcranial focused ultrasound (tFUS) is a non-invasive technique that delivers focused acoustic energy through the skull for neuromodulation and therapeutic applications. However, the heterogeneous structure of the skull induces complex, patient-specific phase and amplitude aberrations that distort the acoustic focus and deviate it from the intended target, compromising therapeutic efficacy and safety. Conventional time-reversal (TR) simulations can correct these aberrations but rely on computationally expensive full-wave solvers, making them impractical for real-time use and iterative treatment planning. We propose a few-shot deep surrogate framework that predicts per-element phase and amplitude corrections for a 96-element 3D phased-array transducer from patient CT images. A geometry-aware encoder extracts skull-path features shared across dedicated phase classification and amplitude regression branches, where phase periodicity is handled via circular expectation decoding. The framework is pretrained on diverse skull geometries and fine-tuned with only ten target points, enabling rapid adaptation to unseen patients without full patient-specific simulation. Evaluated via leave-one-out cross-validation across 12 skulls, it achieves a mean phase CMAE of 0.155 rad and amplitude rMAE of 9.089%, a focal centroid error of 0.467 mm, Dice score of 94.422%, and peak pressure ratio of 92.332%, with an approximately 2,535 times speedup over TR simulation. The code is available at https://github.com/Minju-Seol/fewshot-tfus-correction. |
| 2026-07-31 | [First Investigation of Deep Learning for Intraoperative Gauze Segmentation in Minimally Invasive Abdominal Surgery](http://arxiv.org/abs/2607.29132v1) | Priya Tomar, Maximilian Broß et al. | Surgical gauze is an essential part of surgical procedures, primarily used for controlling bleeding and absorbing bodily fluids. The post-surgical retention of gauze can lead to serious complications and necessitate additional surgery for its removal. Despite the clinical significance, research on gauze segmentation using real-world surgical data remains underexplored, owing in part to the scarcity of annotated datasets. In this work, we investigate the use of deep learning methods for gauze segmentation in robot-assisted minimally invasive abdominal surgeries, utilizing an in-house surgical dataset prepared at a university hospital. The training data reflects realistic surgical settings and captures extensive diversity in spatial, morphological, and visual attributes across three different gauze categories. We evaluate several widely used segmentation architectures, including CNN-based, transformer-based, and hybrid architectures, to establish a proof-of-concept for gauze segmentation in a realistic clinical setting. In addition, we investigate the influence of sub-optimally annotated, auto-tracked segmentation masks as a strategy to address data scarcity and improve performance. Our results demonstrate the efficacy of real-world training data in countering the main challenge reported by prior works, the trade-off between blood presence and gauze detection. The incorporation of auto-tracked annotations yields performance enhancements, particularly in generic surgical scenarios. The integration of effective segmentation approaches can benefit robot-guided surgical procedures and various downstream applications by providing precise delineation of foreign objects, thereby enhancing patient safety and surgical outcomes. |
| 2026-07-31 | [IyawoBench v2.0: Extended Diagnostic Evaluation of Large Language Model Clinical Triage in Nigerian Primary Care](http://arxiv.org/abs/2607.29085v1) | Anthonio Oladimeji Gabriel, Dimeji Olawuyi | Large language models are being deployed as clinical triage tools in low and middle income countries where trained physicians are scarce. Existing safety metrics, however, produce misleading confidence: models scoring 100% on binary "did not send an emergency home" safety measures may nevertheless exhibit systematic failure modes that render them undeployable at scale. We present IyawoBench v2.0, an extended diagnostic evaluation of large language model clinical triage on 200 synthetic vignettes derived from 1,200 real patient encounters at 19 Nigerian primary health centres. We introduce a formal mathematical framework comprising fourteen definitions and two theorems that decompose triage safety into three distinct failure modes: Conservative Escalation Bias, Systematic Downgrade Bias, and Middle-Tier Instability. We propose the Escalation Bias Index and Expected Deployment Cost as novel metrics that expose failure modes hidden by conventional accuracy and sensitivity scores. Evaluated on three frontier models (Claude Sonnet 4.6, Llama 3.3 70B, Llama 3.1 8B) plus five naive baselines, we show that: (1) all three models exhibit at least one formal failure mode; (2) traditional sensitivity metrics conceal a 77 percentage point under-triage gap in Llama 3.1 8B; (3) the optimal model varies across three deployment scenarios (Emergency-Focused, System-Sustainability, Balanced), demonstrating that single-ranking benchmarks are inadequate for LMIC clinical AI selection. IyawoBench v2.0 provides both a rigorous benchmark and a diagnostic framework transferable to any triage-style clinical AI evaluation. All code, data, and analysis pipelines are publicly available. |
| 2026-07-31 | [Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models](http://arxiv.org/abs/2607.29079v1) | Yaoxuan Dou, Yang Shu | Training-free acceleration makes diffusion-based multimodal large language models (dMLLMs) more deployable, but it may silently change generated content. We study this serving-time consistency problem on 300 real images, comparing Fast-dLLM outputs with the same model's unaccelerated outputs. Across the mild parallelism induced in our long-form setting (1.05--1.25 committed tokens per step), confidence-threshold tuning changes decoding behavior but not baseline agreement. State-refresh ablations and an image-swap intervention instead identify stale visual and generated-text states as contributors to drift. For the tested Fast-dLLM implementation, shortening the KV-cache refresh interval yields a monotonic speed--agreement frontier and near-exact agreement at a measured 1.3x speedup. The initial diagnosis also appears with dLLM-Cache and LaViDa, although dLLM-Cache recovers agreement only after both caches are tightened, which removes its speed advantage. Independent prompts and images reproduce the threshold-insensitivity and refresh recovery. A targeted audit finds genuine content substitution in half of 50 low-agreement pairs. In a separate blinded two-annotator evaluation, the pooled accelerated-minus-baseline factual-error difference is 0.00 (95% CI [-0.17,+0.17]); this sample detects no difference but does not establish factual equivalence. Finally, none of the tested adaptive or smoothed-refresh variants beats the fixed interval at matched compute. Our contribution is a paired diagnostic and an implementation-scoped consistency control, not an accuracy or safety guarantee. |
| 2026-07-31 | [On the Generalization of Steering Vectors for Chain-of-Thought Faithfulness](http://arxiv.org/abs/2607.29062v1) | Matthew Nguyen, Kyle Cox et al. | Model capabilities have improved in large part due to scaling chain of thought. This has been a promising development for AI safety--where models verbalize their reasoning, it is possible to monitor it. However, in some cases, models do not verbalize important steps in their reasoning process. For example, models prompted with a cue suggesting the incorrect answer may fail to acknowledge that cue, even when it appears instrumental to their conclusion. When chain of thought (CoT) fails to disclose instrumental reasoning steps, we describe it as unfaithful. Prior work has shown that activation steering can be a useful method to improve faithfulness in CoT. We extend this line of work by studying how well steering for faithfulness generalizes across cue types, datasets, and methods of constructing the steering vector for three models (Gemma-3 4B, Qwen-3.5 9B, Gemma-3 12B) in a cued question-answering setting. While steering reliably increases cue acknowledgment for only the largest model (Gemma-3 12B), we find that when steering is effective, its effect generalizes broadly across cue types and datasets--in cross-cue and cross-dataset analyses, effect size is determined primarily by the evaluation setting, rather than the vector's train setting. How the vector is built also matters little--four construction methods, including one whose optimization target mentions no specific cue, yield similar effect sizes. Finally, we consider the possibility that steering promotes the salience of the cue and causes greater cue use, rather than targeting verbalization behaviors. However, we find no evidence for this--steering leaves the rate of cue use roughly unchanged while reducing hidden cue use, i.e., cue use that is not acknowledged. |
| 2026-07-31 | [Mechanism-Dependent Descriptors Enable Predictive Design of Oxygen Capacity in Perovskite Oxides](http://arxiv.org/abs/2607.29014v1) | Shuiping Gong, Yi Li et al. | Perovskite oxides can reversibly accommodate substantial changes in oxygen stoichiometry, making them attractive for clean-energy technologies including chemical looping and oxygen storage. Despite extensive efforts to optimize their redox properties, predictive descriptors capable of assessing oxygen capacity across diverse compositions remain under development. Here, we combine experiments and first-principles calculations to establish composition and oxygen-capacity relationships in the model perovskite series LnxSr1-xCoO3. We confirm that increasing Sr2+ content promotes the formation of high-valence Co4+, expanding the cationic redox reservoir available during oxygen release and thereby enhancing oxygen capacity. In this regime, oxygen-vacancy formation energy captures the observed trend because oxygen release is primarily compensated by Co4+/Co3+/Co2+ redox. Across the rare-earth series, however, oxygen capacity decreases from La to Lu despite progressively lower oxygen-vacancy formation energies. We reveal that this counterintuitive behavior originates from an alternative charge-compensation pathway, in which lattice oxygen is partially oxidized to O1- -like species during oxygen removal. Heavy rare-earth compositions (Tb-Lu) preferentially stabilize these oxygen-hole species through distinct local bonding environments, with charge compensation involving both oxidized lattice oxygen and reduced rare-earth and cobalt cations, thereby suppressing net oxygen release despite favorable vacancy thermodynamics. We further identify average metal-oxygen bond strength, quantified by integrated crystal orbital Hamilton population, as a physically meaningful descriptor for oxygen capacity when anionic redox becomes dominant. |
| 2026-07-31 | [SULAND v2: A Refined RGB Dataset and Deep Learning Object Detection Benchmark for UAV/UGV-Based SUrface LANDmine Detection Under Domain Shift](http://arxiv.org/abs/2607.28996v1) | Sagar Lekhak, Prasanna Reddy Pulakurthi et al. | RGB imagery offers a practical, low-cost option for Unmanned Aerial/Ground Vehicle (UAV/UGV) survey support in surface-landmine detection, but object detectors remain underexplored in this safety-critical domain. Limited cross-architecture benchmarking and insufficient out-of-distribution (OOD) analysis obscure whether detectors generalize across deployment conditions. This challenge is amplified by the scarcity of public RGB landmine datasets, making SULAND a key benchmark for PFM-1 and PMA-2 detection. However, inspection reveals missing/false annotations, localization errors, inconsistent visibility criteria, visual artifacts, temporal labeling inconsistencies, and an inverted OOD class-ID convention in SULAND. We present SULAND_v2, a refined RGB surface-landmine dataset and benchmark. Preserving original images and splits, we manually revise annotations to ensure completeness, precise localization, label validity, and class consistency. SULAND_v2 contains 33,771 images and 12,433 bounding boxes. We benchmark 35 detector configurations across nine families. Annotation refinement improves YOLOv8 in-distribution (IID) test mAP@50 by 14.6-19.6 percentage points, while fixing the OOD class-ID convention increases mean YOLOv8 OOD mAP@50 by ~25 percentage points. On SULAND_v2, YOLOv12-Small achieves the highest IID mAP@50 (0.908), while RF-DETR-Large yields the strongest OOD performance (0.799 mAP@50, 0.675 recall). Our results demonstrate that high IID accuracy does not guarantee operational readiness. SULAND_v2 provides a reliable benchmark for evaluating domain-shift robustness in RGB-based mine-action survey support. |
| 2026-07-31 | [SafeNexus: Discovering and Steering Modality-Universal Safety Neurons in MLLMs](http://arxiv.org/abs/2607.28969v1) | Jian Yu, Fei Shen et al. | Although Large Language Models (LLMs) have demonstrated promising safety performance, extending them to Multimodal Large Language Models (MLLMs) exposes a significant gap between expanded multimodal capabilities and existing safety mechanisms. Current defenses remain predominantly confined to specific modal settings, thereby limiting their robustness against broader cross-modal threats. To bridge this gap, we introduce SafeNexus, a cross-modal safety alignment framework that adopts a dedicated neuron-level intervention strategy. First, we formulate a neuron localization paradigm that identifies functionally specialized neurons by characterizing intermediate-layer activation patterns and quantifying their functional salience through importance scoring. Building upon this paradigm, we exploit contrastive data to identify modality-bound safety neurons (BS-Neurons), and validate their role in regulating safety behavior within each modality via targeted suppression. Further cross-modal analysis defines modality-universal safety neurons (US-Neurons) as the shared subset of BS-Neurons identified across individual modalities, serving as the core for defending against harmful cross-modal attacks. We observe that suppressing these neurons substantially degrades safety performance across modalities, while leaving overall utility largely unaffected. Building on these insights, we propose two safety alignment strategies: activation-level safety amplifier and safety neuron calibrator. The proposed strategies enhance model safety through two distinct routes: the former amplifies the activation magnitudes of US-Neurons, while the latter selectively calibrates them via targeted fine-tuning. Extensive experiments demonstrate that our method outperforms prevailing state-of-the-art approaches on safety benchmarks spanning diverse modality combinations, while effectively preserving utility. |

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



