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
| 2026-03-13 | [Microscopic flexoelectricity in the canonical PMN relaxor](http://arxiv.org/abs/2603.13202v1) | J. Hlinka | Previously reported neutron scattering investigations of the canonical relaxor ferroelectric perovskite oxide with a chemical formula Pb(Mg(1/3)Nb(2/3))O3 (PMN) are revisited in order to appreciate the role of the intrinsic bulk flexoelectricity. Despite the outstanding electromechanical properties of lead-based relaxors, the magnitude of the flexoelectric coupling coefficient derived here directly from the PMN neutron diffuse scattering data, does not exceed the range of values typical for conventional perovskite ferroelectrics. We explain how these findings are related in the framework of the Ginzburg-Landau-Devonshire and the ferroelectric soft mode theory. We propose that the relaxor properties of PMN might be related to the suppression of the transverse correlation length of the flexoelectrically hybridized translational-polarization fluctuations due to its closeness to the Lifshitz-point regime. |
| 2026-03-13 | [Semantic Invariance in Agentic AI](http://arxiv.org/abs/2603.13173v1) | I. de Zarzà, J. de Curtò et al. | Large Language Models (LLMs) increasingly serve as autonomous reasoning agents in decision support, scientific problem-solving, and multi-agent coordination systems. However, deploying LLM agents in consequential applications requires assurance that their reasoning remains stable under semantically equivalent input variations, a property we term semantic invariance.Standard benchmark evaluations, which assess accuracy on fixed, canonical problem formulations, fail to capture this critical reliability dimension. To address this shortcoming, in this paper we present a metamorphic testing framework for systematically assessing the robustness of LLM reasoning agents, applying eight semantic-preserving transformations (identity, paraphrase, fact reordering, expansion, contraction, academic context, business context, and contrastive formulation) across seven foundation models spanning four distinct architectural families: Hermes (70B, 405B), Qwen3 (30B-A3B, 235B-A22B), DeepSeek-R1, and gpt-oss (20B, 120B). Our evaluation encompasses 19 multi-step reasoning problems across eight scientific domains. The results reveal that model scale does not predict robustness: the smaller Qwen3-30B-A3B achieves the highest stability (79.6% invariant responses, semantic similarity 0.91), while larger models exhibit greater fragility. |
| 2026-03-13 | [Defensible Design for OpenClaw: Securing Autonomous Tool-Invoking Agents](http://arxiv.org/abs/2603.13151v1) | Zongwei Li, Wenkai Li et al. | OpenClaw-like agents offer substantial productivity benefits, yet they are insecure by default because they combine untrusted inputs, autonomous action, extensibility, and privileged system access within a single execution loop. We use OpenClaw as an exemplar of a broader class of agents that interact with interfaces, manipulate files, invoke tools, and install extensions in real operating environments. Consequently, their security should be treated as a software engineering problem rather than as a product-specific concern. To address these architectural vulnerabilities, we propose a blueprint for defensible design. We present a risk taxonomy, secure engineering principles, and a practical research agenda to institutionalize safety in agent construction. Our goal is to transition the community focus from isolated vulnerability patching toward systematic defensive engineering and robust deployment practices. |
| 2026-03-13 | [IceCube Search for MeV Neutrinos from Mergers using Gravitational Wave Catalogs](http://arxiv.org/abs/2603.13076v1) | Nora Valtonen-Mattila | We report on a search using the IceCube Neutrino Observatory for MeV neutrinos from compact binary mergers detected through gravitational waves during the LIGO-Virgo-KAGRA (LVK) O1, O2, and O3 observing runs. The search focuses on events involving at least one candidate neutron star, such as binary neutron star (BNS) and neutron star--black hole (NSBH) mergers, which may produce a burst of thermal neutrinos due to the hot and dense conditions created during the merger. We looked for short-time increases in IceCube's detector activity around each gravitational-wave event, using four time windows centered on the merger time. We also performed a binomial test for two populations, those with and without at least one neutron star. No significant excess of neutrinos was found. We set upper limits on the MeV neutrino flux for each event, and we place constraints on MeV neutrino emission from mergers that have at least one neutron star. We showcase upper limits for GW170817, the first confirmed BNS merger, providing one of the strongest limits to date on MeV neutrino emission from such sources. |
| 2026-03-13 | [Before and After ChatGPT: Revisiting AI-Based Dialogue Systems for Emotional Support](http://arxiv.org/abs/2603.13043v1) | Daeun Lee, Dongje Yoo et al. | Mental health remains a major public health concern, while access to timely psychological support is often limited. AI-based dialogue systems have emerged as promising tools to address these barriers, and recent advances in large language models (LLMs) have significantly transformed this research area. However, a systematic understanding of this technological transition is still limited. This study reviews the technological evolution of AI-driven dialogue systems for mental health, focusing on the shift from task-specific deep learning models to LLM-based approaches. We conducted a bibliometric analysis and qualitative trend review of studies published between 2020 and May 2024 using Web of Science, Scopus, and the ACM Digital Library. The qualitative analysis compared research conducted before and after the widespread adoption of LLMs. Pre-LLM research was represented by highly cited studies and work based on the ESConv dataset, while post-LLM research included highly cited dialogue systems built on LLMs. A total of 146 studies met the inclusion criteria, showing a steady growth in publications over time. Before the widespread use of LLMs, empathetic response generation mainly relied on task-specific deep learning models. Highly cited and ESConv-based studies commonly focused on multi-task learning and the integration of external knowledge. In contrast, recent LLM-based dialogue systems demonstrate improved linguistic flexibility and generalization for emotional support. However, these systems also raise concerns related to reliability and safety in mental health applications. This review highlights the technological transition of AI-based dialogue systems for mental health in the LLM era. By identifying current research trends and limitations, the findings provide guidance for developing more effective and reliable AI-driven counseling systems. |
| 2026-03-13 | [Beyond Imitation: Reinforcement Learning Fine-Tuning for Adaptive Diffusion Navigation Policies](http://arxiv.org/abs/2603.12868v1) | Junhe Sheng, Ruofei Bai et al. | Diffusion-based robot navigation policies trained on large-scale imitation learning datasets, can generate multi-modal trajectories directly from the robot's visual observations, bypassing the traditional localization-mapping-planning pipeline and achieving strong zero-shot generalization. However, their performance remains constrained by the coverage of offline datasets, and when deployed in unseen settings, distribution shift often leads to accumulated trajectory errors and safety-critical failures. Adapting diffusion policies with reinforcement learning is challenging because their iterative denoising structure hinders effective gradient backpropagation, while also making the training of an additional value network computationally expensive and less stable. To address these issues, we propose a reinforcement learning fine-tuning framework tailored for diffusion-based navigation. The method leverages the inherent multi-trajectory sampling mechanism of diffusion models and adopts Group Relative Policy Optimization (GRPO), which estimates relative advantages across sampled trajectories without requiring a separate value network. To preserve pretrained representations while enabling adaptation, we freeze the visual encoder and selectively update the higher decoder layers and action head, enhancing safety-aware behaviors through online environmental feedback. On the PointGoal task in Isaac Sim, our approach improves the Success Rate from 52.0% to 58.7% and SPL from 0.49 to 0.54 on unseen scenes, while reducing collision frequency. Additional experiments show that the fine-tuned policy transfers zero-shot to a real quadruped platform and maintains stable performance in geometrically out-of-distribution environments, suggesting improved adaptability and safe generalization to new domains. |
| 2026-03-13 | [Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](http://arxiv.org/abs/2603.12864v1) | Yifan Zhan, Zhengqing Chen et al. | A major challenge in autonomous driving is the "long tail" of safety-critical edge cases, which often emerge from unusual combinations of common traffic elements. Synthesizing these scenarios is crucial, yet current controllable generative models provide incomplete or entangled guidance, preventing the independent manipulation of scene structure, object identity, and ego actions. We introduce CompoSIA, a compositional driving video simulator that disentangles these traffic factors, enabling fine-grained control over diverse adversarial driving scenarios. To support controllable identity replacement of scene elements, we propose a noise-level identity injection, allowing pose-agnostic identity generation across diverse element poses, all from a single reference image. Furthermore, a hierarchical dual-branch action control mechanism is introduced to improve action controllability. Such disentangled control enables adversarial scenario synthesis-systematically combining safe elements into dangerous configurations that entangled generators cannot produce. Extensive comparisons demonstrate superior controllable generation quality over state-of-the-art baselines, with a 17% improvement in FVD for identity editing and reductions of 30% and 47% in rotation and translation errors for action control. Furthermore, downstream stress-testing reveals substantial planner failures: across editing modalities, the average collision rate of 3s increases by 173%. |
| 2026-03-13 | [Adaptive Vision-Language Model Routing for Computer Use Agents](http://arxiv.org/abs/2603.12823v1) | Xunzhuo Liu, Bowei He et al. | Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically across VLMs, while current CUA systems typically route every action to a single fixed model regardless of difficulty. We propose \textbf{Adaptive VLM Routing} (AVR), a framework that inserts a lightweight semantic routing layer between the CUA orchestrator and a pool of VLMs. For each tool call, AVR estimates action difficulty from multimodal embeddings, probes a small VLM to measure confidence, and routes the action to the cheapest model whose predicted accuracy satisfies a target reliability threshold. For \textit{warm} agents with memory of prior UI interactions, retrieved context further narrows the capability gap between small and large models, allowing many actions to be handled without escalation. We formalize routing as a cost--accuracy trade-off, derive a threshold-based policy for model selection, and evaluate AVR using ScreenSpot-Pro grounding data together with the OpenClaw agent routing benchmark. Across these settings, AVR projects inference cost reductions of up to 78\% while staying within 2 percentage points of an all-large-model baseline. When combined with the Visual Confused Deputy guardrail, AVR also escalates high-risk actions directly to the strongest available model, unifying efficiency and safety within a single routing framework. Materials are also provided Model, benchmark, and code: https://github.com/vllm-project/semantic-router. |
| 2026-03-13 | [Virtual reality for large-scale laboratories based on colorized point clouds: design and pedagogical impact](http://arxiv.org/abs/2603.12727v1) | Lei Fan, Yuxin Li | Effective laboratory training is essential in engineering education, yet conventional on-site instruction is often constrained by time, accessibility, and safety considerations. To address these challenges, this study presents the design, implementation, and evaluation of a web-based virtual reality (WebVR) representation of a large-scale engineering laboratory constructed from massive colorized point cloud data. This study proposes a novel WebVR framework that integrates Unity and Potree for high-fidelity point-cloud visualization combined with advanced interactive capabilities in a browser-based virtual laboratory. It supports immersive first-person exploration, guided navigation, interactive hotspots conveying equipment and safety information, as well as emergency evacuation simulations. The usability, educational effectiveness, and overall acceptance of the virtual laboratory were evaluated through an anonymous questionnaire administered to students and laboratory staff. The results indicate overwhelmingly positive feedback, with all participants rating the system as "good" or "excellent" across all evaluation dimensions. Participants particularly emphasized the benefits of immersive exploration and self-directed learning. In addition, qualitative feedback was systematically analyzed to inform future enhancements of the virtual environment. Overall, the findings demonstrate that the WebVR-based virtual laboratory can effectively complement conventional on-site laboratory instruction, offering a scalable, accessible, and low-risk platform that enhances learning experiences in engineering education. |
| 2026-03-13 | [Colluding LoRA: A Composite Attack on LLM Safety Alignment](http://arxiv.org/abs/2603.12681v1) | Sihao Ding | We introduce Colluding LoRA (CoLoRA), an attack in which each adapter appears benign and plausibly functional in isolation, yet their linear composition consistently compromises safety. Unlike attacks that depend on specific input triggers or prompt patterns, CoLoRA is a composition-triggered broad refusal suppression: once a particular set of adapters is loaded, the model undergoes effective alignment degradation, complying with harmful requests without requiring adversarial prompts or suffixes. This attack exploits the combinatorial blindness of current defense systems, where exhaustively scanning all compositions is computationally intractable. Across several open-weight LLMs, CoLoRA achieves benign behavior individually yet high attack success rate after composition, indicating that securing modular LLM supply-chains requires moving beyond single-module verification toward composition-aware defenses. |
| 2026-03-13 | [98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router](http://arxiv.org/abs/2603.12646v1) | Xunzhuo Liu, Bowei He et al. | System-level routers that intercept LLM requests for safety classification, domain routing, and PII detection must be both fast and operationally lightweight: they should add minimal latency to every request, yet not require a dedicated GPU -- an expensive resource better used for LLM inference itself. When the router co-locates on the same GPU as vLLM serving instances, standard attention's $O(n^2)$ memory makes long-context classification (8K--32K tokens) impossible: at 8K tokens, three concurrent classifiers need ${\sim}$4.5\,GB for attention masks alone, far exceeding the memory left by vLLM. We present three staged optimizations for the vLLM Semantic Router, benchmarked on AMD Instinct MI300X, that solve both the latency and the memory problem. \emph{Stage~1}: a custom CK Flash Attention operator for ONNX Runtime on ROCm reduces attention memory from $O(n^2)$ to $O(n)$ and end-to-end (E2E) latency from 4{,}918\,ms to 127\,ms (\textbf{38.7$\times$}), enabling 8K--32K tokens where SDPA OOMs. \emph{Stage~2}: classical NLP prompt compression (TextRank, position weighting, TF-IDF, and novelty scoring) reduces all inputs to ${\sim}$512 tokens without neural inference, capping both latency and GPU memory at a constant regardless of original prompt length (E2E 127$\to$62\,ms, \textbf{2.0$\times$}). \emph{Stage~3}: near-streaming body processing with adaptive chunking and zero-copy JSON eliminates serialization overhead (E2E 62$\to$50\,ms, \textbf{1.2$\times$}). Cumulatively: \textbf{98$\times$} improvement (4{,}918\,ms to 50\,ms), 16K-token routing in 108\,ms, and a total router GPU footprint under 800\,MB -- small enough to share a GPU with LLM serving and removing the need for a dedicated accelerator. Stage~1 targets AMD ROCm (NVIDIA GPUs already have FlashAttention via cuDNN); Stages~2 and~3 are hardware-agnostic. |
| 2026-03-13 | [RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization](http://arxiv.org/abs/2603.12639v1) | Ruicheng Zhang, Guangyu Chen et al. | Scalable Embodied AI faces fundamental constraints due to prohibitive costs and safety risks of real-world interaction. While Embodied World Models (EWMs) offer promise through imagined rollouts, existing approaches suffer from geometric hallucinations and lack unified optimization frameworks for practical policy improvement. We introduce RoboStereo, a symmetric dual-tower 4D world model that employs bidirectional cross-modal enhancement to ensure spatiotemporal geometric consistency and alleviate physics hallucinations. Building upon this high-fidelity 4D simulator, we present the first unified framework for world-model-based policy optimization: (1) Test-Time Policy Augmentation (TTPA) for pre-execution verification, (2) Imitative-Evolutionary Policy Learning (IEPL) leveraging visual perceptual rewards to learn from expert demonstrations, and (3) Open-Exploration Policy Learning (OEPL) enabling autonomous skill discovery and self-correction. Comprehensive experiments demonstrate RoboStereo achieves state-of-the-art generation quality, with our unified framework delivering >97% average relative improvement on fine-grained manipulation tasks. |
| 2026-03-13 | [Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](http://arxiv.org/abs/2603.12624v1) | Guodong Sun, Qihang Liang et al. | Accurate visual fault detection in freight trains remains a critical challenge for intelligent transportation system maintenance, due to complex operational environments, structurally repetitive components, and frequent occlusions or contaminations in safety-critical regions. Conventional instance segmentation methods based on convolutional neural networks and Transformers often suffer from poor generalization and limited boundary accuracy under such conditions. To address these challenges, we propose a lightweight self-prompted instance segmentation framework tailored for freight train fault detection. Our method leverages the Segment Anything Model by introducing a self-prompt generation module that automatically produces task-specific prompts, enabling effective knowledge transfer from foundation models to domain-specific inspection tasks. In addition, we adopt a Tiny Vision Transformer backbone to reduce computational cost, making the framework suitable for real-time deployment on edge devices in railway monitoring systems. We construct a domain-specific dataset collected from real-world freight inspection stations and conduct extensive evaluations. Experimental results show that our method achieves 74.6 $AP^{\text{box}}$ and 74.2 $AP^{\text{mask}}$ on the dataset, outperforming existing state-of-the-art methods in both accuracy and robustness while maintaining low computational overhead. This work offers a deployable and efficient vision solution for automated freight train inspection, demonstrating the potential of foundation model adaptation in industrial-scale fault diagnosis scenarios. Project page: https://github.com/MVME-HBUT/SAM_FTI-FDet.git |
| 2026-03-13 | [AgentDrift: Unsafe Recommendation Drift Under Tool Corruption Hidden by Ranking Metrics in LLM Agents](http://arxiv.org/abs/2603.12564v1) | Zekun Wu, Adriano Koshiyama et al. | Tool-augmented LLM agents increasingly serve as multi-turn advisors in high-stakes domains, yet their evaluation relies on ranking-quality metrics that measure what is recommended but not whether it is safe for the user. We introduce a paired-trajectory protocol that replays real financial dialogues under clean and contaminated tool-output conditions across seven LLMs (7B to frontier) and decomposes divergence into information-channel and memory-channel mechanisms. Across the seven models tested, we consistently observe the evaluation-blindness pattern: recommendation quality is largely preserved under contamination (utility preservation ratio approximately 1.0) while risk-inappropriate products appear in 65-93% of turns, a systematic safety failure poorly reflected by standard NDCG. Safety violations are predominantly information-channel-driven, emerge at the first contaminated turn, and persist without self-correction over 23-step trajectories; no agent across 1,563 contaminated turns explicitly questions tool-data reliability. Even narrative-only corruption (biased headlines, no numerical manipulation) induces significant drift while completely evading consistency monitors. A safety-penalized NDCG variant (sNDCG) reduces preservation ratios to 0.51-0.74, indicating that much of the evaluation gap becomes visible once safety is explicitly measured. These results motivate considering trajectory-level safety monitoring, beyond single-turn quality, for deployed multi-turn agents in high-stakes settings. |
| 2026-03-12 | [Xe gas bubble re-solution in U-10Mo nuclear fuel](http://arxiv.org/abs/2603.12491v1) | ATM Jahid Hasan, Linu Malakkal et al. | The U.S. High-Performance Research Reactor program aims to convert high-power research reactors from highly enriched uranium to low-enriched uranium using a monolithic U-10Mo fuel design. A critical aspect of U-10Mo fuel performance is fission gas bubble behavior. These bubbles grow by trapping gas atoms (particularly Xe) but can disintegrate via irradiation-induced "re-solution". The interplay between the trapping and re-solution rates governs bubble evolution, impacting fuel performance and safety. In this study, binary collision approximation (BCA) and molecular dynamics (MD) simulations were performed to quantify the Xe gas bubble re-solution rate in U-10Mo fuel. First, the energy loss of fission fragments (FFs) through electronic and nuclear stopping was evaluated. The effect of electronic stopping on re-solution was then analyzed using MD simulations coupled with the two-temperature model. Results indicate that thermal spikes generated by electronic stopping do not contribute to gas bubble re-solution in U-10Mo. To quantify re-solution due to nuclear stopping, BCA simulations of FFs in U-10Mo were performed to obtain the average FF incidence probability, energy, and angle as a function of distance from the FF origin. Subsequent simulations assessed FF--bubble interactions in U-10Mo for different FF energies and bubble radii. From these analyses, an overall re-solution rate $b$ was calculated at equilibrium bubble pressure per unit fission rate density, yielding values ranging from $4.4 \times 10^{-26}$ m$^3$/fission for the largest bubbles to $8.8 \times 10^{-25}$ m$^3$/fission for the smallest. The effect of bubble pressure on the re-solution rate was also evaluated, revealing an inverse relationship between the two. |
| 2026-03-12 | [Hunting CUDA Bugs at Scale with cuFuzz](http://arxiv.org/abs/2603.12485v1) | Mohamed Tarek Ibn ziad, Christos Kozyrakis | GPUs play an increasingly important role in modern software. However, the heterogeneous host-device execution model and expanding software stacks make GPU programs prone to memory-safety and concurrency bugs that evade static analysis. While fuzz-testing, combined with dynamic error checking tools, offers a plausible solution, it remains underutilized for GPUs. In this work, we identify three main obstacles limiting prior GPU fuzzing efforts: (1) kernel-level fuzzing leading to false positives, (2) lack of device-side coverage-guided feedback, and (3) incompatibility between coverage and sanitization tools. We present cuFuzz, the first CUDA-oriented fuzzer that makes GPU fuzzing practical by addressing these obstacles.   cuFuzz uses whole program fuzzing to avoid false positives from independently fuzzing device-side kernels. It leverages NVBit to instrument device-side instructions and merges the resultant coverage with compiler-based host coverage. Finally, cuFuzz decouples sanitization from coverage collection by executing host- and device-side sanitizers in separate processes. cuFuzz uncovers 43 previously unknown bugs (19 in commercial libraries) across 14 CUDA programs, including illegal memory accesses, uninitialized reads, and data races. cuFuzz achieves significantly more discovered edges and unique inputs compared to baseline approaches, especially on closed-source targets. Moreover, we quantify the execution time overheads of the different cuFuzz components and add persistent-mode support to improve the overall fuzzing throughput. Our results demonstrate that cuFuzz is an effective and deployable addition to the GPU testing toolbox. cuFuzz is publicly available at https://github.com/NVlabs/cuFuzz/. |
| 2026-03-12 | [Surg-R1: A Hierarchical Reasoning Foundation Model for Scalable and Interpretable Surgical Decision Support with Multi-Center Clinical Validation](http://arxiv.org/abs/2603.12430v1) | Jian Jiang, Chenxi Lin et al. | Surgical scene understanding demands not only accurate predictions but also interpretable reasoning that surgeons can verify against clinical expertise. However, existing surgical vision-language models generate predictions without reasoning chains, and general-purpose reasoning models fail on compositional surgical tasks without domain-specific knowledge. We present Surg-R1, a surgical Vision-Language Model that addresses this gap through hierarchical reasoning trained via a four-stage pipeline. Our approach introduces three key contributions: (1) a three-level reasoning hierarchy decomposing surgical interpretation into perceptual grounding, relational understanding, and contextual reasoning; (2) the largest surgical chain-of-thought dataset with 320,000 reasoning pairs; and (3) a four-stage training pipeline progressing from supervised fine-tuning to group relative policy optimization and iterative self-improvement. Evaluation on SurgBench, comprising six public benchmarks and six multi-center external validation datasets from five institutions, demonstrates that Surg-R1 achieves the highest Arena Score (64.9%) on public benchmarks versus Gemini 3.0 Pro (46.1%) and GPT-5.1 (37.9%), outperforming both proprietary reasoning models and specialized surgical VLMs on the majority of tasks spanning instrument localization, triplet recognition, phase recognition, action recognition, and critical view of safety assessment, with a 15.2 percentage point improvement over the strongest surgical baseline on external validation. |
| 2026-03-12 | [A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](http://arxiv.org/abs/2603.12421v1) | Hongyan Wei, Wael AbdAlmageed | Existing end-to-end autonomous driving models rely heavily on purely data-driven inductive reasoning. This "black-box" nature leads to a lack of interpretability and absolute safety guarantees in complex, long-tail scenarios. To overcome this bottleneck, we propose a novel neuro-symbolic trajectory planning framework that seamlessly integrates rigorous deductive reasoning into end-to-end neural networks. Specifically, our framework utilizes a Large Language Model (LLM) to dynamically extract scene rules and employs an Answer Set Programming (ASP) solver for deterministic logical arbitration, generating safe and traceable discrete driving decisions. To bridge the gap between discrete symbols and continuous trajectories, we introduce a decision-conditioned decoding mechanism that transforms high-level logical decisions into learnable embedding vectors, simultaneously constraining the planning query and the physical initial velocity of a differentiable Kinematic Bicycle Model (KBM). By combining KBM-generated physical baseline trajectories with neural residual corrections, our approach inherently guarantees kinematic feasibility while ensuring a high degree of transparency. On the nuScenes benchmark, our method comprehensively outperforms the state-of-the-art baseline MomAD, reducing the L2 mean error to 0.57 m, decreasing the collision rate to 0.075%, and optimizing trajectory prediction consistency (TPC) to 0.47 m. |
| 2026-03-12 | [SpectralGuard: Detecting Memory Collapse Attacks in State Space Models](http://arxiv.org/abs/2603.12414v1) | Davi Bonetto | State Space Models (SSMs) such as Mamba achieve linear-time sequence processing through input-dependent recurrence, but this mechanism introduces a critical safety vulnerability. We show that the spectral radius rho(A-bar) of the discretized transition operator governs effective memory horizon: when an adversary drives rho toward zero through gradient-based Hidden State Poisoning, memory collapses from millions of tokens to mere dozens, silently destroying reasoning capacity without triggering output-level alarms. We prove an Evasion Existence Theorem showing that for any output-only defense, adversarial inputs exist that simultaneously induce spectral collapse and evade detection, then introduce SpectralGuard, a real-time monitor that tracks spectral stability across all model layers. SpectralGuard achieves F1=0.961 against non-adaptive attackers and retains F1=0.842 under the strongest adaptive setting, with sub-15ms per-token latency. Causal interventions and cross-architecture transfer to hybrid SSM-Attention systems confirm that spectral monitoring provides a principled, deployable safety layer for recurrent foundation models. |
| 2026-03-12 | [Budget-Sensitive Discovery Scoring: A Formally Verified Framework for Evaluating AI-Guided Scientific Selection](http://arxiv.org/abs/2603.12349v1) | Abhinaba Basu, Pavan Chakraborty | Scientific discovery increasingly relies on AI systems to select candidates for expensive experimental validation, yet no principled, budget-aware evaluation framework exists for comparing selection strategies -- a gap intensified by large language models (LLMs), which generate plausible scientific proposals without reliable downstream evaluation. We introduce the Budget-Sensitive Discovery Score (BSDS), a formally verified metric -- 20 theorems machine-checked by the Lean 4 proof assistant -- that jointly penalizes false discoveries (lambda-weighted FDR) and excessive abstention (gamma-weighted coverage gap) at each budget level. Its budget-averaged form, the Discovery Quality Score (DQS), provides a single summary statistic that no proposer can inflate by performing well at a cherry-picked budget.   As a case study, we apply BSDS/DQS to: do LLMs add marginal value to an existing ML pipeline for drug discovery candidate selection? We evaluate 39 proposers -- 11 mechanistic variants, 14 zero-shot LLM configurations, and 14 few-shot LLM configurations -- using SMILES representations on MoleculeNet HIV (41,127 compounds, 3.5% active, 1,000 bootstrap replicates) under both random and scaffold splits. Three findings emerge. First, the simple RF-based Greedy-ML proposer achieves the best DQS (-0.046), outperforming all MLP variants and LLM configurations. Second, no LLM surpasses the Greedy-ML baseline under zero-shot or few-shot evaluation on HIV or Tox21, establishing that LLMs provide no marginal value over an existing trained classifier. Third, the proposer hierarchy generalizes across five MoleculeNet benchmarks spanning 0.18%-46.2% prevalence, a non-drug AV safety domain, and a 9x7 grid of penalty parameters (tau >= 0.636, mean tau = 0.863). The framework applies to any setting where candidates are selected under budget constraints and asymmetric error costs. |

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



