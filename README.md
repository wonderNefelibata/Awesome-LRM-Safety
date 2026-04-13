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
| 2026-04-10 | [Large Language Models Generate Harmful Content Using a Distinct, Unified Mechanism](http://arxiv.org/abs/2604.09544v1) | Hadas Orgad, Boyi Wei et al. | Large language models (LLMs) undergo alignment training to avoid harmful behaviors, yet the resulting safeguards remain brittle: jailbreaks routinely bypass them, and fine-tuning on narrow domains can induce ``emergent misalignment'' that generalizes broadly. Whether this brittleness reflects a fundamental lack of coherent internal organization for harmfulness remains unclear. Here we use targeted weight pruning as a causal intervention to probe the internal organization of harmfulness in LLMs. We find that harmful content generation depends on a compact set of weights that are general across harm types and distinct from benign capabilities. Aligned models exhibit a greater compression of harm generation weights than unaligned counterparts, indicating that alignment reshapes harmful representations internally--despite the brittleness of safety guardrails at the surface level. This compression explains emergent misalignment: if weights of harmful capabilities are compressed, fine-tuning that engages these weights in one domain can trigger broad misalignment. Consistent with this, pruning harm generation weights in a narrow domain substantially reduces emergent misalignment. Notably, LLMs harmful generation capability is dissociated from how they recognize and explain such content. Together, these results reveal a coherent internal structure for harmfulness in LLMs that may serve as a foundation for more principled approaches to safety. |
| 2026-04-10 | [Policy-Aware Edge LLM-RAG Framework for Internet of Battlefield Things Mission Orchestration](http://arxiv.org/abs/2604.09493v1) | Om Solanki, Lopamudra Praharaj et al. | Large Language Models (LLMs) offer a promising interface for intent-driven control of autonomous cyber-physical systems, but their direct use in mission-critical Internet of Battlefield Things (IoBT) environments raises significant safety, reliability, and policy-compliance concerns. This paper presents a Policy-Aware Large Language Model Retrieval-Augmented Generation (referred as PA-LLM-RAG), an edge-deployed LLM orchestration framework for IoBT mission control that integrates retrieval-augmented reasoning and independent command verification. The proposed PA-LLM-RAG framework combines a lightweight retrieval module that grounds decisions in operational policies and telemetry with a locally hosted LLM for mission planning and a secondary JudgeLLM for validating user generated commands prior to execution.   To evaluate PA-LLM-RAG, we implement a simulated IoBT environment using RoboDK and assess four open-source LLMs across controlled mission scenarios of increasing complexity, including baseline operations, threat detection, coverage recovery, multi-event coordination, and policy-violation requests. Experimental results demonstrate that the framework effectively detects policy-violating commands while maintaining low-latency response suitable for edge deployment. Gemma-2B achieving the highest overall reliability with 4.17 sec latency and 100% success rate. The findings highlight a clear tradeoff between reasoning capacity and responsiveness across models and show that combining deterministic safeguards with JudgeLLM verification significantly improves reliability in LLM-driven IoBT orchestration. |
| 2026-04-10 | [SafeMind: A Risk-Aware Differentiable Control Framework for Adaptive and Safe Quadruped Locomotion](http://arxiv.org/abs/2604.09474v1) | Zukun Zhang, Kai Shu et al. | Learning-based quadruped controllers achieve impressive agility but typically lack formal safety guarantees under model uncertainty, perception noise, and unstructured contact conditions. We introduce SafeMind, a differentiable stochastic safety-control framework that unifies probabilistic Control Barrier Functions with semantic context understanding and meta-adaptive risk calibration. SafeMind explicitly models epistemic and aleatoric uncertainty through a variance-aware barrier constraint embedded in a differentiable quadratic program, thereby preserving gradient flow for end-to-end training. A semantics-to-constraint encoder modulates safety margins using perceptual or language cues, while a meta-adaptive learner continuously adjusts risk sensitivity across environments. We provide theoretical conditions for probabilistic forward invariance, feasibility, and stability under stochastic dynamics. SafeMind is deployed on Unitree A1 and ANYmal C at 200~Hz and validated across 12 terrain types, dynamic obstacles, morphology perturbations, and semantically defined tasks. Experiments show that SafeMind reduces safety violations by 3--10x and energy consumption by 10--15% relative to state-of-the-art CBF, MPC, and hybrid RL baselines, while maintaining real-time control performance. |
| 2026-04-10 | [SafeAdapt: Provably Safe Policy Updates in Deep Reinforcement Learning](http://arxiv.org/abs/2604.09452v1) | Maksim Anisimov, Francesco Belardinelli et al. | Safety guarantees are a prerequisite to the deployment of reinforcement learning (RL) agents in safety-critical tasks. Often, deployment environments exhibit non-stationary dynamics or are subject to changing performance goals, requiring updates to the learned policy. This leads to a fundamental challenge: how to update an RL policy while preserving its safety properties on previously encountered tasks? The majority of current approaches either do not provide formal guarantees or verify policy safety only a posteriori. We propose a novel a priori approach to safe policy updates in continual RL by introducing the Rashomon set: a region in policy parameter space certified to meet safety constraints within the demonstration data distribution. We then show that one can provide formal, provable guarantees for arbitrary RL algorithms used to update a policy by projecting their updates onto the Rashomon set. Empirically, we validate this approach across grid-world navigation environments (Frozen Lake and Poisoned Apple) where we guarantee an a priori provably deterministic safety on the source task during downstream adaptation. In contrast, we observe that regularisation-based baselines experience catastrophic forgetting of safety constraints while our approach enables strong adaptation with provable guarantees that safety is preserved. |
| 2026-04-10 | [EGLOCE: Training-Free Energy-Guided Latent Optimization for Concept Erasure](http://arxiv.org/abs/2604.09405v1) | Junyeong Ahn, Seojin Yoon et al. | As text-to-image diffusion models grow increasingly prevalent, the ability to remove specific concepts-mostly explicit content and many copyrighted characters or styles-has become essential for safety and compliance. Existing unlearning approaches often require costly re-training, modify parameters at the cost of degradation of unrelated concept fidelity, or depend on indirect inference-time adjustment that compromise the effectiveness of concept erasure. Inspired by the success of energy-guided sampling for preservation of the condition of diffusion models, we introduce Energy-Guided Latent Optimization for Concept Erasure (EGLOCE), a training-free approach that removes unwanted concepts by re-directing noisy latent during inference. Our method employs a dual-objective framework: a repulsion energy that steers generation away from target concepts via gradient descent in latent space, and a retention energy that preserves semantic alignment to the original prompt. Combined with previous approaches that either require erroneous modified model weights or provide weak inference-time guidance, EGLOCE operates entirely at inference and enhances erasure performance, enabling plug-and-play integration. Extensive experiments demonstrate that EGLOCE improves concept removal while maintaining image quality and prompt alignment across baselines, even with adversarial attacks. To the best of our knowledge, our work is the first to establish a new paradigm for safe and controllable image generation through dual energy-based guidance during sampling. |
| 2026-04-10 | [Multimodal Anomaly Detection for Human-Robot Interaction](http://arxiv.org/abs/2604.09326v1) | Guilherme Ribeiro, Iordanis Antypas et al. | Ensuring safety and reliability in human-robot interaction (HRI) requires the timely detection of unexpected events that could lead to system failures or unsafe behaviours. Anomaly detection thus plays a critical role in enabling robots to recognize and respond to deviations from normal operation during collaborative tasks. While reconstruction models have been actively explored in HRI, approaches that operate directly on feature vectors remain largely unexplored. In this work, we propose MADRI, a framework that first transforms video streams into semantically meaningful feature vectors before performing reconstruction-based anomaly detection. Additionally, we augment these visual feature vectors with the robot's internal sensors' readings and a Scene Graph, enabling the model to capture both external anomalies in the visual environment and internal failures within the robot itself. To evaluate our approach, we collected a custom dataset consisting of a simple pick-and-place robotic task under normal and anomalous conditions. Experimental results demonstrate that reconstruction on vision-based feature vectors alone is effective for detecting anomalies, while incorporating other modalities further improves detection performance, highlighting the benefits of multimodal feature reconstruction for robust anomaly detection in human-robot collaboration. |
| 2026-04-10 | [VAGNet: Vision-based accident anticipation with global features](http://arxiv.org/abs/2604.09305v1) | Vipooshan Vipulananthan, Charith D. Chitraranjan | Traffic accidents are a leading cause of fatalities and injuries across the globe. Therefore, the ability to anticipate hazardous situations in advance is essential. Automated accident anticipation enables timely intervention through driver alerts and collision avoidance maneuvers, forming a key component of advanced driver assistance systems. In autonomous driving, such predictive capabilities support proactive safety behaviors, such as initiating defensive driving and human takeover when required. Using dashcam video as input offers a cost-effective solution, but it is challenging due to the complexity of real-world driving scenes. Accident anticipation systems need to operate in real-time. However, current methods involve extracting features from each detected object, which is computationally intensive. We propose VAGNet, a deep neural network that learns to predict accidents from dash-cam video using global features of traffic scenes without requiring explicit object-level features. The network consists of transformer and graph modules, and we use the vision foundation model VideoMAE-V2 for global feature extraction. Experiments on four benchmark datasets (DAD, DoTA, DADA, and Nexar) show that our method anticipates accidents with higher average precision and mean time-to-accident while being computationally more efficient compared to existing methods. |
| 2026-04-10 | [EthicMind: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](http://arxiv.org/abs/2604.09265v1) | Jiawen Deng, Wei Li et al. | Intelligent dialogue systems are increasingly deployed in emotionally and ethically sensitive settings, where failures in either emotional attunement or ethical judgment can cause significant harm. Existing dialogue models typically address empathy and ethical safety in isolation, and often fail to adapt their behavior as ethical risk and user emotion evolve across multi-turn interactions. We formulate ethical-emotional alignment in dialogue as an explicit turn-level decision problem, and propose \textsc{EthicMind}, a risk-aware framework that implements this formulation in multi-turn dialogue at inference time. At each turn, \textsc{EthicMind} jointly analyzes ethical risk signals and user emotion, plans a high-level response strategy, and generates context-sensitive replies that balance ethical guidance with emotional engagement, without requiring additional model training. To evaluate alignment behavior under ethically complex interactions, we introduce a risk-stratified, multi-turn evaluation protocol with a context-aware user simulation procedure. Experimental results show that \textsc{EthicMind} achieves more consistent ethical guidance and emotional engagement than competitive baselines, particularly in high-risk and morally ambiguous scenarios. |
| 2026-04-10 | [Mosaic: Multimodal Jailbreak against Closed-Source VLMs via Multi-View Ensemble Optimization](http://arxiv.org/abs/2604.09253v1) | Yuqin Lan, Gen Li et al. | Vision-Language Models (VLMs) are powerful but remain vulnerable to multimodal jailbreak attacks. Existing attacks mainly rely on either explicit visual prompt attacks or gradient-based adversarial optimization. While the former is easier to detect, the latter produces subtle perturbations that are less perceptible, but is usually optimized and evaluated under homogeneous open-source surrogate-target settings, leaving its effectiveness on commercial closed-source VLMs under heterogeneous settings unclear. To examine this issue, we study different surrogate-target settings and observe a consistent gap between homogeneous and heterogeneous settings, a phenomenon we term surrogate dependency. Motivated by this finding, we propose Mosaic, a Multi-view ensemble optimization framework for multimodal jailbreak against closed-source VLMs, which alleviates surrogate dependency under heterogeneous surrogate-target settings by reducing over-reliance on any single surrogate model and visual view. Specifically, Mosaic incorporates three core components: a Text-Side Transformation module, which perturbs refusal-sensitive lexical patterns; a Multi-View Image Optimization module, which updates perturbations under diverse cropped views to avoid overfitting to a single visual view; and a Surrogate Ensemble Guidance module, which aggregates optimization signals from multiple surrogate VLMs to reduce surrogate-specific bias. Extensive experiments on safety benchmarks demonstrate that Mosaic achieves state-of-the-art Attack Success Rate and Average Toxicity against commercial closed-source VLMs. |
| 2026-04-10 | [LandSAR: Visceralizing Landslide Data for Enhanced Situational Awareness in Immersive Analytics](http://arxiv.org/abs/2604.09241v1) | Wong Kam-Kwai, Yi-Lin Ye et al. | Landslides pose a significant threat to public safety, but their dynamic processes are difficult to analyze from post-event observation alone. Computational simulation is therefore essential, but it generates vast, abstract datasets that create a cognitive gap between the analyst and the real-world, physical terrain. While Immersive Analytics (IA) begins to bridge this gap by visualizing data in 3D, we explore how these systems evolve beyond abstract data and integrate data visceralization to enhance Situational Awareness (SA). We present LandSAR, an immersive analytics system that enhances SA for landslide analysis by visceralizing landslide data through integrated simulations and visualizations. LandSAR supports real-time simulations of landslide dynamics, prevention strategies, and climate impacts, enabling multi-perspective what-if analyses. The system uses 3D-printed terrain models as tangible interfaces to facilitate haptic feedback and enable gesture-based exploration, allowing for intuitive geographical perception. Expert interviews and workshops demonstrate that LandSAR effectively improves SA and engagement. |
| 2026-04-10 | [Unreal Thinking: Chain-of-Thought Hijacking via Two-stage Backdoor](http://arxiv.org/abs/2604.09235v1) | Wenhan Chang, Tianqing Zhu et al. | Large Language Models (LLMs) are increasingly deployed in settings where Chain-of-Thought (CoT) is interpreted by users. This creates a new safety risk: attackers may manipulate the model's observable CoT to make malicious behaviors. In open-weight ecosystems, such manipulation can be embedded in lightweight adapters that are easy to distribute and attach to base models. In practice, persistent CoT hijacking faces three main challenges: the difficulty of directly hijacking CoT tokens within one continuous long CoT-output sequence while maintaining stable downstream outputs, the scarcity of malicious CoT data, and the instability of naive backdoor injection methods. To address the data scarcity issue, we propose Multiple Reverse Tree Search (MRTS), a reverse synthesis procedure that constructs output-aligned CoTs from prompt-output pairs without directly eliciting malicious CoTs from aligned models. Building on MRTS, we introduce Two-stage Backdoor Hijacking (TSBH), which first induces a trigger-conditioned mismatch between intermediate CoT and malicious outputs, and then fine-tunes the model on MRTS-generated CoTs that have lower embedding distance to the malicious outputs, thereby ensuring stronger semantic similarity. Experiments across multiple open-weight models demonstrate that our method successfully induces trigger-activated CoT hijacking while maintaining a quantifiable distinction between hijacked and baseline states under our evaluation framework. We further explore a reasoning-based mitigation approach and release a safety-reasoning dataset to support future research on safety-aware and reliable reasoning. Our code is available at https://github.com/ChangWenhan/TSBH_official. |
| 2026-04-10 | [GRM: Utility-Aware Jailbreak Attacks on Audio LLMs via Gradient-Ratio Masking](http://arxiv.org/abs/2604.09222v1) | Yunqiang Wang, Hengyuan Na et al. | Audio large language models (ALLMs) enable rich speech-text interaction, but they also introduce jailbreak vulnerabilities in the audio modality. Existing audio jailbreak methods mainly optimize jailbreak success while overlooking utility preservation, as reflected in transcription quality and question answering performance. In practice, stronger attacks often come at the cost of degraded utility. To study this trade-off, we revisit existing attacks by varying their perturbation coverage in the frequency domain, from partial-band to full-band, and find that broader frequency coverage does not necessarily improve jailbreak performance, while utility consistently deteriorates. This suggests that concentrating perturbation on a subset of bands can yield a better attack-utility trade-off than indiscriminate full-band coverage. Based on this insight, we propose GRM, a utility-aware frequency-selective jailbreak framework. It ranks Mel bands by their attack contribution relative to utility sensitivity, perturbs only a selected subset of bands, and learns a reusable universal perturbation under a semantic-preservation objective. Experiments on four representative ALLMs show that GRM achieves an average Jailbreak Success Rate (JSR) of 88.46% while providing a better attack-utility trade-off than representative baselines. These results highlight the potential of frequency-selective perturbation for better balancing attack effectiveness and utility preservation in audio jailbreak. Content Warning: This paper includes harmful query examples and unsafe model responses. |
| 2026-04-10 | [Do LLMs Follow Their Own Rules? A Reflexive Audit of Self-Stated Safety Policies](http://arxiv.org/abs/2604.09189v1) | Avni Mittal | LLMs internalize safety policies through RLHF, yet these policies are never formally specified and remain difficult to inspect. Existing benchmarks evaluate models against external standards but do not measure whether models understand and enforce their own stated boundaries. We introduce the Symbolic-Neural Consistency Audit (SNCA), a framework that (1) extracts a model's self-stated safety rules via structured prompts, (2) formalizes them as typed predicates (Absolute, Conditional, Adaptive), and (3) measures behavioral compliance via deterministic comparison against harm benchmarks. Evaluating four frontier models across 45 harm categories and 47,496 observations reveals systematic gaps between stated policy and observed behavior: models claiming absolute refusal frequently comply with harmful prompts, reasoning models achieve the highest self-consistency but fail to articulate policies for 29% of categories, and cross-model agreement on rule types is remarkably low (11%). These results demonstrate that the gap between what LLMs say and what they do is measurable and architecture-dependent, motivating reflexive consistency audits as a complement to behavioral benchmarks. |
| 2026-04-10 | [CORA: Conformal Risk-Controlled Agents for Safeguarded Mobile GUI Automation](http://arxiv.org/abs/2604.09155v1) | Yushi Feng, Junye Du et al. | Graphical user interface (GUI) agents powered by vision language models (VLMs) are rapidly moving from passive assistance to autonomous operation. However, this unrestricted action space exposes users to severe and irreversible financial, privacy or social harm. Existing safeguards rely on prompt engineering, brittle heuristics and VLM-as-critic lack formal verification and user-tunable guarantees. We propose CORA (COnformal Risk-controlled GUI Agent), a post-policy, pre-action safeguarding framework that provides statistical guarantees on harmful executed actions. CORA reformulates safety as selective action execution: we train a Guardian model to estimate action-conditional risk for each proposed step. Rather than thresholding raw scores, we leverage Conformal Risk Control to calibrate an execute/abstain boundary that satisfies a user-specified risk budget and route rejected actions to a trainable Diagnostician model, which performs multimodal reasoning over rejected actions to recommend interventions (e.g., confirm, reflect, or abort) to minimize user burden. A Goal-Lock mechanism anchors assessment to a clarified, frozen user intent to resist visual injection attacks. To rigorously evaluate this paradigm, we introduce Phone-Harm, a new benchmark of mobile safety violations with step-level harm labels under real-world settings. Experiments on Phone-Harm and public benchmarks against diverse baselines validate that CORA improves the safety--helpfulness--interruption Pareto frontier, offering a practical, statistically grounded safety paradigm for autonomous GUI execution. Code and benchmark are available at cora-agent.github.io. |
| 2026-04-10 | [Hierarchical Alignment: Enforcing Hierarchical Instruction-Following in LLMs through Logical Consistency](http://arxiv.org/abs/2604.09075v1) | Shu Yang, Zihao Zhou et al. | Large language models increasingly operate under multiple instructions from heterogeneous sources with different authority levels, including system policies, user requests, tool outputs, and retrieved context. While prior work on instruction hierarchy highlights the importance of respecting instruction priorities, it mainly focuses on adversarial attacks and overlooks the benign but common instruction conflicts that arise in real-world applications. In such settings, models must not only avoid security violations but also preserve task utility and behavioral consistency when instructions partially or implicitly conflict. We propose Neuro-Symbolic Hierarchical Alignment (NSHA) for hierarchical instruction-following by explicitly modeling and enforcing instruction priorities. At inference time, we introduce solver-guided reasoning that formulates instruction resolution as a constraint satisfaction problem, enabling the model to derive a maximally consistent set of applicable instructions under hierarchical constraints. At training time, NSHA distills solver-based decisions into model parameters using automatically constructed supervision. We evaluate our approach on rule following, task execution, tool use, and safety, covering both single-turn and multi-turn interactions, and show that NSHA significantly improves performance under such conflicts while maintaining competitive utility in reference settings. |
| 2026-04-10 | [Learning Vision-Language-Action World Models for Autonomous Driving](http://arxiv.org/abs/2604.09059v1) | Guoqing Wang, Pin Tang et al. | Vision-Language-Action (VLA) models have recently achieved notable progress in end-to-end autonomous driving by integrating perception, reasoning, and control within a unified multimodal framework. However, they often lack explicit modeling of temporal dynamics and global world consistency, which limits their foresight and safety. In contrast, world models can simulate plausible future scenes but generally struggle to reason about or evaluate the imagined future they generate. In this work, we present VLA-World, a simple yet effective VLA world model that unifies predictive imagination with reflective reasoning to improve driving foresight. VLA-World first uses an action-derived feasible trajectory to guide the generation of the next-frame image, capturing rich spatial and temporal cues that describe how the surrounding environment evolves. The model then reasons over this self-generated future imagined frame to refine the predicted trajectory, achieving higher performance and better interpretability. To support this pipeline, we curate nuScenes-GR-20K, a generative reasoning dataset derived from nuScenes, and employ a three-stage training strategy that includes pretraining, supervised fine-tuning, and reinforcement learning. Extensive experiments demonstrate that VLA-World consistently surpasses state-of-the-art VLA and world-model baselines on both planning and future-generation benchmarks. Project page: https://vlaworld.github.io |
| 2026-04-10 | [Conversations Risk Detection LLMs in Financial Agents via Multi-Stage Generative Rollout](http://arxiv.org/abs/2604.09056v1) | Xiaotong Jiang, Jun Wu | With the rapid adoption of large language models (LLMs) in financial service scenarios, dialogue security detection under high regulatory risk presents significant challenges. Existing methods mainly rely on single-dimensional semantic judgments or fixed rules, making them inadequate for handling multi-turn semantic evolution and complex regulatory clauses; moreover, they lack models specifically designed for financial security detection. To address these issues, this paper proposes FinSec, a four-tier security detection framework for financial agent. FinSec enables structured, interpretable, and end-to-end identification of actual financial risks, incorporating suspicious behavior pattern analysis, delayed risk and adversarial inference, semantic security analysis, and integrated risk-based decision-making. Notably, FinSec significantly enhances the robustness of high-risk dialogue detection while maintaining model utility. Experimental results demonstrate FinSec's leading performance. In terms of overall detection capability, FinSec achieves an F1 score of 90.13%, improving upon baseline models by 6--14 percentage points; its ASR is reduced to 9.09%, markedly lowering the probability of unsafe outputs; and the AUPRC increases to 0.9189 -- an approximate 9.7% gain over general frameworks. Additionally, in balancing utility and safety, FinSec obtains a composite score of 0.9098, delivering robust and efficient protection for financial agent dialogues. |
| 2026-04-10 | [Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Images via Visual Prompt Injection](http://arxiv.org/abs/2604.09024v1) | Zedian Shao, Hongbin Liu et al. | Multi-modal large language models (MLLMs) have emerged as powerful tools for analyzing Internet-scale image data, offering significant benefits but also raising critical safety and societal concerns. In particular, open-weight MLLMs may be misused to extract sensitive information from personal images at scale, such as identities, locations, or other private details. In this work, we propose ImageProtector, a user-side method that proactively protects images before sharing by embedding a carefully crafted, nearly imperceptible perturbation that acts as a visual prompt injection attack on MLLMs. As a result, when an adversary analyzes a protected image with an MLLM, the MLLM is consistently induced to generate a refusal response such as "I'm sorry, I can't help with that request." We empirically demonstrate the effectiveness of ImageProtector across six MLLMs and four datasets. Additionally, we evaluate three potential countermeasures, Gaussian noise, DiffPure, and adversarial training, and show that while they partially mitigate the impact of ImageProtector, they simultaneously degrade model accuracy and/or efficiency. Our study focuses on the practically important setting of open-weight MLLMs and large-scale automated image analysis, and highlights both the promise and the limitations of perturbation-based privacy protection. |
| 2026-04-10 | [Synthesizing Safety in Infinite-Horizon Optimal Control for Disturbed High-Relative-Degree Systems via Barrier-Regulating Auxiliary Variables](http://arxiv.org/abs/2604.09004v1) | Zhanglin Shangguan, Wei Xiao et al. | Optimal stabilization of safety-critical nonlinear systems requires balancing long-term performance and strict safety constraints. Existing quadratic-programming-based control barrier function (CBF) safety filters are point-wise and may exhibit myopic behavior and local trapping when the safeguarding action conflicts with the nominal optimal control. This paper develops a safety-aware infinite-horizon optimal control framework by embedding a barrier-Lyapunov function (BLF)-based safeguarding action into the system dynamics and introducing a barrier-regulating auxiliary variable, thereby reformulating the original constrained problem as an unconstrained one on an extended state space. To mitigate local trapping, we introduce an adaptive alignment-conditioned tangential excitation orthogonal to the safety direction, with activation adaptively modulated by the degree of directional alignment between the nominal and safeguarding controllers, and incorporate it as an admissible $\mathcal{L}2$ disturbance in an $H\infty$ formulation. For high-relative-degree systems under disturbances, we further augment the recursive high-order safe-set construction with barrier compensation terms to obtain a high-order BLF and formulate an adversarial disturbance attenuation problem, which is approximately solved via safe-exploration-enhanced online critic learning. Simulations demonstrate reduced local trapping, improved safety--performance trade-offs, and safe operation under disturbances. |
| 2026-04-10 | [PilotBench: A Benchmark for General Aviation Agents with Safety Constraints](http://arxiv.org/abs/2604.08987v1) | Yalun Wu, Haotian Liu et al. | As Large Language Models (LLMs) advance toward embodied AI agents operating in physical environments, a fundamental question emerges: can models trained on text corpora reliably reason about complex physics while adhering to safety constraints? We address this through PilotBench, a benchmark evaluating LLMs on safety-critical flight trajectory and attitude prediction. Built from 708 real-world general aviation trajectories spanning nine operationally distinct flight phases with synchronized 34-channel telemetry, PilotBench systematically probes the intersection of semantic understanding and physics-governed prediction through comparative analysis of LLMs and traditional forecasters. We introduce Pilot-Score, a composite metric balancing 60% regression accuracy with 40% instruction adherence and safety compliance. Comparative evaluation across 41 models uncovers a Precision-Controllability Dichotomy: traditional forecasters achieve superior MAE of 7.01 but lack semantic reasoning capabilities, while LLMs gain controllability with 86--89% instruction-following at the cost of 11--14 MAE precision. Phase-stratified analysis further exposes a Dynamic Complexity Gap-LLM performance degrades sharply in high-workload phases such as Climb and Approach, suggesting brittle implicit physics models. These empirical discoveries motivate hybrid architectures combining LLMs' symbolic reasoning with specialized forecasters' numerical precision. PilotBench provides a rigorous foundation for advancing embodied AI in safety-constrained domains. |

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



