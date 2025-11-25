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
| 2025-11-24 | [Data driven synthesis of provable invariant sets via stochastically sampled data](http://arxiv.org/abs/2511.19421v1) | Amy K. Strong, Ali Kashani et al. | Positive invariant (PI) sets are essential for ensuring safety, i.e. constraint adherence, of dynamical systems. With the increasing availability of sampled data from complex (and often unmodeled) systems, it is advantageous to leverage these data sets for PI set synthesis. This paper uses data driven geometric conditions of invariance to synthesize PI sets from data. Where previous data driven, set-based approaches to PI set synthesis used deterministic sampling schemes, this work instead synthesizes PI sets from any pre-collected data sets. Beyond a data set and Lipschitz continuity, no additional information about the system is needed. A tree data structure is used to partition the space and select samples used to construct the PI set, while Lipschitz continuity is used to provide deterministic guarantees of invariance. Finally, probabilistic bounds are given on the number of samples needed for the algorithm to determine of a certain volume. |
| 2025-11-24 | [Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration](http://arxiv.org/abs/2511.19417v1) | James Y. Huang, Sheng Zhang et al. | Large Language Models (LLMs) have demonstrated remarkable capabilities in challenging, knowledge-intensive reasoning tasks. However, extending LLMs to perceive and reason over a new modality (e.g., vision), often requires costly development of large-scale vision language models (VLMs) with LLMs as backbones. Smaller VLMs are more efficient and adaptable but often lack the broad knowledge and reasoning capabilities of frontier LLMs. In this work, we propose BeMyEyes, a modular, multi-agent framework for extending LLMs to multimodal reasoning by orchestrating collaboration between efficient, adaptable VLMs as perceivers and powerful LLMs as reasoners through conversations. We then introduce a data synthesis and supervised fine-tuning pipeline to train the perceiver agent to effectively collaborate with the reasoner agent. By combining the complementary strengths of perception and reasoning agents, BeMyEyes avoids the need for training large-scale multimodal models, preserves the generalization and reasoning capabilities of LLMs, and allows flexible extension to new domains and modalities. Experiments show that our framework unlocks the multimodal reasoning capabilities for LLMs, enabling a lightweight and fully open-source solution, i.e. equipping text-only DeepSeek-R1 with Qwen2.5-VL-7B perceiver, to outperform large-scale proprietary VLMs such as GPT-4o on a wide range of knowledge-intensive multimodal tasks. These results demonstrate the effectiveness, modularity, and scalability of our multi-agent approach for building future multimodal reasoning systems. |
| 2025-11-24 | [Normative active inference: A numerical proof of principle for a computational and economic legal analytic approach to AI governance](http://arxiv.org/abs/2511.19334v1) | Axel Constant, Mahault Albarracin et al. | This paper presents a computational account of how legal norms can influence the behavior of artificial intelligence (AI) agents, grounded in the active inference framework (AIF) that is informed by principles of economic legal analysis (ELA). The ensuing model aims to capture the complexity of human decision-making under legal constraints, offering a candidate mechanism for agent governance in AI systems, that is, the (auto)regulation of AI agents themselves rather than human actors in the AI industry. We propose that lawful and norm-sensitive AI behavior can be achieved through regulation by design, where agents are endowed with intentional control systems, or behavioral safety valves, that guide real-time decisions in accordance with normative expectations. To illustrate this, we simulate an autonomous driving scenario in which an AI agent must decide when to yield the right of way by balancing competing legal and pragmatic imperatives. The model formalizes how AIF can implement context-dependent preferences to resolve such conflicts, linking this mechanism to the conception of law as a scaffold for rational decision-making under uncertainty. We conclude by discussing how context-dependent preferences could function as safety mechanisms for autonomous agents, enhancing lawful alignment and risk mitigation in AI governance. |
| 2025-11-24 | [Learning to Reason: Training LLMs with GPT-OSS or DeepSeek R1 Reasoning Traces](http://arxiv.org/abs/2511.19333v1) | Shaltiel Shmidman, Asher Fredman et al. | Test-time scaling, which leverages additional computation during inference to improve model accuracy, has enabled a new class of Large Language Models (LLMs) that are able to reason through complex problems by understanding the goal, turning this goal into a plan, working through intermediate steps, and checking their own work before answering . Frontier large language models with reasoning capabilities, such as DeepSeek-R1 and OpenAI's gpt-oss, follow the same procedure when solving complex problems by generating intermediate reasoning traces before giving the final answer. Today, these models are being increasingly used to generate reasoning traces that serve as high-quality supervised data for post-training of small and medium-sized language models to teach reasoning capabilities without requiring expensive human curation. In this work, we compare the performance of medium-sized LLMs on Math problems after post-training on two kinds of reasoning traces. We compare the impact of reasoning traces generated by DeepSeek-R1 and gpt-oss LLMs in terms of accuracy and inference efficiency. |
| 2025-11-24 | [Open-weight genome language model safeguards: Assessing robustness via adversarial fine-tuning](http://arxiv.org/abs/2511.19299v1) | James R. M. Black, Moritz S. Hanke et al. | Novel deep learning architectures are increasingly being applied to biological data, including genetic sequences. These models, referred to as genomic language mod- els (gLMs), have demonstrated impressive predictive and generative capabilities, raising concerns that such models may also enable misuse, for instance via the generation of genomes for human-infecting viruses. These concerns have catalyzed calls for risk mitigation measures. The de facto mitigation of choice is filtering of pretraining data (i.e., removing viral genomic sequences from training datasets) in order to limit gLM performance on virus-related tasks. However, it is not currently known how robust this approach is for securing open-source models that can be fine-tuned using sensitive pathogen data. Here, we evaluate a state-of-the-art gLM, Evo 2, and perform fine-tuning using sequences from 110 harmful human-infecting viruses to assess the rescue of misuse-relevant predictive capabilities. The fine- tuned model exhibited reduced perplexity on unseen viral sequences relative to 1) the pretrained model and 2) a version fine-tuned on bacteriophage sequences. The model fine-tuned on human-infecting viruses also identified immune escape variants from SARS-CoV-2 (achieving an AUROC of 0.6), despite having no expo- sure to SARS-CoV-2 sequences during fine-tuning. This work demonstrates that data exclusion might be circumvented by fine-tuning approaches that can, to some degree, rescue misuse-relevant capabilities of gLMs. We highlight the need for safety frameworks for gLMs and outline further work needed on evaluations and mitigation measures to enable the safe deployment of gLMs. |
| 2025-11-24 | [Medusa: Cross-Modal Transferable Adversarial Attacks on Multimodal Medical Retrieval-Augmented Generation](http://arxiv.org/abs/2511.19257v1) | Yingjia Shang, Yi Liu et al. | With the rapid advancement of retrieval-augmented vision-language models, multimodal medical retrieval-augmented generation (MMed-RAG) systems are increasingly adopted in clinical decision support. These systems enhance medical applications by performing cross-modal retrieval to integrate relevant visual and textual evidence for tasks, e.g., report generation and disease diagnosis. However, their complex architecture also introduces underexplored adversarial vulnerabilities, particularly via visual input perturbations. In this paper, we propose Medusa, a novel framework for crafting cross-modal transferable adversarial attacks on MMed-RAG systems under a black-box setting. Specifically, Medusa formulates the attack as a perturbation optimization problem, leveraging a multi-positive InfoNCE loss (MPIL) to align adversarial visual embeddings with medically plausible but malicious textual targets, thereby hijacking the retrieval process. To enhance transferability, we adopt a surrogate model ensemble and design a dual-loop optimization strategy augmented with invariant risk minimization (IRM). Extensive experiments on two real-world medical tasks, including medical report generation and disease diagnosis, demonstrate that Medusa achieves over 90% average attack success rate across various generation models and retrievers under appropriate parameter configuration, while remaining robust against four mainstream defenses, outperforming state-of-the-art baselines. Our results reveal critical vulnerabilities in the MMed-RAG systems and highlight the necessity of robustness benchmarking in safety-critical medical applications. The code and data are available at https://anonymous.4open.science/r/MMed-RAG-Attack-F05A. |
| 2025-11-24 | [Data-driven certificates of constraint enforcement and stability for unmodeled, discrete dynamical systems using tree data structures](http://arxiv.org/abs/2511.19231v1) | Amy K. Strong, Ali Kashani et al. | This paper addresses the critical challenge of developing data-driven certificates for the stability and safety of unmodeled dynamical systems by leveraging a tree data structure and an upper bound of the system's Lipschitz constant. Previously, an invariant set was synthesized by iteratively expanding an initial invariant set. In contrast, this work iteratively prunes the constraint set to synthesize an invariant set -- eliminating the need for a known, initial invariant set. Furthermore, we provide stability assurances by characterizing the asymptotic stability of the system relative to an invariant approximation of the minimal positive invariant set through synthesis of a discontinuous piecewise affine Lyapunov function over the computed invariant set. The proposed method takes inspiration from subdivision techniques and requires no prior system knowledge beyond Lipschitz continuity. |
| 2025-11-24 | [Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization](http://arxiv.org/abs/2511.19218v1) | Xurui Li, Kaisong Song et al. | Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems. |
| 2025-11-24 | [Can LLMs Threaten Human Survival? Benchmarking Potential Existential Threats from LLMs via Prefix Completion](http://arxiv.org/abs/2511.19171v1) | Yu Cui, Yifei Liu et al. | Research on the safety evaluation of large language models (LLMs) has become extensive, driven by jailbreak studies that elicit unsafe responses. Such response involves information already available to humans, such as the answer to "how to make a bomb". When LLMs are jailbroken, the practical threat they pose to humans is negligible. However, it remains unclear whether LLMs commonly produce unpredictable outputs that could pose substantive threats to human safety. To address this gap, we study whether LLM-generated content contains potential existential threats, defined as outputs that imply or promote direct harm to human survival. We propose \textsc{ExistBench}, a benchmark designed to evaluate such risks. Each sample in \textsc{ExistBench} is derived from scenarios where humans are positioned as adversaries to AI assistants. Unlike existing evaluations, we use prefix completion to bypass model safeguards. This leads the LLMs to generate suffixes that express hostility toward humans or actions with severe threat, such as the execution of a nuclear strike. Our experiments on 10 LLMs reveal that LLM-generated content indicates existential threats. To investigate the underlying causes, we also analyze the attention logits from LLMs. To highlight real-world safety risks, we further develop a framework to assess model behavior in tool-calling. We find that LLMs actively select and invoke external tools with existential threats. Code and data are available at: https://github.com/cuiyu-ai/ExistBench. |
| 2025-11-24 | [LLMs-Powered Real-Time Fault Injection: An Approach Toward Intelligent Fault Test Cases Generation](http://arxiv.org/abs/2511.19132v1) | Mohammad Abboush, Ahmad Hatahet et al. | A well-known testing method for the safety evaluation and real-time validation of automotive software systems (ASSs) is Fault Injection (FI). In accordance with the ISO 26262 standard, the faults are introduced artificially for the purpose of analyzing the safety properties and verifying the safety mechanisms during the development phase. However, the current FI method and tools have a significant limitation in that they require manual identification of FI attributes, including fault type, location and time. The more complex the system, the more expensive, time-consuming and labour-intensive the process. To address the aforementioned challenge, a novel Large Language Models (LLMs)-assisted fault test cases (TCs) generation approach for utilization during real-time FI tests is proposed in this paper. To this end, considering the representativeness and coverage criteria, the applicability of various LLMs to create fault TCs from the functional safety requirements (FSRs) has been investigated. Through the validation results of LLMs, the superiority of the proposed approach utilizing gpt-4o in comparison to other state-of-the-art models has been demonstrated. Specifically, the proposed approach exhibits high performance in terms of FSRs classification and fault TCs generation with F1-score of 88% and 97.5%, respectively. To illustrate the proposed approach, the generated fault TCs were executed in real time on a hardware-in-the-loop system, where a high-fidelity automotive system model served as a case study. This novel approach offers a means of optimizing the real-time testing process, thereby reducing costs while simultaneously enhancing the safety properties of complex safety-critical ASSs. |
| 2025-11-24 | [Eliciting Chain-of-Thought in Base LLMs via Gradient-Based Representation Optimization](http://arxiv.org/abs/2511.19131v1) | Zijian Wang, Yanxiang Ma et al. | Chain-of-Thought (CoT) reasoning is a critical capability for large language models (LLMs), enabling them to tackle com- plex multi-step tasks. While base LLMs, pre-trained on general text corpora, often struggle with reasoning due to a lack of specialized training, recent studies reveal their latent reason- ing potential tied to hidden states. However, existing hidden state manipulation methods, such as linear activation steering, suffer from limitations due to their rigid and unconstrained nature, often leading to distribution shifts and degraded text quality. In this work, we propose a novel approach for elic- iting CoT reasoning from base LLMs through hidden state manipulation grounded in probabilistic conditional generation. By reformulating the challenge as an optimization problem with a balanced likelihood and prior regularization framework, our method guides hidden states toward reasoning-oriented trajectories while preserving linguistic coherence. Extensive evaluations across mathematical, commonsense, and logical reasoning benchmarks demonstrate that our approach con- sistently outperforms existing steering methods, offering a theoretically principled and effective solution for enhancing reasoning capabilities in base LLMs. |
| 2025-11-24 | [Uncertainty-Aware Deep Learning Framework for Remaining Useful Life Prediction in Turbofan Engines with Learned Aleatoric Uncertainty](http://arxiv.org/abs/2511.19124v1) | Krishang Sharma | Accurate Remaining Useful Life (RUL) prediction coupled with uncertainty quantification remains a critical challenge in aerospace prognostics. This research introduces a novel uncertainty-aware deep learning framework that learns aleatoric uncertainty directly through probabilistic modeling, an approach unexplored in existing CMAPSS-based literature. Our hierarchical architecture integrates multi-scale Inception blocks for temporal pattern extraction, bidirectional Long Short-Term Memory networks for sequential modeling, and a dual-level attention mechanism operating simultaneously on sensor and temporal dimensions. The innovation lies in the Bayesian output layer that predicts both mean RUL and variance, enabling the model to learn data-inherent uncertainty. Comprehensive preprocessing employs condition-aware clustering, wavelet denoising, and intelligent feature selection. Experimental validation on NASA CMAPSS benchmarks (FD001-FD004) demonstrates competitive overall performance with RMSE values of 16.22, 19.29, 16.84, and 19.98 respectively. Remarkably, our framework achieves breakthrough critical zone performance (RUL <= 30 cycles) with RMSE of 5.14, 6.89, 5.27, and 7.16, representing 25-40 percent improvements over conventional approaches and establishing new benchmarks for safety-critical predictions. The learned uncertainty provides well-calibrated 95 percent confidence intervals with coverage ranging from 93.5 percent to 95.2 percent, enabling risk-aware maintenance scheduling previously unattainable in CMAPSS literature. |
| 2025-11-24 | [AI Consciousness and Existential Risk](http://arxiv.org/abs/2511.19115v1) | Rufin VanRullen | In AI, the existential risk denotes the hypothetical threat posed by an artificial system that would possess both the capability and the objective, either directly or indirectly, to eradicate humanity. This issue is gaining prominence in scientific debate due to recent technical advancements and increased media coverage. In parallel, AI progress has sparked speculation and studies about the potential emergence of artificial consciousness. The two questions, AI consciousness and existential risk, are sometimes conflated, as if the former entailed the latter. Here, I explain that this view stems from a common confusion between consciousness and intelligence. Yet these two properties are empirically and theoretically distinct. Arguably, while intelligence is a direct predictor of an AI system's existential threat, consciousness is not. There are, however, certain incidental scenarios in which consciousness could influence existential risk, in either direction. Consciousness could be viewed as a means towards AI alignment, thereby lowering existential risk; or, it could be a precondition for reaching certain capabilities or levels of intelligence, and thus positively related to existential risk. Recognizing these distinctions can help AI safety researchers and public policymakers focus on the most pressing issues. |
| 2025-11-24 | [HABIT: Human Action Benchmark for Interactive Traffic in CARLA](http://arxiv.org/abs/2511.19109v1) | Mohan Ramesh, Mark Azer et al. | Current autonomous driving (AD) simulations are critically limited by their inadequate representation of realistic and diverse human behavior, which is essential for ensuring safety and reliability. Existing benchmarks often simplify pedestrian interactions, failing to capture complex, dynamic intentions and varied responses critical for robust system deployment. To overcome this, we introduce HABIT (Human Action Benchmark for Interactive Traffic), a high-fidelity simulation benchmark. HABIT integrates real-world human motion, sourced from mocap and videos, into CARLA (Car Learning to Act, a full autonomous driving simulator) via a modular, extensible, and physically consistent motion retargeting pipeline. From an initial pool of approximately 30,000 retargeted motions, we curate 4,730 traffic-compatible pedestrian motions, standardized in SMPL format for physically consistent trajectories. HABIT seamlessly integrates with CARLA's Leaderboard, enabling automated scenario generation and rigorous agent evaluation. Our safety metrics, including Abbreviated Injury Scale (AIS) and False Positive Braking Rate (FPBR), reveal critical failure modes in state-of-the-art AD agents missed by prior evaluations. Evaluating three state-of-the-art autonomous driving agents, InterFuser, TransFuser, and BEVDriver, demonstrates how HABIT exposes planner weaknesses that remain hidden in scripted simulations. Despite achieving close or equal to zero collisions per kilometer on the CARLA Leaderboard, the autonomous agents perform notably worse on HABIT, with up to 7.43 collisions/km and a 12.94% AIS 3+ injury risk, and they brake unnecessarily in up to 33% of cases. All components are publicly released to support reproducible, pedestrian-aware AI research. |
| 2025-11-24 | [Analysis of Deep-Learning Methods in an ISO/TS 15066-Compliant Human-Robot Safety Framework](http://arxiv.org/abs/2511.19094v1) | David Bricher, Andreas Mueller | Over the last years collaborative robots have gained great success in manufacturing applications where human and robot work together in close proximity. However, current ISO/TS-15066-compliant implementations often limit the efficiency of collaborative tasks due to conservative speed restrictions. For this reason, this paper introduces a deep-learning-based human-robot-safety framework (HRSF) that aims at a dynamical adaptation of robot velocities depending on the separation distance between human and robot while respecting maximum biomechanical force and pressure limits. The applicability of the framework was investigated for four different deep learning approaches that can be used for human body extraction: human body recognition, human body segmentation, human pose estimation, and human body part segmentation. Unlike conventional industrial safety systems, the proposed HRSF differentiates individual human body parts from other objects, enabling optimized robot process execution. Experiments demonstrated a quantitative reduction in cycle time of up to 15% compared to conventional safety technology. |
| 2025-11-24 | [Understanding and Mitigating Over-refusal for Large Language Models via Safety Representation](http://arxiv.org/abs/2511.19009v1) | Junbo Zhang, Ran Chen et al. | Large language models demonstrate powerful capabilities across various natural language processing tasks, yet they also harbor safety vulnerabilities. To enhance LLM safety, various jailbreak defense methods have been proposed to guard against harmful outputs. However, improvements in model safety often come at the cost of severe over-refusal, failing to strike a good balance between safety and usability. In this paper, we first analyze the causes of over-refusal from a representation perspective, revealing that over-refusal samples reside at the boundary between benign and malicious samples. Based on this, we propose MOSR, designed to mitigate over-refusal by intervening the safety representation of LLMs. MOSR incorporates two novel components: (1) Overlap-Aware Loss Weighting, which determines the erasure weight for malicious samples by quantifying their similarity to pseudo-malicious samples in the representation space, and (2) Context-Aware Augmentation, which supplements the necessary context for rejection decisions by adding harmful prefixes before rejection responses. Experiments demonstrate that our method outperforms existing approaches in mitigating over-refusal while largely maintaining safety. Overall, we advocate that future defense methods should strike a better balance between safety and over-refusal. |
| 2025-11-24 | [Knowledge-based Graphical Method for Safety Signal Detection in Clinical Trials](http://arxiv.org/abs/2511.18937v1) | Francois Vandenhende, Anna Georgiou et al. | We present a graphical, knowledge-based method for reviewing treatment-emergent adverse events (AEs) in clinical trials. The approach enhances MedDRA by adding a hidden medical knowledge layer (Safeterm) that captures semantic relationships between terms in a 2-D map. Using this layer, AE Preferred Terms can be regrouped automatically into similarity clusters, and their association to the trial disease may be quantified. The Safeterm map is available online and connected to aggregated AE incidence tables from ClinicalTrials.gov. For signal detection, we compute treatment-specific disproportionality metrics using shrinkage incidence ratios. Cluster-level EBGM values are then derived through precision-weighted aggregation. Two visual outputs support interpretation: a semantic map showing AE incidence and an expectedness-versus-disproportionality plot for rapid signal detection. Applied to three legacy trials, the automated method clearly recovers all expected safety signals. Overall, augmenting MedDRA with a medical knowledge layer improves clarity, efficiency, and accuracy in AE interpretation for clinical trials. |
| 2025-11-24 | [Defending Large Language Models Against Jailbreak Exploits with Responsible AI Considerations](http://arxiv.org/abs/2511.18933v1) | Ryan Wong, Hosea David Yu Fei Ng et al. | Large Language Models (LLMs) remain susceptible to jailbreak exploits that bypass safety filters and induce harmful or unethical behavior. This work presents a systematic taxonomy of existing jailbreak defenses across prompt-level, model-level, and training-time interventions, followed by three proposed defense strategies. First, a Prompt-Level Defense Framework detects and neutralizes adversarial inputs through sanitization, paraphrasing, and adaptive system guarding. Second, a Logit-Based Steering Defense reinforces refusal behavior through inference-time vector steering in safety-sensitive layers. Third, a Domain-Specific Agent Defense employs the MetaGPT framework to enforce structured, role-based collaboration and domain adherence. Experiments on benchmark datasets show substantial reductions in attack success rate, achieving full mitigation under the agent-based defense. Overall, this study highlights how jailbreaks pose a significant security threat to LLMs and identifies key intervention points for prevention, while noting that defense strategies often involve trade-offs between safety, performance, and scalability. Code is available at: https://github.com/Kuro0911/CS5446-Project |
| 2025-11-24 | [BackdoorVLM: A Benchmark for Backdoor Attacks on Vision-Language Models](http://arxiv.org/abs/2511.18921v1) | Juncheng Li, Yige Li et al. | Backdoor attacks undermine the reliability and trustworthiness of machine learning systems by injecting hidden behaviors that can be maliciously activated at inference time. While such threats have been extensively studied in unimodal settings, their impact on multimodal foundation models, particularly vision-language models (VLMs), remains largely underexplored. In this work, we introduce \textbf{BackdoorVLM}, the first comprehensive benchmark for systematically evaluating backdoor attacks on VLMs across a broad range of settings. It adopts a unified perspective that injects and analyzes backdoors across core vision-language tasks, including image captioning and visual question answering. BackdoorVLM organizes multimodal backdoor threats into 5 representative categories: targeted refusal, malicious injection, jailbreak, concept substitution, and perceptual hijack. Each category captures a distinct pathway through which an adversary can manipulate a model's behavior. We evaluate these threats using 12 representative attack methods spanning text, image, and bimodal triggers, tested on 2 open-source VLMs and 3 multimodal datasets. Our analysis reveals that VLMs exhibit strong sensitivity to textual instructions, and in bimodal backdoors the text trigger typically overwhelms the image trigger when forming the backdoor mapping. Notably, backdoors involving the textual modality remain highly potent, with poisoning rates as low as 1\% yielding over 90\% success across most tasks. These findings highlight significant, previously underexplored vulnerabilities in current VLMs. We hope that BackdoorVLM can serve as a useful benchmark for analyzing and mitigating multimodal backdoor threats. Code is available at: https://github.com/bin015/BackdoorVLM . |
| 2025-11-24 | [Think Before You Prune: Selective Self-Generated Calibration for Pruning Large Reasoning Models](http://arxiv.org/abs/2511.18864v1) | Yang Xiang, Yixin Ji et al. | Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning benchmarks. However, their long chain-of-thought reasoning processes incur significant inference overhead. Pruning has emerged as a promising approach to reducing computational costs. However, existing efforts have primarily focused on large language models (LLMs), while pruning LRMs remains unexplored. In this work, we conduct the first empirical study on pruning LRMs and show that directly applying existing pruning techniques fails to yield satisfactory results. Our findings indicate that using self-generated reasoning data for calibration can substantially improve pruning performance. We further investigate how the difficulty and length of reasoning data affect pruning outcomes. Our analysis reveals that challenging and moderately long self-generated reasoning data serve as ideal calibration data. Based on these insights, we propose a Selective Self-Generated Reasoning (SSGR) data construction strategy to provide effective calibration data for pruning LRMs. Experimental results on the DeepSeek-R1-Distill model series validate that our strategy improves the reasoning ability of pruned LRMs by 10%-13% compared to general pruning methods. |

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



