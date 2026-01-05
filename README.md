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
| 2026-01-02 | [Geometry of Reason: Spectral Signatures of Valid Mathematical Reasoning](http://arxiv.org/abs/2601.00791v1) | Valentin Noël | We present a training-free method for detecting valid mathematical reasoning in large language models through spectral analysis of attention patterns. By treating attention matrices as adjacency matrices of dynamic graphs over tokens, we extract four interpretable spectral diagnostics, the Fiedler value (algebraic connectivity), high-frequency energy ratio (HFER), graph signal smoothness, and spectral entropy, that exhibit statistically significant differences between valid and invalid mathematical proofs. Experiments across seven transformer models from four independent architectural families (Meta Llama, Alibaba Qwen, Microsoft Phi, and Mistral AI) demonstrate that this spectral signature produces effect sizes up to Cohen's $d = 3.30$ ($p < 10^{-116}$), enabling 85.0--95.6\% classification accuracy under rigorous evaluation, with calibrated thresholds reaching 93--95\% on the full dataset. The method requires no training data, fine-tuning, or learned classifiers: a single threshold on a spectral metric suffices for high accuracy. Through systematic label correction, we discover that the spectral method detects logical coherence rather than compiler acceptance, identifying mathematically valid proofs that formal verifiers reject due to technical failures. We further identify an architectural dependency: Mistral-7B's Sliding Window Attention shifts the discriminative signal from HFER to late-layer Smoothness ($d = 2.09$, $p_{\text{MW}} = 1.16 \times 10^{-48}$), revealing that attention mechanism design affects which spectral features capture reasoning validity. These findings establish spectral graph analysis as a principled framework for reasoning verification with immediate applications to hallucination detection and AI safety monitoring. |
| 2026-01-02 | [Towards Understanding and Characterizing Vulnerabilities in Intelligent Connected Vehicles through Real-World Exploits](http://arxiv.org/abs/2601.00627v1) | Yuelin Wang, Yuqiao Ning et al. | Intelligent Connected Vehicles (ICVs) are a core component of modern transportation systems, and their security is crucial as it directly relates to user safety. Despite prior research, most existing studies focus only on specific sub-components of ICVs due to their inherent complexity. As a result, there is a lack of systematic understanding of ICV vulnerabilities. Moreover, much of the current literature relies on human subjective analysis, such as surveys and interviews, which tends to be high-level and unvalidated, leaving a significant gap between theoretical findings and real-world attacks. To address this issue, we conducted the first large-scale empirical study on ICV vulnerabilities. We began by analyzing existing ICV security literature and summarizing the prevailing taxonomies in terms of vulnerability locations and types. To evaluate their real-world relevance, we collected a total of 649 exploitable vulnerabilities, including 592 from eight ICV vulnerability discovery competitions, Anonymous Cup, between January 2023 and April 2024, covering 48 different vehicles. The remaining 57 vulnerabilities were submitted daily by researchers. Based on this dataset, we assessed the coverage of existing taxonomies and identified several gaps, discovering one new vulnerability location and 13 new vulnerability types. We further categorized these vulnerabilities into 6 threat types (e.g., privacy data breach) and 4 risk levels (ranging from low to critical) and analyzed participants' skills and the types of ICVs involved in the competitions. This study provides a comprehensive and data-driven analysis of ICV vulnerabilities, offering actionable insights for researchers, industry practitioners, and policymakers. To support future research, we have made our vulnerability dataset publicly available. |
| 2026-01-02 | [Vision-based Goal-Reaching Control for Mobile Robots Using a Hierarchical Learning Framework](http://arxiv.org/abs/2601.00610v1) | Mehdi Heydari Shahna, Pauli Mustalahti et al. | Reinforcement learning (RL) is effective in many robotic applications, but it requires extensive exploration of the state-action space, during which behaviors can be unsafe. This significantly limits its applicability to large robots with complex actuators operating on unstable terrain. Hence, to design a safe goal-reaching control framework for large-scale robots, this paper decomposes the whole system into a set of tightly coupled functional modules. 1) A real-time visual pose estimation approach is employed to provide accurate robot states to 2) an RL motion planner for goal-reaching tasks that explicitly respects robot specifications. The RL module generates real-time smooth motion commands for the actuator system, independent of its underlying dynamic complexity. 3) In the actuation mechanism, a supervised deep learning model is trained to capture the complex dynamics of the robot and provide this model to 4) a model-based robust adaptive controller that guarantees the wheels track the RL motion commands even on slip-prone terrain. 5) Finally, to reduce human intervention, a mathematical safety supervisor monitors the robot, stops it on unsafe faults, and autonomously guides it back to a safe inspection area. The proposed framework guarantees uniform exponential stability of the actuation system and safety of the whole operation. Experiments on a 6,000 kg robot in different scenarios confirm the effectiveness of the proposed framework. |
| 2026-01-02 | [NMPC-Augmented Visual Navigation and Safe Learning Control for Large-Scale Mobile Robots](http://arxiv.org/abs/2601.00609v1) | Mehdi Heydari Shahna, Pauli Mustalahti et al. | A large-scale mobile robot (LSMR) is a high-order multibody system that often operates on loose, unconsolidated terrain, which reduces traction. This paper presents a comprehensive navigation and control framework for an LSMR that ensures stability and safety-defined performance, delivering robust operation on slip-prone terrain by jointly leveraging high-performance techniques. The proposed architecture comprises four main modules: (1) a visual pose-estimation module that fuses onboard sensors and stereo cameras to provide an accurate, low-latency robot pose, (2) a high-level nonlinear model predictive control that updates the wheel motion commands to correct robot drift from the robot reference pose on slip-prone terrain, (3) a low-level deep neural network control policy that approximates the complex behavior of the wheel-driven actuation mechanism in LSMRs, augmented with robust adaptive control to handle out-of-distribution disturbances, ensuring that the wheels accurately track the updated commands issued by high-level control module, and (4) a logarithmic safety module to monitor the entire robot stack and guarantees safe operation. The proposed low-level control framework guarantees uniform exponential stability of the actuation subsystem, while the safety module ensures the whole system-level safety during operation. Comparative experiments on a 6,000 kg LSMR actuated by two complex electro-hydrostatic drives, while synchronizing modules operating at different frequencies. |
| 2026-01-02 | [SafeMo: Linguistically Grounded Unlearning for Trustworthy Text-to-Motion Generation](http://arxiv.org/abs/2601.00590v1) | Yiling Wang, Zeyu Zhang et al. | Text-to-motion (T2M) generation with diffusion backbones achieves strong realism and alignment. Safety concerns in T2M methods have been raised in recent years; existing methods replace discrete VQ-VAE codebook entries to steer the model away from unsafe behaviors. However, discrete codebook replacement-based methods have two critical flaws: firstly, replacing codebook entries which are reused by benign prompts leads to drifts on everyday tasks, degrading the model's benign performance; secondly, discrete token-based methods introduce quantization and smoothness loss, resulting in artifacts and jerky transitions. Moreover, existing text-to-motion datasets naturally contain unsafe intents and corresponding motions, making them unsuitable for safety-driven machine learning. To address these challenges, we propose SafeMo, a trustworthy motion generative framework integrating Minimal Motion Unlearning (MMU), a two-stage machine unlearning strategy, enabling safe human motion generation in continuous space, preserving continuous kinematics without codebook loss and delivering strong safety-utility trade-offs compared to current baselines. Additionally, we present the first safe text-to-motion dataset SafeMoVAE-29K integrating rewritten safe text prompts and continuous refined motion for trustworthy human motion unlearning. Built upon DiP, SafeMo efficiently generates safe human motions with natural transitions. Experiments demonstrate effective unlearning performance of SafeMo by showing strengthened forgetting on unsafe prompts, reaching 2.5x and 14.4x higher forget-set FID on HumanML3D and Motion-X respectively, compared to the previous SOTA human motion unlearning method LCR, with benign performance on safe prompts being better or comparable. Code: https://github.com/AIGeeksGroup/SafeMo. Website: https://aigeeksgroup.github.io/SafeMo. |
| 2026-01-02 | [CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns](http://arxiv.org/abs/2601.00588v1) | Zhenhong Zhou, Shilinlu Yan et al. | Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice. |
| 2026-01-02 | [Cracking IoT Security: Can LLMs Outsmart Static Analysis Tools?](http://arxiv.org/abs/2601.00559v1) | Jason Quantrill, Noura Khajehnouri et al. | Smart home IoT platforms such as openHAB rely on Trigger Action Condition (TAC) rules to automate device behavior, but the interplay among these rules can give rise to interaction threats, unintended or unsafe behaviors emerging from implicit dependencies, conflicting triggers, or overlapping conditions. Identifying these threats requires semantic understanding and structural reasoning that traditionally depend on symbolic, constraint-driven static analysis. This work presents the first comprehensive evaluation of Large Language Models (LLMs) across a multi-category interaction threat taxonomy, assessing their performance on both the original openHAB (oHC/IoTB) dataset and a structurally challenging Mutation dataset designed to test robustness under rule transformations. We benchmark Llama 3.1 8B, Llama 70B, GPT-4o, Gemini-2.5-Pro, and DeepSeek-R1 across zero-, one-, and two-shot settings, comparing their results against oHIT's manually validated ground truth. Our findings show that while LLMs exhibit promising semantic understanding, particularly on action- and condition-related threats, their accuracy degrades significantly for threats requiring cross-rule structural reasoning, especially under mutated rule forms. Model performance varies widely across threat categories and prompt settings, with no model providing consistent reliability. In contrast, the symbolic reasoning baseline maintains stable detection across both datasets, unaffected by rule rewrites or structural perturbations. These results underscore that LLMs alone are not yet dependable for safety critical interaction-threat detection in IoT environments. We discuss the implications for tool design and highlight the potential of hybrid architectures that combine symbolic analysis with LLM-based semantic interpretation to reduce false positives while maintaining structural rigor. |
| 2026-01-02 | [Trajectory Guard -- A Lightweight, Sequence-Aware Model for Real-Time Anomaly Detection in Agentic AI](http://arxiv.org/abs/2601.00516v1) | Laksh Advani | Autonomous LLM agents generate multi-step action plans that can fail due to contextual misalignment or structural incoherence. Existing anomaly detection methods are ill-suited for this challenge: mean-pooling embeddings dilutes anomalous steps, while contrastive-only approaches ignore sequential structure. Standard unsupervised methods on pre-trained embeddings achieve F1-scores no higher than 0.69. We introduce Trajectory Guard, a Siamese Recurrent Autoencoder with a hybrid loss function that jointly learns task-trajectory alignment via contrastive learning and sequential validity via reconstruction. This dual objective enables unified detection of both "wrong plan for this task" and "malformed plan structure." On benchmarks spanning synthetic perturbations and real-world failures from security audits (RAS-Eval) and multi-agent systems (Who\&When), we achieve F1-scores of 0.88-0.94 on balanced sets and recall of 0.86-0.92 on imbalanced external benchmarks. At 32 ms inference latency, our approach runs 17-27$\times$ faster than LLM Judge baselines, enabling real-time safety verification in production deployments. |
| 2026-01-02 | [The Illusion of Insight in Reasoning Models](http://arxiv.org/abs/2601.00514v1) | Liv G. d'Aliberti, Manoel Horta Ribeiro | Do reasoning models have "Aha!" moments? Prior work suggests that models like DeepSeek-R1-Zero undergo sudden mid-trace realizations that lead to accurate outputs, implying an intrinsic capacity for self-correction. Yet, it remains unclear whether such intrinsic shifts in reasoning strategy actually improve performance. Here, we study mid-reasoning shifts and instrument training runs to detect them. Our analysis spans 1M+ reasoning traces, hundreds of training checkpoints, three reasoning domains, and multiple decoding temperatures and model architectures. We find that reasoning shifts are rare, do not become more frequent with training, and seldom improve accuracy, indicating that they do not correspond to prior perceptions of model insight. However, their effect varies with model uncertainty. Building on this finding, we show that artificially triggering extrinsic shifts under high entropy reliably improves accuracy. Our results show that mid-reasoning shifts are symptoms of unstable inference behavior rather than an intrinsic mechanism for self-correction. |
| 2026-01-01 | [STELLAR: A Search-Based Testing Framework for Large Language Model Applications](http://arxiv.org/abs/2601.00497v1) | Lev Sorokin, Ivan Vasilev et al. | Large Language Model (LLM)-based applications are increasingly deployed across various domains, including customer service, education, and mobility. However, these systems are prone to inaccurate, fictitious, or harmful responses, and their vast, high-dimensional input space makes systematic testing particularly challenging. To address this, we present STELLAR, an automated search-based testing framework for LLM-based applications that systematically uncovers text inputs leading to inappropriate system responses. Our framework models test generation as an optimization problem and discretizes the input space into stylistic, content-related, and perturbation features. Unlike prior work that focuses on prompt optimization or coverage heuristics, our work employs evolutionary optimization to dynamically explore feature combinations that are more likely to expose failures. We evaluate STELLAR on three LLM-based conversational question-answering systems. The first focuses on safety, benchmarking both public and proprietary LLMs against malicious or unsafe prompts. The second and third target navigation, using an open-source and an industrial retrieval-augmented system for in-vehicle venue recommendations. Overall, STELLAR exposes up to 4.3 times (average 2.5 times) more failures than the existing baseline approaches. |
| 2026-01-01 | [Safety for Weakly-Hard Control Systems via Graph-Based Barrier Functions](http://arxiv.org/abs/2601.00494v1) | Marc Seidel, Mahathi Anand et al. | Despite significant advancement in technology, communication and computational failures are still prevalent in safety-critical engineering applications. Often, networked control systems experience packet dropouts, leading to open-loop behavior that significantly affects the behavior of the system. Similarly, in real-time control applications, control tasks frequently experience computational overruns and thus occasionally no new actuator command is issued. This article addresses the safety verification and controller synthesis problem for a class of control systems subject to weakly-hard constraints, i.e., a set of window-based constraints where the number of failures are bounded within a given time horizon. The results are based on a new notion of graph-based barrier functions that are specifically tailored to the considered system class, offering a set of constraints whose satisfaction leads to safety guarantees despite communication failures. Subsequent reformulations of the safety constraints are proposed to alleviate conservatism and improve computational tractability, and the resulting trade-offs are discussed. Finally, several numerical case studies demonstrate the effectiveness of the proposed approach. |
| 2026-01-01 | [Safe Adaptive Feedback Control via Barrier States](http://arxiv.org/abs/2601.00476v1) | Trivikram Satharasi, Tochukwu E. Ogri et al. | This paper presents a safe feedback control framework for nonlinear control-affine systems with parametric uncertainty by leveraging adaptive dynamic programming (ADP) with barrier-state augmentation. The developed ADP-based controller enforces control invariance by optimizing a value function that explicitly penalizes the barrier state, thereby embedding safety directly into the Bellman structure. The near-optimal control policy computed using model-based reinforcement learning is combined with a concurrent learning estimator to identify the unknown parameters and guarantee uniform convergence without requiring persistency of excitation. Using a barrier-state Lyapunov function, we establish boundedness of the barrier dynamics and prove closed-loop stability and safety. Numerical simulations on an optimal obstacle-avoidance problem validate the effectiveness of the developed approach. |
| 2026-01-01 | [Defensive M2S: Training Guardrail Models on Compressed Multi-turn Conversations](http://arxiv.org/abs/2601.00454v1) | Hyunjun Kim | Guardrail models are essential for ensuring the safety of Large Language Model (LLM) deployments, but processing full multi-turn conversation histories incurs significant computational cost. We propose Defensive M2S, a training paradigm that fine-tunes guardrail models on Multi-turn to Single-turn (M2S) compressed conversations rather than complete dialogue histories. We provide a formal complexity analysis showing that M2S reduces training cost from $O(n^2)$ to $O(n)$ for $n$-turn conversations. Empirically, on our training dataset (779 samples, avg. 10.6 turns), M2S requires only 169K tokens compared to 15.7M tokens for the multi-turn baseline -- a 93$\times$ reduction. We evaluate Defensive M2S across three guardrail model families (LlamaGuard, Nemotron, Qwen3Guard) and three compression templates (hyphenize, numberize, pythonize) on SafeDialBench, a comprehensive multi-turn jailbreak benchmark. Our best configuration, Qwen3Guard with hyphenize compression, achieves 93.8% attack detection recall while reducing inference tokens by 94.6% (from 3,231 to 173 tokens per conversation). This represents a 38.9 percentage point improvement over the baseline while dramatically reducing both training and inference costs. Our findings demonstrate that M2S compression can serve as an effective efficiency technique for guardrail deployment, enabling scalable safety screening of long multi-turn conversations. |
| 2026-01-01 | [RoLID-11K: A Dashcam Dataset for Small-Object Roadside Litter Detection](http://arxiv.org/abs/2601.00398v1) | Tao Wu, Qing Xu et al. | Roadside litter poses environmental, safety and economic challenges, yet current monitoring relies on labour-intensive surveys and public reporting, providing limited spatial coverage. Existing vision datasets for litter detection focus on street-level still images, aerial scenes or aquatic environments, and do not reflect the unique characteristics of dashcam footage, where litter appears extremely small, sparse and embedded in cluttered road-verge backgrounds. We introduce RoLID-11K, the first large-scale dataset for roadside litter detection from dashcams, comprising over 11k annotated images spanning diverse UK driving conditions and exhibiting pronounced long-tail and small-object distributions. We benchmark a broad spectrum of modern detectors, from accuracy-oriented transformer architectures to real-time YOLO models, and analyse their strengths and limitations on this challenging task. Our results show that while CO-DETR and related transformers achieve the best localisation accuracy, real-time models remain constrained by coarse feature hierarchies. RoLID-11K establishes a challenging benchmark for extreme small-object detection in dynamic driving scenes and aims to support the development of scalable, low-cost systems for roadside-litter monitoring. The dataset is available at https://github.com/xq141839/RoLID-11K. |
| 2026-01-01 | [Intelligent Traffic Surveillance for Real-Time Vehicle Detection, License Plate Recognition, and Speed Estimation](http://arxiv.org/abs/2601.00344v1) | Bruce Mugizi, Sudi Murindanyi et al. | Speeding is a major contributor to road fatalities, particularly in developing countries such as Uganda, where road safety infrastructure is limited. This study proposes a real-time intelligent traffic surveillance system tailored to such regions, using computer vision techniques to address vehicle detection, license plate recognition, and speed estimation. The study collected a rich dataset using a speed gun, a Canon Camera, and a mobile phone to train the models. License plate detection using YOLOv8 achieved a mean average precision (mAP) of 97.9%. For character recognition of the detected license plate, the CNN model got a character error rate (CER) of 3.85%, while the transformer model significantly reduced the CER to 1.79%. Speed estimation used source and target regions of interest, yielding a good performance of 10 km/h margin of error. Additionally, a database was established to correlate user information with vehicle detection data, enabling automated ticket issuance via SMS via Africa's Talking API. This system addresses critical traffic management needs in resource-constrained environments and shows potential to reduce road accidents through automated traffic enforcement in developing countries where such interventions are urgently needed. |
| 2026-01-01 | [Offline Multi-Agent Reinforcement Learning for 6G Communications: Fundamentals, Applications and Future Directions](http://arxiv.org/abs/2601.00321v1) | Eslam Eldeeb, Hirley Alves | The next-generation wireless technologies, including beyond 5G and 6G networks, are paving the way for transformative applications such as vehicle platooning, smart cities, and remote surgery. These innovations are driven by a vast array of interconnected wireless entities, including IoT devices, access points, UAVs, and CAVs, which increase network complexity and demand more advanced decision-making algorithms. Artificial intelligence (AI) and machine learning (ML), especially reinforcement learning (RL), are key enablers for such networks, providing solutions to high-dimensional and complex challenges. However, as networks expand to multi-agent environments, traditional online RL approaches face cost, safety, and scalability limitations. Offline multi-agent reinforcement learning (MARL) offers a promising solution by utilizing pre-collected data, reducing the need for real-time interaction. This article introduces a novel offline MARL algorithm based on conservative Q-learning (CQL), ensuring safe and efficient training. We extend this with meta-learning to address dynamic environments and validate the approach through use cases in radio resource management and UAV networks. Our work highlights offline MARL's advantages, limitations, and future directions in wireless applications. |
| 2026-01-01 | [ClinicalReTrial: A Self-Evolving AI Agent for Clinical Trial Protocol Optimization](http://arxiv.org/abs/2601.00290v1) | Sixue Xing, Xuanye Xia et al. | Clinical trial failure remains a central bottleneck in drug development, where minor protocol design flaws can irreversibly compromise outcomes despite promising therapeutics. Although cutting-edge AI methods achieve strong performance in predicting trial success, they are inherently reactive for merely diagnosing risk without offering actionable remedies once failure is anticipated. To fill this gap, this paper proposes ClinicalReTrial, a self-evolving AI agent framework that addresses this gap by casting clinical trial reasoning as an iterative protocol redesign problem. Our method integrates failure diagnosis, safety-aware modification, and candidate evaluation in a closed-loop, reward-driven optimization framework. Serving the outcome prediction model as a simulation environment, ClinicalReTrial enables low-cost evaluation of protocol modifications and provides dense reward signals for continuous self-improvement. To support efficient exploration, the framework maintains hierarchical memory that captures iteration-level feedback within trials and distills transferable redesign patterns across trials. Empirically, ClinicalReTrial improves 83.3% of trial protocols with a mean success probability gain of 5.7%, and retrospective case studies demonstrate strong alignment between the discovered redesign strategies and real-world clinical trial modifications. |
| 2026-01-01 | [FaithSCAN: Model-Driven Single-Pass Hallucination Detection for Faithful Visual Question Answering](http://arxiv.org/abs/2601.00269v1) | Chaodong Tong, Qi Zhang et al. | Faithfulness hallucinations in VQA occur when vision-language models produce fluent yet visually ungrounded answers, severely undermining their reliability in safety-critical applications. Existing detection methods mainly fall into two categories: external verification approaches relying on auxiliary models or knowledge bases, and uncertainty-driven approaches using repeated sampling or uncertainty estimates. The former suffer from high computational overhead and are limited by external resource quality, while the latter capture only limited facets of model uncertainty and fail to sufficiently explore the rich internal signals associated with the diverse failure modes. Both paradigms thus have inherent limitations in efficiency, robustness, and detection performance. To address these challenges, we propose FaithSCAN: a lightweight network that detects hallucinations by exploiting rich internal signals of VLMs, including token-level decoding uncertainty, intermediate visual representations, and cross-modal alignment features. These signals are fused via branch-wise evidence encoding and uncertainty-aware attention. We also extend the LLM-as-a-Judge paradigm to VQA hallucination and propose a low-cost strategy to automatically generate model-dependent supervision signals, enabling supervised training without costly human labels while maintaining high detection accuracy. Experiments on multiple VQA benchmarks show that FaithSCAN significantly outperforms existing methods in both effectiveness and efficiency. In-depth analysis shows hallucinations arise from systematic internal state variations in visual perception, cross-modal reasoning, and language decoding. Different internal signals provide complementary diagnostic cues, and hallucination patterns vary across VLM architectures, offering new insights into the underlying causes of multimodal hallucinations. |
| 2026-01-01 | [ActErase: A Training-Free Paradigm for Precise Concept Erasure via Activation Patching](http://arxiv.org/abs/2601.00267v1) | Yi Sun, Xinhao Zhong et al. | Recent advances in text-to-image diffusion models have demonstrated remarkable generation capabilities, yet they raise significant concerns regarding safety, copyright, and ethical implications. Existing concept erasure methods address these risks by removing sensitive concepts from pre-trained models, but most of them rely on data-intensive and computationally expensive fine-tuning, which poses a critical limitation. To overcome these challenges, inspired by the observation that the model's activations are predominantly composed of generic concepts, with only a minimal component can represent the target concept, we propose a novel training-free method (ActErase) for efficient concept erasure. Specifically, the proposed method operates by identifying activation difference regions via prompt-pair analysis, extracting target activations and dynamically replacing input activations during forward passes. Comprehensive evaluations across three critical erasure tasks (nudity, artistic style, and object removal) demonstrates that our training-free method achieves state-of-the-art (SOTA) erasure performance, while effectively preserving the model's overall generative capability. Our approach also exhibits strong robustness against adversarial attacks, establishing a new plug-and-play paradigm for lightweight yet effective concept manipulation in diffusion models. |
| 2026-01-01 | [μACP: A Formal Calculus for Expressive, Resource-Constrained Agent Communication](http://arxiv.org/abs/2601.00219v1) | Arnab Mallick, Indraveni Chebolu | Agent communication remains a foundational problem in multi-agent systems: protocols such as FIPA-ACL guarantee semantic richness but are intractable for constrained environments, while lightweight IoT protocols achieve efficiency at the expense of expressiveness. This paper presents $μ$ACP, a formal calculus for expressive agent communication under explicit resource bounds. We formalize the Resource-Constrained Agent Communication (RCAC) model, prove that a minimal four-verb basis \textit{\{PING, TELL, ASK, OBSERVE\}} is suffices to encode finite-state FIPA protocols, and establish tight information-theoretic bounds on message complexity. We further show that $μ$ACP can implement standard consensus under partial synchrony and crash faults, yielding a constructive coordination framework for edge-native agents. Formal verification in TLA$^{+}$ (model checking) and Coq (mechanized invariants) establishes safety and boundedness, and supports liveness under modeled assumptions. Large-scale system simulations confirm ACP achieves a median end-to-end message latency of 34 ms (95th percentile 104 ms) at scale, outperforming prior agent and IoT protocols under severe resource constraints. The main contribution is a unified calculus that reconciles semantic expressiveness with provable efficiency, providing a rigorous foundation for the next generation of resource-constrained multi-agent systems. |

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



