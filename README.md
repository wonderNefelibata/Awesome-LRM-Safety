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
| 2025-10-21 | [EffiReasonTrans: RL-Optimized Reasoning for Code Translation](http://arxiv.org/abs/2510.18863v1) | Yanlin Wang, Rongyi Ou et al. | Code translation is a crucial task in software development and maintenance. While recent advancements in large language models (LLMs) have improved automated code translation accuracy, these gains often come at the cost of increased inference latency, hindering real-world development workflows that involve human-in-the-loop inspection. To address this trade-off, we propose EffiReasonTrans, a training framework designed to improve translation accuracy while balancing inference latency. We first construct a high-quality reasoning-augmented dataset by prompting a stronger language model, DeepSeek-R1, to generate intermediate reasoning and target translations. Each (source code, reasoning, target code) triplet undergoes automated syntax and functionality checks to ensure reliability. Based on this dataset, we employ a two-stage training strategy: supervised fine-tuning on reasoning-augmented samples, followed by reinforcement learning to further enhance accuracy and balance inference latency. We evaluate EffiReasonTrans on six translation pairs. Experimental results show that it consistently improves translation accuracy (up to +49.2% CA and +27.8% CodeBLEU compared to the base model) while reducing the number of generated tokens (up to -19.3%) and lowering inference latency in most cases (up to -29.0%). Ablation studies further confirm the complementary benefits of the two-stage training framework. Additionally, EffiReasonTrans demonstrates improved translation accuracy when integrated into agent-based frameworks. Our code and data are available at https://github.com/DeepSoftwareAnalytics/EffiReasonTrans. |
| 2025-10-21 | [Lyapunov-Aware Quantum-Inspired Reinforcement Learning for Continuous-Time Vehicle Control: A Feasibility Study](http://arxiv.org/abs/2510.18852v1) | Nutkritta Kraipatthanapong, Natthaphat Thathong et al. | This paper presents a novel Lyapunov-Based Quantum Reinforcement Learning (LQRL) framework that integrates quantum policy optimization with Lyapunov stability analysis for continuous-time vehicle control. The proposed approach combines the representational power of variational quantum circuits (VQCs) with a stability-aware policy gradient mechanism to ensure asymptotic convergence and safe decision-making under dynamic environments. The vehicle longitudinal control problem was formulated as a continuous-state reinforcement learning task, where the quantum policy network generates control actions subject to Lyapunov stability constraints. Simulation experiments were conducted in a closed-loop adaptive cruise control scenario using a quantum-inspired policy trained under stability feedback. The results demonstrate that the LQRL framework successfully embeds Lyapunov stability verification into quantum policy learning, enabling interpretable and stability-aware control performance. Although transient overshoot and Lyapunov divergence were observed under aggressive acceleration, the system maintained bounded state evolution, validating the feasibility of integrating safety guarantees within quantum reinforcement learning architectures. The proposed framework provides a foundational step toward provably safe quantum control in autonomous systems and hybrid quantum-classical optimization domains. |
| 2025-10-21 | [HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models](http://arxiv.org/abs/2510.18728v1) | Sidhant Narula, Javad Rafiei Asl et al. | Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement. |
| 2025-10-21 | [Least Restrictive Hyperplane Control Barrier Functions](http://arxiv.org/abs/2510.18643v1) | Mattias Trende, Petter Ögren | Control Barrier Functions (CBFs) can provide provable safety guarantees for dynamic systems. However, finding a valid CBF for a system of interest is often non-trivial, especially if the shape of the unsafe region is complex and the CBFs are of higher order. A common solution to this problem is to make a conservative approximation of the unsafe region in the form of a line/hyperplane, and use the corresponding conservative Hyperplane-CBF when deciding on safe control actions. In this letter, we note that conservative constraints are only a problem if they prevent us from doing what we want. Thus, instead of first choosing a CBF and then choosing a safe control with respect to the CBF, we optimize over a combination of CBFs and safe controls to get as close as possible to our desired control, while still having the safety guarantee provided by the CBF. We call the corresponding CBF the least restrictive Hyperplane-CBF. Finally, we also provide a way of creating a smooth parameterization of the CBF-family for the optimization, and illustrate the approach on a double integrator dynamical system with acceleration constraints, moving through a group of arbitrarily shaped static and moving obstacles. |
| 2025-10-21 | [VAR: Visual Attention Reasoning via Structured Search and Backtracking](http://arxiv.org/abs/2510.18619v1) | Wei Cai, Jian Zhao et al. | Multimodal Large Language Models (MLLMs), despite their advances, are hindered by their high hallucination tendency and heavy reliance on brittle, linear reasoning processes, leading to failures in complex tasks. To address these limitations, we introduce Visual Attention Reasoning (VAR), a novel framework that recasts grounded reasoning as a structured search over a reasoning trajectory space. VAR decomposes the reasoning process into two key stages: traceable evidence grounding and search-based chain-of-thought (CoT) generation, which incorporates a backtracking mechanism for self-correction. The search is guided by a multi-faceted reward function with semantic and geometric self-verification components, which penalize outputs that are not faithfully grounded in the visual input. We provide a theoretical analysis for our search strategy, validating its capability to find the correct solution with high probability. Experimental results show that our 7B model, VAR-7B, sets a new state-of-the-art on a comprehensive suite of hallucination and safety benchmarks, significantly outperforming existing open-source models and demonstrating competitive performance against leading proprietary systems. |
| 2025-10-21 | [Electromagnetic Field Exposure Assessment and Mitigation Strategies for Wireless Power Transfer Systems: A Review and Future Perspectives](http://arxiv.org/abs/2510.18570v1) | Akimasa Hirata, Teruo Onishi et al. | Wireless power transfer (WPT) technologies are increasingly being applied in fields ranging from consumer electronics and electric vehicles to space-based energy systems and medical implants. While WPT offers contactless power delivery, it introduces electromagnetic field (EMF) emissions, necessitating careful assessment to address safety and public health concerns. Exposure guidelines developed by ICNIRP and IEEE define frequency-dependent limits based on internal quantities, such as electric field strength and specific absorption rate, intended to prevent tissue nerve stimulation < 100 kHz and heating > 100 kHz, respectively. Complementing these guidelines, assessment standards including the International Electrotechnical Commission (IEC)/IEEE 63184 and IEC Technical Report 63377, provide practical procedures for evaluating the EMF exposure in WPT systems. This review offers a comparative overview of major WPT modalities, with a focus on recent developments in computational dosimetry and standardized assessment techniques for the complex, non-uniform fields typical of WPT environments. It also discusses electromagnetic interference with medical devices and exposure scenarios involving partial body proximity and various postures. A notable observation across modalities is the considerable variability, often spanning an order of magnitude, in the allowable transfer power, depending on the field distribution and assessment approach. Remaining challenges include the lack of harmonized guidance for intermediate frequencies and localized exposure, underscoring the importance of further coordination in international standardization efforts. Addressing these issues is essential for the safe and widespread deployment of WPT technologies. |
| 2025-10-21 | [The Trust Paradox in LLM-Based Multi-Agent Systems: When Collaboration Becomes a Security Vulnerability](http://arxiv.org/abs/2510.18563v1) | Zijie Xu, Minfeng Qi et al. | Multi-agent systems powered by large language models are advancing rapidly, yet the tension between mutual trust and security remains underexplored. We introduce and empirically validate the Trust-Vulnerability Paradox (TVP): increasing inter-agent trust to enhance coordination simultaneously expands risks of over-exposure and over-authorization. To investigate this paradox, we construct a scenario-game dataset spanning 3 macro scenes and 19 sub-scenes, and run extensive closed-loop interactions with trust explicitly parameterized. Using Minimum Necessary Information (MNI) as the safety baseline, we propose two unified metrics: Over-Exposure Rate (OER) to detect boundary violations, and Authorization Drift (AD) to capture sensitivity to trust levels. Results across multiple model backends and orchestration frameworks reveal consistent trends: higher trust improves task success but also heightens exposure risks, with heterogeneous trust-to-risk mappings across systems. We further examine defenses such as Sensitive Information Repartitioning and Guardian-Agent enablement, both of which reduce OER and attenuate AD. Overall, this study formalizes TVP, establishes reproducible baselines with unified metrics, and demonstrates that trust must be modeled and scheduled as a first-class security variable in multi-agent system design. |
| 2025-10-21 | [Building Trust in Clinical LLMs: Bias Analysis and Dataset Transparency](http://arxiv.org/abs/2510.18556v1) | Svetlana Maslenkova, Clement Christophe et al. | Large language models offer transformative potential for healthcare, yet their responsible and equitable development depends critically on a deeper understanding of how training data characteristics influence model behavior, including the potential for bias. Current practices in dataset curation and bias assessment often lack the necessary transparency, creating an urgent need for comprehensive evaluation frameworks to foster trust and guide improvements. In this study, we present an in-depth analysis of potential downstream biases in clinical language models, with a focus on differential opioid prescription tendencies across diverse demographic groups, such as ethnicity, gender, and age. As part of this investigation, we introduce HC4: Healthcare Comprehensive Commons Corpus, a novel and extensively curated pretraining dataset exceeding 89 billion tokens. Our evaluation leverages both established general benchmarks and a novel, healthcare-specific methodology, offering crucial insights to support fairness and safety in clinical AI applications. |
| 2025-10-21 | [Extracting alignment data in open models](http://arxiv.org/abs/2510.18554v1) | Federico Barbero, Xiangming Gu et al. | In this work, we show that it is possible to extract significant amounts of alignment training data from a post-trained model -- useful to steer the model to improve certain capabilities such as long-context reasoning, safety, instruction following, and maths. While the majority of related work on memorisation has focused on measuring success of training data extraction through string matching, we argue that embedding models are better suited for our specific goals. Distances measured through a high quality embedding model can identify semantic similarities between strings that a different metric such as edit distance will struggle to capture. In fact, in our investigation, approximate string matching would have severely undercounted (by a conservative estimate of $10\times$) the amount of data that can be extracted due to trivial artifacts that deflate the metric. Interestingly, we find that models readily regurgitate training data that was used in post-training phases such as SFT or RL. We show that this data can be then used to train a base model, recovering a meaningful amount of the original performance. We believe our work exposes a possibly overlooked risk towards extracting alignment data. Finally, our work opens up an interesting discussion on the downstream effects of distillation practices: since models seem to be regurgitating aspects of their training set, distillation can therefore be thought of as indirectly training on the model's original dataset. |
| 2025-10-21 | [Deep Q-Learning Assisted Bandwidth Reservation for Multi-Operator Time-Sensitive Vehicular Networking](http://arxiv.org/abs/2510.18553v1) | Abdullah Al-Khatib, Albert Gergus et al. | Very few available individual bandwidth reservation schemes provide efficient and cost-effective bandwidth reservation that is required for safety-critical and time-sensitive vehicular networked applications. These schemes allow vehicles to make reservation requests for the required resources. Accordingly, a Mobile Network Operator (MNO) can allocate and guarantee bandwidth resources based on these requests. However, due to uncertainty in future reservation time and bandwidth costs, the design of an optimized reservation strategy is challenging. In this article, we propose a novel multi-objective bandwidth reservation update approach with an optimal strategy based on Double Deep Q-Network (DDQN). The key design objectives are to minimize the reservation cost with multiple MNOs and to ensure reliable resource provisioning in uncertain situations by solving scenarios such as underbooked and overbooked reservations along the driving path. The enhancements and advantages of our proposed strategy have been demonstrated through extensive experimental results when compared to other methods like greedy update or other deep reinforcement learning approaches. Our strategy demonstrates a 40% reduction in bandwidth costs across all investigated scenarios and simultaneously resolves uncertain situations in a cost-effective manner. |
| 2025-10-21 | [Descriptor: Occluded nuScenes: A Multi-Sensor Dataset for Evaluating Perception Robustness in Automated Driving](http://arxiv.org/abs/2510.18552v1) | Sanjay Kumar, Tim Brophy et al. | Robust perception in automated driving requires reliable performance under adverse conditions, where sensors may be affected by partial failures or environmental occlusions. Although existing autonomous driving datasets inherently contain sensor noise and environmental variability, very few enable controlled, parameterised, and reproducible degradations across multiple sensing modalities. This gap limits the ability to systematically evaluate how perception and fusion architectures perform under well-defined adverse conditions. To address this limitation, we introduce the Occluded nuScenes Dataset, a novel extension of the widely used nuScenes benchmark. For the camera modality, we release both the full and mini versions with four types of occlusions, two adapted from public implementations and two newly designed. For radar and LiDAR, we provide parameterised occlusion scripts that implement three types of degradations each, enabling flexible and repeatable generation of occluded data. This resource supports consistent, reproducible evaluation of perception models under partial sensor failures and environmental interference. By releasing the first multi-sensor occlusion dataset with controlled and reproducible degradations, we aim to advance research on robust sensor fusion, resilience analysis, and safety-critical perception in automated driving. |
| 2025-10-21 | [Basis-Sensitive Quantum Typing via Realisability](http://arxiv.org/abs/2510.18542v1) | Alejandro Díaz-Caro, Octavio Malherbe et al. | We present $\lambda_B$, a quantum-control $\lambda$-calculus that refines previous basis-sensitive systems by allowing abstractions to be expressed with respect to arbitrary -- possibly entangled -- bases. Each abstraction and let construct is annotated with a basis, and a new basis-dependent substitution governs the decomposition of value distributions. These extensions preserve the expressive power of earlier calculi while enabling finer reasoning about programs under basis changes. A realisability semantics connects the reduction system with the type system, yielding a direct characterisation of unitary operators and ensuring safety by construction. From this semantics we derive a validated family of typing rules, forming the foundation of a type-safe quantum programming language. We illustrate the expressive benefits of $\lambda_B$ through examples such as Deutsch's algorithm and quantum teleportation, where basis-aware typing captures classical determinism and deferred-measurement behaviour within a uniform framework. |
| 2025-10-21 | [Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation](http://arxiv.org/abs/2510.18541v1) | Giovanni De Muri, Mark Vero et al. | LLMs are often used by downstream users as teacher models for knowledge distillation, compressing their capabilities into memory-efficient models. However, as these teacher models may stem from untrusted parties, distillation can raise unexpected security risks. In this paper, we investigate the security implications of knowledge distillation from backdoored teacher models. First, we show that prior backdoors mostly do not transfer onto student models. Our key insight is that this is because existing LLM backdooring methods choose trigger tokens that rarely occur in usual contexts. We argue that this underestimates the security risks of knowledge distillation and introduce a new backdooring technique, T-MTB, that enables the construction and study of transferable backdoors. T-MTB carefully constructs a composite backdoor trigger, made up of several specific tokens that often occur individually in anticipated distillation datasets. As such, the poisoned teacher remains stealthy, while during distillation the individual presence of these tokens provides enough signal for the backdoor to transfer onto the student. Using T-MTB, we demonstrate and extensively study the security risks of transferable backdoors across two attack scenarios, jailbreaking and content modulation, and across four model families of LLMs. |
| 2025-10-21 | [One Size Fits All? A Modular Adaptive Sanitization Kit (MASK) for Customizable Privacy-Preserving Phone Scam Detection](http://arxiv.org/abs/2510.18493v1) | Kangzhong Wang, Zitong Shen et al. | Phone scams remain a pervasive threat to both personal safety and financial security worldwide. Recent advances in large language models (LLMs) have demonstrated strong potential in detecting fraudulent behavior by analyzing transcribed phone conversations. However, these capabilities introduce notable privacy risks, as such conversations frequently contain sensitive personal information that may be exposed to third-party service providers during processing. In this work, we explore how to harness LLMs for phone scam detection while preserving user privacy. We propose MASK (Modular Adaptive Sanitization Kit), a trainable and extensible framework that enables dynamic privacy adjustment based on individual preferences. MASK provides a pluggable architecture that accommodates diverse sanitization methods - from traditional keyword-based techniques for high-privacy users to sophisticated neural approaches for those prioritizing accuracy. We also discuss potential modeling approaches and loss function designs for future development, enabling the creation of truly personalized, privacy-aware LLM-based detection systems that balance user trust and detection effectiveness, even beyond phone scam context. |
| 2025-10-21 | [Learning to Navigate Under Imperfect Perception: Conformalised Segmentation for Safe Reinforcement Learning](http://arxiv.org/abs/2510.18485v1) | Daniel Bethell, Simos Gerasimou et al. | Reliable navigation in safety-critical environments requires both accurate hazard perception and principled uncertainty handling to strengthen downstream safety handling. Despite the effectiveness of existing approaches, they assume perfect hazard detection capabilities, while uncertainty-aware perception approaches lack finite-sample guarantees. We present COPPOL, a conformal-driven perception-to-policy learning approach that integrates distribution-free, finite-sample safety guarantees into semantic segmentation, yielding calibrated hazard maps with rigorous bounds for missed detections. These maps induce risk-aware cost fields for downstream RL planning. Across two satellite-derived benchmarks, COPPOL increases hazard coverage (up to 6x) compared to comparative baselines, achieving near-complete detection of unsafe regions while reducing hazardous violations during navigation (up to approx 50%). More importantly, our approach remains robust to distributional shift, preserving both safety and efficiency. |
| 2025-10-21 | [Safe But Not Sorry: Reducing Over-Conservatism in Safety Critics via Uncertainty-Aware Modulation](http://arxiv.org/abs/2510.18478v1) | Daniel Bethell, Simos Gerasimou et al. | Ensuring the safe exploration of reinforcement learning (RL) agents is critical for deployment in real-world systems. Yet existing approaches struggle to strike the right balance: methods that tightly enforce safety often cripple task performance, while those that prioritize reward leave safety constraints frequently violated, producing diffuse cost landscapes that flatten gradients and stall policy improvement. We introduce the Uncertain Safety Critic (USC), a novel approach that integrates uncertainty-aware modulation and refinement into critic training. By concentrating conservatism in uncertain and costly regions while preserving sharp gradients in safe areas, USC enables policies to achieve effective reward-safety trade-offs. Extensive experiments show that USC reduces safety violations by approximately 40% while maintaining competitive or higher rewards, and reduces the error between predicted and true cost gradients by approximately 83%, breaking the prevailing trade-off between safety and performance and paving the way for scalable safe RL. |
| 2025-10-21 | [Engagement Undermines Safety: How Stereotypes and Toxicity Shape Humor in Language Models](http://arxiv.org/abs/2510.18454v1) | Atharvan Dogra, Soumya Suvra Ghosal et al. | Large language models are increasingly used for creative writing and engagement content, raising safety concerns about the outputs. Therefore, casting humor generation as a testbed, this work evaluates how funniness optimization in modern LLM pipelines couples with harmful content by jointly measuring humor, stereotypicality, and toxicity. This is further supplemented by analyzing incongruity signals through information-theoretic metrics. Across six models, we observe that harmful outputs receive higher humor scores which further increase under role-based prompting, indicating a bias amplification loop between generators and evaluators. Information-theoretic analyses show harmful cues widen predictive uncertainty and surprisingly, can even make harmful punchlines more expected for some models, suggesting structural embedding in learned humor distributions. External validation on an additional satire-generation task with human perceived funniness judgments shows that LLM satire increases stereotypicality and typically toxicity, including for closed models. Quantitatively, stereotypical/toxic jokes gain $10-21\%$ in mean humor score, stereotypical jokes appear $11\%$ to $28\%$ more often among the jokes marked funny by LLM-based metric and up to $10\%$ more often in generations perceived as funny by humans. |
| 2025-10-21 | [Automated urban waterlogging assessment and early warning through a mixture of foundation models](http://arxiv.org/abs/2510.18425v1) | Chenxu Zhang, Fuxiang Huang et al. | With climate change intensifying, urban waterlogging poses an increasingly severe threat to global public safety and infrastructure. However, existing monitoring approaches rely heavily on manual reporting and fail to provide timely and comprehensive assessments. In this study, we present Urban Waterlogging Assessment (UWAssess), a foundation model-driven framework that automatically identifies waterlogged areas in surveillance images and generates structured assessment reports. To address the scarcity of labeled data, we design a semi-supervised fine-tuning strategy and a chain-of-thought (CoT) prompting strategy to unleash the potential of the foundation model for data-scarce downstream tasks. Evaluations on challenging visual benchmarks demonstrate substantial improvements in perception performance. GPT-based evaluations confirm the ability of UWAssess to generate reliable textual reports that accurately describe waterlogging extent, depth, risk and impact. This dual capability enables a shift of waterlogging monitoring from perception to generation, while the collaborative framework of multiple foundation models lays the groundwork for intelligent and scalable systems, supporting urban management, disaster response and climate resilience. |
| 2025-10-21 | [MMRHP: A Miniature Mixed-Reality HIL Platform for Auditable Closed-Loop Evaluation](http://arxiv.org/abs/2510.18371v1) | Mingxin Li, Haibo Hu et al. | Validation of autonomous driving systems requires a trade-off between test fidelity, cost, and scalability. While miniaturized hardware-in-the-loop (HIL) platforms have emerged as a promising solution, a systematic framework supporting rigorous quantitative analysis is generally lacking, limiting their value as scientific evaluation tools. To address this challenge, we propose MMRHP, a miniature mixed-reality HIL platform that elevates miniaturized testing from functional demonstration to rigorous, reproducible quantitative analysis. The core contributions are threefold. First, we propose a systematic three-phase testing process oriented toward the Safety of the Intended Functionality(SOTIF)standard, providing actionable guidance for identifying the performance limits and triggering conditions of otherwise correctly functioning systems. Second, we design and implement a HIL platform centered around a unified spatiotemporal measurement core to support this process, ensuring consistent and traceable quantification of physical motion and system timing. Finally, we demonstrate the effectiveness of this solution through comprehensive experiments. The platform itself was first validated, achieving a spatial accuracy of 10.27 mm RMSE and a stable closed-loop latency baseline of approximately 45 ms. Subsequently, an in-depth Autoware case study leveraged this validated platform to quantify its performance baseline and identify a critical performance cliff at an injected latency of 40 ms. This work shows that a structured process, combined with a platform offering a unified spatio-temporal benchmark, enables reproducible, interpretable, and quantitative closed-loop evaluation of autonomous driving systems. |
| 2025-10-21 | [Ensembling Pruned Attention Heads For Uncertainty-Aware Efficient Transformers](http://arxiv.org/abs/2510.18358v1) | Firas Gabetni, Giuseppe Curci et al. | Uncertainty quantification (UQ) is essential for deploying deep neural networks in safety-critical settings. Although methods like Deep Ensembles achieve strong UQ performance, their high computational and memory costs hinder scalability to large models. We introduce Hydra Ensembles, an efficient transformer-based ensemble that prunes attention heads to create diverse members and merges them via a new multi-head attention with grouped fully-connected layers. This yields a compact model with inference speed close to a single network, matching or surpassing Deep Ensembles in UQ performance without retraining from scratch. We also provide an in-depth analysis of pruning, showing that naive approaches can harm calibration, whereas Hydra Ensembles preserves robust uncertainty. Experiments on image and text classification tasks, with various architectures, show consistent gains over Deep Ensembles. Remarkably, in zero-shot classification on ImageNet-1k, our approach surpasses state of the art methods, even without requiring additional training. |

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



