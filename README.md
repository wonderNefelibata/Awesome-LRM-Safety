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
| 2025-10-02 | [Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks](http://arxiv.org/abs/2510.02286v1) | Ruohao Guo, Afshin Oroojlooy et al. | Despite recent rapid progress in AI safety, current large language models remain vulnerable to adversarial attacks in multi-turn interaction settings, where attackers strategically adapt their prompts across conversation turns and pose a more critical yet realistic challenge. Existing approaches that discover safety vulnerabilities either rely on manual red-teaming with human experts or employ automated methods using pre-defined templates and human-curated attack data, with most focusing on single-turn attacks. However, these methods did not explore the vast space of possible multi-turn attacks, failing to consider novel attack trajectories that emerge from complex dialogue dynamics and strategic conversation planning. This gap is particularly critical given recent findings that LLMs exhibit significantly higher vulnerability to multi-turn attacks compared to single-turn attacks. We propose DialTree-RPO, an on-policy reinforcement learning framework integrated with tree search that autonomously discovers diverse multi-turn attack strategies by treating the dialogue as a sequential decision-making problem, enabling systematic exploration without manually curated data. Through extensive experiments, our approach not only achieves more than 25.9% higher ASR across 10 target models compared to previous state-of-the-art approaches, but also effectively uncovers new attack strategies by learning optimal dialogue policies that maximize attack success across multiple turns. |
| 2025-10-02 | [More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration](http://arxiv.org/abs/2510.02227v1) | Xiaoyang Yuan, Yujuan Ding et al. | Reinforcement Learning with Verifiable Rewards (RLVR) is a promising paradigm for enhancing the reasoning ability in Large Language Models (LLMs). However, prevailing methods primarily rely on self-exploration or a single off-policy teacher to elicit long chain-of-thought (LongCoT) reasoning, which may introduce intrinsic model biases and restrict exploration, ultimately limiting reasoning diversity and performance. Drawing inspiration from multi-teacher strategies in knowledge distillation, we introduce Adaptive Multi-Guidance Policy Optimization (AMPO), a novel framework that adaptively leverages guidance from multiple proficient teacher models, but only when the on-policy model fails to generate correct solutions. This "guidance-on-demand" approach expands exploration while preserving the value of self-discovery. Moreover, AMPO incorporates a comprehension-based selection mechanism, prompting the student to learn from the reasoning paths that it is most likely to comprehend, thus balancing broad exploration with effective exploitation. Extensive experiments show AMPO substantially outperforms a strong baseline (GRPO), with a 4.3% improvement on mathematical reasoning tasks and 12.2% on out-of-distribution tasks, while significantly boosting Pass@k performance and enabling more diverse exploration. Notably, using four peer-sized teachers, our method achieves comparable results to approaches that leverage a single, more powerful teacher (e.g., DeepSeek-R1) with more data. These results demonstrate a more efficient and scalable path to superior reasoning and generalizability. Our code is available at https://github.com/SII-Enigma/AMPO. |
| 2025-10-02 | [Computing Control Lyapunov-Barrier Functions: Softmax Relaxation and Smooth Patching with Formal Guarantees](http://arxiv.org/abs/2510.02223v1) | Jun Liu, Maxwell Fitzsimmons | We present a computational framework for synthesizing a single smooth Lyapunov function that certifies both asymptotic stability and safety. We show that the existence of a strictly compatible pair of control barrier and control Lyapunov functions (CBF-CLF) guarantees the existence of such a function on the exact safe set certified by the barrier. To maximize the certifiable safe domain while retaining differentiability, we employ a log-sum-exp (softmax) relaxation of the nonsmooth maximum barrier, together with a counterexample-guided refinement that inserts half-space cuts until a strict barrier condition is verifiable. We then patch the softmax barrier with a CLF via an explicit smooth bump construction, which is always feasible under the strict compatibility condition. All conditions are formally verified using a satisfiability modulo theories (SMT) solver, enabled by a reformulation of Farkas' lemma for encoding strict compatibility. On benchmark systems, including a power converter, we show that the certified safe stabilization regions obtained with the proposed approach are often less conservative than those achieved by state-of-the-art sum-of-squares (SOS) compatible CBF-CLF designs. |
| 2025-10-02 | [UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models](http://arxiv.org/abs/2510.02194v1) | Yuhao Sun, Zhuoer Xu et al. | Large Language Models (LLMs) have achieved remarkable progress across a wide range of tasks, but remain vulnerable to safety risks such as harmful content generation and jailbreak attacks. Existing safety techniques -- including external guardrails, inference-time guidance, and post-training alignment -- each face limitations in balancing safety, utility, and controllability. In this work, we propose UpSafe$^\circ$C, a unified framework for enhancing LLM safety through safety-aware upcycling. Our approach first identifies safety-critical layers and upcycles them into a sparse Mixture-of-Experts (MoE) structure, where the router acts as a soft guardrail that selectively activates original MLPs and added safety experts. We further introduce a two-stage SFT strategy to strengthen safety discrimination while preserving general capabilities. To enable flexible control at inference time, we introduce a safety temperature mechanism, allowing dynamic adjustment of the trade-off between safety and utility. Experiments across multiple benchmarks, base model, and model scales demonstrate that UpSafe$^\circ$C achieves robust safety improvements against harmful and jailbreak inputs, while maintaining competitive performance on general tasks. Moreover, analysis shows that safety temperature provides fine-grained inference-time control that achieves the Pareto-optimal frontier between utility and safety. Our results highlight a new direction for LLM safety: moving from static alignment toward dynamic, modular, and inference-aware control. |
| 2025-10-02 | [Towards fairer public transit: Real-time tensor-based multimodal fare evasion and fraud detection](http://arxiv.org/abs/2510.02165v1) | Peter Wauyo, Dalia Bwiza et al. | This research introduces a multimodal system designed to detect fraud and fare evasion in public transportation by analyzing closed circuit television (CCTV) and audio data. The proposed solution uses the Vision Transformer for Video (ViViT) model for video feature extraction and the Audio Spectrogram Transformer (AST) for audio analysis. The system implements a Tensor Fusion Network (TFN) architecture that explicitly models unimodal and bimodal interactions through a 2-fold Cartesian product. This advanced fusion technique captures complex cross-modal dynamics between visual behaviors (e.g., tailgating,unauthorized access) and audio cues (e.g., fare transaction sounds). The system was trained and tested on a custom dataset, achieving an accuracy of 89.5%, precision of 87.2%, and recall of 84.0% in detecting fraudulent activities, significantly outperforming early fusion baselines and exceeding the 75% recall rates typically reported in state-of-the-art transportation fraud detection systems. Our ablation studies demonstrate that the tensor fusion approach provides a 7.0% improvement in the F1 score and an 8.8% boost in recall compared to traditional concatenation methods. The solution supports real-time detection, enabling public transport operators to reduce revenue loss, improve passenger safety, and ensure operational compliance. |
| 2025-10-02 | [Mirage Fools the Ear, Mute Hides the Truth: Precise Targeted Adversarial Attacks on Polyphonic Sound Event Detection Systems](http://arxiv.org/abs/2510.02158v1) | Junjie Su, Weifei Jin et al. | Sound Event Detection (SED) systems are increasingly deployed in safety-critical applications such as industrial monitoring and audio surveillance. However, their robustness against adversarial attacks has not been well explored. Existing audio adversarial attacks targeting SED systems, which incorporate both detection and localization capabilities, often lack effectiveness due to SED's strong contextual dependencies or lack precision by focusing solely on misclassifying the target region as the target event, inadvertently affecting non-target regions. To address these challenges, we propose the Mirage and Mute Attack (M2A) framework, which is designed for targeted adversarial attacks on polyphonic SED systems. In our optimization process, we impose specific constraints on the non-target output, which we refer to as preservation loss, ensuring that our attack does not alter the model outputs for non-target region, thus achieving precise attacks. Furthermore, we introduce a novel evaluation metric Editing Precison (EP) that balances effectiveness and precision, enabling our method to simultaneously enhance both. Comprehensive experiments show that M2A achieves 94.56% and 99.11% EP on two state-of-the-art SED models, demonstrating that the framework is sufficiently effective while significantly enhancing attack precision. |
| 2025-10-02 | [Unlocking Vision-Language Models for Video Anomaly Detection via Fine-Grained Prompting](http://arxiv.org/abs/2510.02155v1) | Shu Zou, Xinyu Tian et al. | Prompting has emerged as a practical way to adapt frozen vision-language models (VLMs) for video anomaly detection (VAD). Yet, existing prompts are often overly abstract, overlooking the fine-grained human-object interactions or action semantics that define complex anomalies in surveillance videos. We propose ASK-Hint, a structured prompting framework that leverages action-centric knowledge to elicit more accurate and interpretable reasoning from frozen VLMs. Our approach organizes prompts into semantically coherent groups (e.g. violence, property crimes, public safety) and formulates fine-grained guiding questions that align model predictions with discriminative visual cues. Extensive experiments on UCF-Crime and XD-Violence show that ASK-Hint consistently improves AUC over prior baselines, achieving state-of-the-art performance compared to both fine-tuned and training-free methods. Beyond accuracy, our framework provides interpretable reasoning traces towards anomaly and demonstrates strong generalization across datasets and VLM backbones. These results highlight the critical role of prompt granularity and establish ASK-Hint as a new training-free and generalizable solution for explainable video anomaly detection. |
| 2025-10-02 | [Recurrent Control Barrier Functions: A Path Towards Nonparametric Safety Verification](http://arxiv.org/abs/2510.02127v1) | Jixian Liu, Enrique Mallada | Ensuring the safety of complex dynamical systems often relies on Hamilton-Jacobi (HJ) Reachability Analysis or Control Barrier Functions (CBFs). Both methods require computing a function that characterizes a safe set that can be made (control) invariant. However, the computational burden of solving high-dimensional partial differential equations (for HJ Reachability) or large-scale semidefinite programs (for CBFs) makes finding such functions challenging. In this paper, we introduce the notion of Recurrent Control Barrier Functions (RCBFs), a novel class of CBFs that leverages a recurrent property of the trajectories, i.e., coming back to a safe set, for safety verification. Under mild assumptions, we show that the RCBF condition holds for the signed-distance function, turning function design into set identification. Notably, the resulting set need not be invariant to certify safety. We further propose a data-driven nonparametric method to compute safe sets that is massively parallelizable and trades off conservativeness against computational cost. |
| 2025-10-02 | [Do AI Models Perform Human-like Abstract Reasoning Across Modalities?](http://arxiv.org/abs/2510.02125v1) | Claas Beger, Ryan Yi et al. | OpenAI's o3-preview reasoning model exceeded human accuracy on the ARC-AGI benchmark, but does that mean state-of-the-art models recognize and reason with the abstractions that the task creators intended? We investigate models' abstraction abilities on ConceptARC. We evaluate models under settings that vary the input modality (textual vs. visual), whether the model is permitted to use external Python tools, and, for reasoning models, the amount of reasoning effort. In addition to measuring output accuracy, we perform fine-grained evaluation of the natural-language rules that models generate to explain their solutions. This dual evaluation lets us assess whether models solve tasks using the abstractions ConceptARC was designed to elicit, rather than relying on surface-level patterns. Our results show that, while some models using text-based representations match human output accuracy, the best models' rules are often based on surface-level ``shortcuts'' and capture intended abstractions far less often than humans. Thus their capabilities for general abstract reasoning may be overestimated by evaluations based on accuracy alone. In the visual modality, AI models' output accuracy drops sharply, yet our rule-level analysis reveals that models might be underestimated, as they still exhibit a substantial share of rules that capture intended abstractions, but are often unable to correctly apply these rules. In short, our results show that models still lag humans in abstract reasoning, and that using accuracy alone to evaluate abstract reasoning on ARC-like tasks may overestimate abstract-reasoning capabilities in textual modalities and underestimate it in visual modalities. We believe that our evaluation framework offers a more faithful picture of multimodal models' abstract reasoning abilities and a more principled way to track progress toward human-like, abstraction-centered intelligence. |
| 2025-10-02 | [A Copula-Based Variational Autoencoder for Uncertainty Quantification in Inverse Problems: Application to Damage Identification in an Offshore Wind Turbine](http://arxiv.org/abs/2510.02013v1) | Ana Fernandez-Navamuel, Matteo Croci et al. | Structural Health Monitoring of Floating Offshore Wind Turbines (FOWTs) is critical for ensuring operational safety and efficiency. However, identifying damage in components like mooring systems from limited sensor data poses a challenging inverse problem, often characterized by multimodal solutions where various damage states could explain the observed response. To overcome it, we propose a Variational Autoencoder (VAE) architecture, where the encoder approximates the inverse operator, while the decoder approximates the forward. The posterior distribution of the latent space variables is probabilistically modeled, describing the uncertainties in the estimates. This work tackles the limitations of conventional Gaussian Mixtures used within VAEs, which can be either too restrictive or computationally prohibitive for high-dimensional spaces. We propose a novel Copula-based VAE architecture that decouples the marginal distribution of the variables from their dependence structure, offering a flexible method for representing complex, correlated posterior distributions. We provide a comprehensive comparison of three different approaches for approximating the posterior: a Gaussian Mixture with a diagonal covariance matrix, a Gaussian Mixture with a full covariance matrix, and a Gaussian Copula. Our analysis, conducted on a high-fidelity synthetic dataset, demonstrates that the Copula VAE offers a promising and tractable solution in high-dimensional spaces. Although the present work remains in the two-dimensional space, the results suggest efficient scalability to higher dimensions. It achieves superior performance with significantly fewer parameters than the Gaussian Mixture alternatives, whose parametrization grows prohibitively with the dimensionality. The results underscore the potential of Copula-based VAEs as a tool for uncertainty-aware damage identification in FOWT mooring systems. |
| 2025-10-02 | [LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target](http://arxiv.org/abs/2510.01995v1) | Md Arid Hasan, Firoj Alam et al. | Online social media platforms are central to everyday communication and information seeking. While these platforms serve positive purposes, they also provide fertile ground for the spread of hate speech, offensive language, and bullying content targeting individuals, organizations, and communities. Such content undermines safety, participation, and equity online. Reliable detection systems are therefore needed, especially for low-resource languages where moderation tools are limited. In Bangla, prior work has contributed resources and models, but most are single-task (e.g., binary hate/offense) with limited coverage of multi-facet signals (type, severity, target). We address these gaps by introducing the first multi-task Bangla hate-speech dataset, BanglaMultiHate, one of the largest manually annotated corpus to date. Building on this resource, we conduct a comprehensive, controlled comparison spanning classical baselines, monolingual pretrained models, and LLMs under zero-shot prompting and LoRA fine-tuning. Our experiments assess LLM adaptability in a low-resource setting and reveal a consistent trend: although LoRA-tuned LLMs are competitive with BanglaBERT, culturally and linguistically grounded pretraining remains critical for robust performance. Together, our dataset and findings establish a stronger benchmark for developing culturally aligned moderation tools in low-resource contexts. For reproducibility, we will release the dataset and all related scripts. |
| 2025-10-02 | [Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors](http://arxiv.org/abs/2510.01934v1) | Guangyao Zhai, Yue Zhou et al. | Few-shot anomaly detection streamlines and simplifies industrial safety inspection. However, limited samples make accurate differentiation between normal and abnormal features challenging, and even more so under category-agnostic conditions. Large-scale pre-training of foundation visual encoders has advanced many fields, as the enormous quantity of data helps to learn the general distribution of normal images. We observe that the anomaly amount in an image directly correlates with the difference in the learnt embeddings and utilize this to design a few-shot anomaly detector termed FoundAD. This is done by learning a nonlinear projection operator onto the natural image manifold. The simple operator acts as an effective tool for anomaly detection to characterize and identify out-of-distribution regions in an image. Extensive experiments show that our approach supports multi-class detection and achieves competitive performance while using substantially fewer parameters than prior methods. Backed up by evaluations with multiple foundation encoders, including fresh DINOv3, we believe this idea broadens the perspective on foundation features and advances the field of few-shot anomaly detection. |
| 2025-10-02 | [Learning Representations Through Contrastive Neural Model Checking](http://arxiv.org/abs/2510.01853v1) | Vladimir Krsmanovic, Matthias Cosler et al. | Model checking is a key technique for verifying safety-critical systems against formal specifications, where recent applications of deep learning have shown promise. However, while ubiquitous for vision and language domains, representation learning remains underexplored in formal verification. We introduce Contrastive Neural Model Checking (CNML), a novel method that leverages the model checking task as a guiding signal for learning aligned representations. CNML jointly embeds logical specifications and systems into a shared latent space through a self-supervised contrastive objective. On industry-inspired retrieval tasks, CNML considerably outperforms both algorithmic and neural baselines in cross-modal and intra-modal settings.We further show that the learned representations effectively transfer to downstream tasks and generalize to more complex formulas. These findings demonstrate that model checking can serve as an objective for learning representations for formal languages. |
| 2025-10-02 | [Human-AI Teaming Co-Learning in Military Operations](http://arxiv.org/abs/2510.01815v1) | Clara Maathuis, Kasper Cools | In a time of rapidly evolving military threats and increasingly complex operational environments, the integration of AI into military operations proves significant advantages. At the same time, this implies various challenges and risks regarding building and deploying human-AI teaming systems in an effective and ethical manner. Currently, understanding and coping with them are often tackled from an external perspective considering the human-AI teaming system as a collective agent. Nevertheless, zooming into the dynamics involved inside the system assures dealing with a broader palette of relevant multidimensional responsibility, safety, and robustness aspects. To this end, this research proposes the design of a trustworthy co-learning model for human-AI teaming in military operations that encompasses a continuous and bidirectional exchange of insights between the human and AI agents as they jointly adapt to evolving battlefield conditions. It does that by integrating four dimensions. First, adjustable autonomy for dynamically calibrating the autonomy levels of agents depending on aspects like mission state, system confidence, and environmental uncertainty. Second, multi-layered control which accounts continuous oversight, monitoring of activities, and accountability. Third, bidirectional feedback with explicit and implicit feedback loops between the agents to assure a proper communication of reasoning, uncertainties, and learned adaptations that each of the agents has. And fourth, collaborative decision-making which implies the generation, evaluation, and proposal of decisions associated with confidence levels and rationale behind them. The model proposed is accompanied by concrete exemplifications and recommendations that contribute to further developing responsible and trustworthy human-AI teaming systems in military operations. |
| 2025-10-02 | [MedQ-Bench: Evaluating and Exploring Medical Image Quality Assessment Abilities in MLLMs](http://arxiv.org/abs/2510.01691v1) | Jiyao Liu, Jinjie Wei et al. | Medical Image Quality Assessment (IQA) serves as the first-mile safety gate for clinical AI, yet existing approaches remain constrained by scalar, score-based metrics and fail to reflect the descriptive, human-like reasoning process central to expert evaluation. To address this gap, we introduce MedQ-Bench, a comprehensive benchmark that establishes a perception-reasoning paradigm for language-based evaluation of medical image quality with Multi-modal Large Language Models (MLLMs). MedQ-Bench defines two complementary tasks: (1) MedQ-Perception, which probes low-level perceptual capability via human-curated questions on fundamental visual attributes; and (2) MedQ-Reasoning, encompassing both no-reference and comparison reasoning tasks, aligning model evaluation with human-like reasoning on image quality. The benchmark spans five imaging modalities and over forty quality attributes, totaling 2,600 perceptual queries and 708 reasoning assessments, covering diverse image sources including authentic clinical acquisitions, images with simulated degradations via physics-based reconstructions, and AI-generated images. To evaluate reasoning ability, we propose a multi-dimensional judging protocol that assesses model outputs along four complementary axes. We further conduct rigorous human-AI alignment validation by comparing LLM-based judgement with radiologists. Our evaluation of 14 state-of-the-art MLLMs demonstrates that models exhibit preliminary but unstable perceptual and reasoning skills, with insufficient accuracy for reliable clinical use. These findings highlight the need for targeted optimization of MLLMs in medical IQA. We hope that MedQ-Bench will catalyze further exploration and unlock the untapped potential of MLLMs for medical image quality evaluation. |
| 2025-10-02 | [Uncovering Overconfident Failures in CXR Models via Augmentation-Sensitivity Risk Scoring](http://arxiv.org/abs/2510.01683v1) | Han-Jay Shu, Wei-Ning Chiu et al. | Deep learning models achieve strong performance in chest radiograph (CXR) interpretation, yet fairness and reliability concerns persist. Models often show uneven accuracy across patient subgroups, leading to hidden failures not reflected in aggregate metrics. Existing error detection approaches -- based on confidence calibration or out-of-distribution (OOD) detection -- struggle with subtle within-distribution errors, while image- and representation-level consistency-based methods remain underexplored in medical imaging. We propose an augmentation-sensitivity risk scoring (ASRS) framework to identify error-prone CXR cases. ASRS applies clinically plausible rotations ($\pm 15^\circ$/$\pm 30^\circ$) and measures embedding shifts with the RAD-DINO encoder. Sensitivity scores stratify samples into stability quartiles, where highly sensitive cases show substantially lower recall ($-0.2$ to $-0.3$) despite high AUROC and confidence. ASRS provides a label-free means for selective prediction and clinician review, improving fairness and safety in medical AI. |
| 2025-10-02 | [A Locally Executable AI System for Improving Preoperative Patient Communication: A Multi-Domain Clinical Evaluation](http://arxiv.org/abs/2510.01671v1) | Motoki Sato, Yuki Matsushita et al. | Patients awaiting invasive procedures often have unanswered pre-procedural questions; however, time-pressured workflows and privacy constraints limit personalized counseling. We present LENOHA (Low Energy, No Hallucination, Leave No One Behind Architecture), a safety-first, local-first system that routes inputs with a high-precision sentence-transformer classifier and returns verbatim answers from a clinician-curated FAQ for clinical queries, eliminating free-text generation in the clinical path. We evaluated two domains (tooth extraction and gastroscopy) using expert-reviewed validation sets (n=400/domain) for thresholding and independent test sets (n=200/domain). Among the four encoders, E5-large-instruct (560M) achieved an overall accuracy of 0.983 (95% CI 0.964-0.991), AUC 0.996, and seven total errors, which were statistically indistinguishable from GPT-4o on this task; Gemini made no errors on this test set. Energy logging shows that the non-generative clinical path consumes ~1.0 mWh per input versus ~168 mWh per small-talk reply from a local 8B SLM, a ~170x difference, while maintaining ~0.10 s latency on a single on-prem GPU. These results indicate that near-frontier discrimination and generation-induced errors are structurally avoided in the clinical path by returning vetted FAQ answers verbatim, supporting privacy, sustainability, and equitable deployment in bandwidth-limited environments. |
| 2025-10-02 | [Just Do It!? Computer-Use Agents Exhibit Blind Goal-Directedness](http://arxiv.org/abs/2510.01670v1) | Erfan Shayegani, Keegan Hines et al. | Computer-Use Agents (CUAs) are an increasingly deployed class of agents that take actions on GUIs to accomplish user goals. In this paper, we show that CUAs consistently exhibit Blind Goal-Directedness (BGD): a bias to pursue goals regardless of feasibility, safety, reliability, or context. We characterize three prevalent patterns of BGD: (i) lack of contextual reasoning, (ii) assumptions and decisions under ambiguity, and (iii) contradictory or infeasible goals. We develop BLIND-ACT, a benchmark of 90 tasks capturing these three patterns. Built on OSWorld, BLIND-ACT provides realistic environments and employs LLM-based judges to evaluate agent behavior, achieving 93.75% agreement with human annotations. We use BLIND-ACT to evaluate nine frontier models, including Claude Sonnet and Opus 4, Computer-Use-Preview, and GPT-5, observing high average BGD rates (80.8%) across them. We show that BGD exposes subtle risks that arise even when inputs are not directly harmful. While prompting-based interventions lower BGD levels, substantial risk persists, highlighting the need for stronger training- or inference-time interventions. Qualitative analysis reveals observed failure modes: execution-first bias (focusing on how to act over whether to act), thought-action disconnect (execution diverging from reasoning), and request-primacy (justifying actions due to user request). Identifying BGD and introducing BLIND-ACT establishes a foundation for future research on studying and mitigating this fundamental risk and ensuring safe CUA deployment. |
| 2025-10-02 | [NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BERT](http://arxiv.org/abs/2510.01644v1) | John Hawkins, Aditya Pramar et al. | Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention. |
| 2025-10-02 | [Dedicated-frequency analysis of gravitational-wave bursts from core-collapse supernovae with minimal assumptions](http://arxiv.org/abs/2510.01614v1) | Yi Shuen C. Lee, Marek J Szczepańczyk et al. | Gravitational-wave (GW) emissions from core-collapse supernovae (CCSNe) provide insights into the internal processes leading up to their explosions. Theory predicts that CCSN explosions are driven by hydrodynamical instabilities like the standing accretion shock instability (SASI) or neutrino-driven convection, and simulations show that these mechanisms emit GWs at low frequencies ($\lesssim 0.25 \,{\rm kHz}$). Thus the detection of low-frequency GWs, or lack thereof, is useful for constraining explosion mechanisms in CCSNe. This paper introduces the dedicated-frequency framework, which is designed to follow-up GW burst detections using bandpass analyses. The primary aim is to study whether low-frequency (LF) follow-up analyses, limited to $\leq 256 \,{\rm Hz}$, constrain CCSN explosion models in practical observing scenarios. The analysis dataset comprises waveforms from five CCSN models with different strengths of low-frequency GW emissions induced by SASI and/or neutrino-driven convection, injected into the Advanced LIGO data from the Third Observing Run (O3). Eligible candidates for the LF follow-up must satisfy a benchmark detection significance and are identified using the coherent WaveBurst (cWB) algorithm. The LF follow-up analyses are performed using the BayesWave algorithm. Both cWB and BayesWave make minimal assumptions about the signal's morphology. The results suggest that the successful detection of a CCSN in the LF follow-up analysis constrains its explosion mechanism. The dedicated-frequency framework also has other applications. As a demonstration, the loudest trigger from the SN 2019fcn supernova search is followed-up using a high-frequency (HF) analysis, limited to $\geq 256 \,{\rm Hz}$. The trigger has negligible power below $256 \, {\rm Hz}$, and the HF analysis successfully enhances its detection significance. |

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



