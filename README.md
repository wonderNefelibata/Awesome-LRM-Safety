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
| 2025-10-07 | [Conformalized Gaussian processes for online uncertainty quantification over graphs](http://arxiv.org/abs/2510.06181v1) | Jinwen Xu, Qin Lu et al. | Uncertainty quantification (UQ) over graphs arises in a number of safety-critical applications in network science. The Gaussian process (GP), as a classical Bayesian framework for UQ, has been developed to handle graph-structured data by devising topology-aware kernel functions. However, such GP-based approaches are limited not only by the prohibitive computational complexity, but also the strict modeling assumptions that might yield poor coverage, especially with labels arriving on the fly. To effect scalability, we devise a novel graph-aware parametric GP model by leveraging the random feature (RF)-based kernel approximation, which is amenable to efficient recursive Bayesian model updates. To further allow for adaptivity, an ensemble of graph-aware RF-based scalable GPs have been leveraged, with per-GP weight adapted to data arriving incrementally. To ensure valid coverage with robustness to model mis-specification, we wed the GP-based set predictors with the online conformal prediction framework, which post-processes the prediction sets using adaptive thresholds. Experimental results the proposed method yields improved coverage and efficient prediction sets over existing baselines by adaptively ensembling the GP models and setting the key threshold parameters in CP. |
| 2025-10-07 | [Towards Autonomous Tape Handling for Robotic Wound Redressing](http://arxiv.org/abs/2510.06127v1) | Xiao Liang, Lu Shen et al. | Chronic wounds, such as diabetic, pressure, and venous ulcers, affect over 6.5 million patients in the United States alone and generate an annual cost exceeding \$25 billion. Despite this burden, chronic wound care remains a routine yet manual process performed exclusively by trained clinicians due to its critical safety demands. We envision a future in which robotics and automation support wound care to lower costs and enhance patient outcomes. This paper introduces an autonomous framework for one of the most fundamental yet challenging subtasks in wound redressing: adhesive tape manipulation. Specifically, we address two critical capabilities: tape initial detachment (TID) and secure tape placement. To handle the complex adhesive dynamics of detachment, we propose a force-feedback imitation learning approach trained from human teleoperation demonstrations. For tape placement, we develop a numerical trajectory optimization method based to ensure smooth adhesion and wrinkle-free application across diverse anatomical surfaces. We validate these methods through extensive experiments, demonstrating reliable performance in both quantitative evaluations and integrated wound redressing pipelines. Our results establish tape manipulation as an essential step toward practical robotic wound care automation. |
| 2025-10-07 | [The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives](http://arxiv.org/abs/2510.06096v1) | Matthieu Bou, Nyal Patel et al. | The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI. |
| 2025-10-07 | [Learning from Failures: Understanding LLM Alignment through Failure-Aware Inverse RL](http://arxiv.org/abs/2510.06092v1) | Nyal Patel, Matthieu Bou et al. | Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with human preferences, yet the underlying reward signals they internalize remain hidden, posing a critical challenge for interpretability and safety. Existing approaches attempt to extract these latent incentives using Inverse Reinforcement Learning (IRL), but treat all preference pairs equally, often overlooking the most informative signals: those examples the extracted reward model misclassifies or assigns nearly equal scores, which we term \emph{failures}. We introduce a novel \emph{failure-aware} IRL algorithm that focuses on misclassified or difficult examples to recover the latent rewards defining model behaviors. By learning from these failures, our failure-aware IRL extracts reward functions that better reflect the true objectives behind RLHF. We demonstrate that failure-aware IRL outperforms existing IRL baselines across multiple metrics when applied to LLM detoxification, without requiring external classifiers or supervision. Crucially, failure-aware IRL yields rewards that better capture the true incentives learned during RLHF, enabling more effective re-RLHF training than standard IRL. This establishes failure-aware IRL as a robust, scalable method for auditing model alignment and reducing ambiguity in the IRL process. |
| 2025-10-07 | [Refusal Falls off a Cliff: How Safety Alignment Fails in Reasoning?](http://arxiv.org/abs/2510.06036v1) | Qingyu Yin, Chak Tou Leong et al. | Large reasoning models (LRMs) with multi-step reasoning capabilities have shown remarkable problem-solving abilities, yet they exhibit concerning safety vulnerabilities that remain poorly understood. In this work, we investigate why safety alignment fails in reasoning models through a mechanistic interpretability lens. Using a linear probing approach to trace refusal intentions across token positions, we discover a striking phenomenon termed as \textbf{refusal cliff}: many poorly-aligned reasoning models correctly identify harmful prompts and maintain strong refusal intentions during their thinking process, but experience a sharp drop in refusal scores at the final tokens before output generation. This suggests that these models are not inherently unsafe; rather, their refusal intentions are systematically suppressed. Through causal intervention analysis, we identify a sparse set of attention heads that negatively contribute to refusal behavior. Ablating just 3\% of these heads can reduce attack success rates below 10\%. Building on these mechanistic insights, we propose \textbf{Cliff-as-a-Judge}, a novel data selection method that identifies training examples exhibiting the largest refusal cliff to efficiently repair reasoning models' safety alignment. This approach achieves comparable safety improvements using only 1.7\% of the vanilla safety training data, demonstrating a less-is-more effect in safety alignment. |
| 2025-10-07 | [Out-of-Distribution Detection from Small Training Sets using Bayesian Neural Network Classifiers](http://arxiv.org/abs/2510.06025v1) | Kevin Raina, Tanya Schmah | Out-of-Distribution (OOD) detection is critical to AI reliability and safety, yet in many practical settings, only a limited amount of training data is available. Bayesian Neural Networks (BNNs) are a promising class of model on which to base OOD detection, because they explicitly represent epistemic (i.e. model) uncertainty. In the small training data regime, BNNs are especially valuable because they can incorporate prior model information. We introduce a new family of Bayesian posthoc OOD scores based on expected logit vectors, and compare 5 Bayesian and 4 deterministic posthoc OOD scores. Experiments on MNIST and CIFAR-10 In-Distributions, with 5000 training samples or less, show that the Bayesian methods outperform corresponding deterministic methods. |
| 2025-10-07 | [AI-Enabled Capabilities to Facilitate Next-Generation Rover Surface Operations](http://arxiv.org/abs/2510.05985v1) | Cristina Luna, Robert Field et al. | Current planetary rovers operate at traverse speeds of approximately 10 cm/s, fundamentally limiting exploration efficiency. This work presents integrated AI systems which significantly improve autonomy through three components: (i) the FASTNAV Far Obstacle Detector (FOD), capable of facilitating sustained 1.0 m/s speeds via computer vision-based obstacle detection; (ii) CISRU, a multi-robot coordination framework enabling human-robot collaboration for in-situ resource utilisation; and (iii) the ViBEKO and AIAXR deep learning-based terrain classification studies. Field validation in Mars analogue environments demonstrated these systems at Technology Readiness Level 4, providing measurable improvements in traverse speed, classification accuracy, and operational safety for next-generation planetary missions. |
| 2025-10-07 | [Diffusion Models for Low-Light Image Enhancement: A Multi-Perspective Taxonomy and Performance Analysis](http://arxiv.org/abs/2510.05976v1) | Eashan Adhikarla, Yixin Liu et al. | Low-light image enhancement (LLIE) is vital for safety-critical applications such as surveillance, autonomous navigation, and medical imaging, where visibility degradation can impair downstream task performance. Recently, diffusion models have emerged as a promising generative paradigm for LLIE due to their capacity to model complex image distributions via iterative denoising. This survey provides an up-to-date critical analysis of diffusion models for LLIE, distinctively featuring an in-depth comparative performance evaluation against Generative Adversarial Network and Transformer-based state-of-the-art methods, a thorough examination of practical deployment challenges, and a forward-looking perspective on the role of emerging paradigms like foundation models. We propose a multi-perspective taxonomy encompassing six categories: Intrinsic Decomposition, Spectral & Latent, Accelerated, Guided, Multimodal, and Autonomous; that map enhancement methods across physical priors, conditioning schemes, and computational efficiency. Our taxonomy is grounded in a hybrid view of both the model mechanism and the conditioning signals. We evaluate qualitative failure modes, benchmark inconsistencies, and trade-offs between interpretability, generalization, and inference efficiency. We also discuss real-world deployment constraints (e.g., memory, energy use) and ethical considerations. This survey aims to guide the next generation of diffusion-based LLIE research by highlighting trends and surfacing open research questions, including novel conditioning, real-time adaptation, and the potential of foundation models. |
| 2025-10-07 | [LexiCon: a Benchmark for Planning under Temporal Constraints in Natural Language](http://arxiv.org/abs/2510.05972v1) | Periklis Mantenoglou, Rishi Hazra et al. | Owing to their reasoning capabilities, large language models (LLMs) have been evaluated on planning tasks described in natural language. However, LLMs have largely been tested on planning domains without constraints. In order to deploy them in real-world settings where adherence to constraints, in particular safety constraints, is critical, we need to evaluate their performance on constrained planning tasks. We introduce LexiCon -- a natural language-based (Lexi) constrained (Con) planning benchmark, consisting of a suite of environments, that can be used to evaluate the planning capabilities of LLMs in a principled fashion. The core idea behind LexiCon is to take existing planning environments and impose temporal constraints on the states. These constrained problems are then translated into natural language and given to an LLM to solve. A key feature of LexiCon is its extensibility. That is, the set of supported environments can be extended with new (unconstrained) environment generators, for which temporal constraints are constructed automatically. This renders LexiCon future-proof: the hardness of the generated planning problems can be increased as the planning capabilities of LLMs improve. Our experiments reveal that the performance of state-of-the-art LLMs, including reasoning models like GPT-5, o3, and R1, deteriorates as the degree of constrainedness of the planning tasks increases. |
| 2025-10-07 | [Distributed Platoon Control Under Quantization: Stability Analysis and Privacy Preservation](http://arxiv.org/abs/2510.05959v1) | Kaixiang Zhang, Zhaojian Li et al. | Distributed control of connected and automated vehicles has attracted considerable interest for its potential to improve traffic efficiency and safety. However, such control schemes require sharing privacy-sensitive vehicle data, which introduces risks of information leakage and potential malicious activities. This paper investigates the stability and privacy-preserving properties of distributed platoon control under two types of quantizers: deterministic and probabilistic. For deterministic quantization, we show that the resulting control strategy ensures the system errors remain uniformly ultimately bounded. Moreover, in the absence of auxiliary information, an eavesdropper cannot uniquely infer sensitive vehicle states. In contrast, the use of probabilistic quantization enables asymptotic convergence of the vehicle platoon in expectation with bounded variance. Importantly, probabilistic quantizers can satisfy differential privacy guarantees, thereby preserving privacy even when the eavesdropper possesses arbitrary auxiliary information. We further analyze the trade-off between control performance and privacy by formulating an optimization problem that characterizes the impact of the quantization step on both metrics. Numerical simulations are provided to illustrate the performance differences between the two quantization strategies. |
| 2025-10-07 | [Safe Landing on Small Celestial Bodies with Gravitational Uncertainty Using Disturbance Estimation and Control Barrier Functions](http://arxiv.org/abs/2510.05895v1) | Felipe Arenas-Uribe, T. Michael Seigler et al. | Soft landing on small celestial bodies (SCBs) poses unique challenges, as uncertainties in gravitational models and poorly characterized, dynamic environments require a high level of autonomy. Existing control approaches lack formal guarantees for safety constraint satisfaction, necessary to ensure the safe execution of the maneuvers. This paper introduces a control that addresses this limitation by integrating trajectory tracking, disturbance estimation, and safety enforcement. An extended high-gain observer is employed to estimate disturbances resulting from gravitational model uncertainties. We then apply a feedback-linearizing and disturbance-canceling controller that achieves exponential tracking of reference trajectories. Finally, we use a control barrier function based minimum-intervention controller to enforce state and input constraints through out the maneuver execution. This control combines trajectory tracking of offline generated reference trajectories with formal guarantees of safety, which follows common guidance and control architectures for spacecraft and allows aggressive maneuvers to be executed without compromising safety. Numerical simulations using fuel-optimal trajectories demonstrate the effectiveness of the controller in achieving precise and safe soft-landing, highlighting its potential for autonomous SCB missions. |
| 2025-10-07 | [The Safety Challenge of World Models for Embodied AI Agents: A Review](http://arxiv.org/abs/2510.05865v1) | Lorenzo Baraldi, Zifan Zeng et al. | The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results. |
| 2025-10-07 | [Evaluating the Sensitivity of LLMs to Harmful Contents in Long Input](http://arxiv.org/abs/2510.05864v1) | Faeze Ghorbanpour, Alexander Fraser | Large language models (LLMs) increasingly support applications that rely on extended context, from document processing to retrieval-augmented generation. While their long-context capabilities are well studied for reasoning and retrieval, little is known about their behavior in safety-critical scenarios. We evaluate LLMs' sensitivity to harmful content under extended context, varying type (explicit vs. implicit), position (beginning, middle, end), prevalence (0.01-0.50 of the prompt), and context length (600-6000 tokens). Across harmful content categories such as toxic, offensive, and hate speech, with LLaMA-3, Qwen-2.5, and Mistral, we observe similar patterns: performance peaks at moderate harmful prevalence (0.25) but declines when content is very sparse or dominant; recall decreases with increasing context length; harmful sentences at the beginning are generally detected more reliably; and explicit content is more consistently recognized than implicit. These findings provide the first systematic view of how LLMs prioritize and calibrate harmful content in long contexts, highlighting both their emerging strengths and the challenges that remain for safety-critical use. |
| 2025-10-07 | [Risk level dependent Minimax Quantile lower bounds for Interactive Statistical Decision Making](http://arxiv.org/abs/2510.05808v1) | Raghav Bongole, Amirreza Zamani et al. | Minimax risk and regret focus on expectation, missing rare failures critical in safety-critical bandits and reinforcement learning. Minimax quantiles capture these tails. Three strands of prior work motivate this study: minimax-quantile bounds restricted to non-interactive estimation; unified interactive analyses that focus on expected risk rather than risk level specific quantile bounds; and high-probability bandit bounds that still lack a quantile-specific toolkit for general interactive protocols. To close this gap, within the interactive statistical decision making framework, we develop high-probability Fano and Le Cam tools and derive risk level explicit minimax-quantile bounds, including a quantile-to-expectation conversion and a tight link between strict and lower minimax quantiles. Instantiating these results for the two-armed Gaussian bandit immediately recovers optimal-rate bounds. |
| 2025-10-07 | [Primal-Dual Direct Preference Optimization for Constrained LLM Alignment](http://arxiv.org/abs/2510.05703v1) | Yihan Du, Seo Taek Kong et al. | The widespread application of Large Language Models (LLMs) imposes increasing demands on safety, such as reducing harmful content and fake information, and avoiding certain forbidden tokens due to rules and laws. While there have been several recent works studying safe alignment of LLMs, these works either require the training of reward and cost models and incur high memory and computational costs, or need prior knowledge about the optimal solution. Motivated by this fact, we study the problem of constrained alignment in LLMs, i.e., maximizing the output reward while restricting the cost due to potentially unsafe content to stay below a threshold. For this problem, we propose a novel primal-dual DPO approach, which first trains a model using standard DPO on reward preference data to provide reward information, and then adopts a rearranged Lagrangian DPO objective utilizing the provided reward information to fine-tune LLMs on cost preference data. Our approach significantly reduces memory and computational costs, and does not require extra prior knowledge. Moreover, we establish rigorous theoretical guarantees on the suboptimality and constraint violation of the output policy. We also extend our approach to an online data setting by incorporating exploration bonuses, which enables our approach to explore uncovered prompt-response space, and then provide theoretical results that get rid of the dependence on preference data coverage. Experimental results on the widely-used preference dataset PKU-SafeRLHF demonstrate the effectiveness of our approach. |
| 2025-10-07 | [vAttention: Verified Sparse Attention](http://arxiv.org/abs/2510.05688v1) | Aditya Desai, Kumar Krishna Agrawal et al. | State-of-the-art sparse attention methods for reducing decoding latency fall into two main categories: approximate top-$k$ (and its extension, top-$p$) and recently introduced sampling-based estimation. However, these approaches are fundamentally limited in their ability to approximate full attention: they fail to provide consistent approximations across heads and query vectors and, most critically, lack guarantees on approximation quality, limiting their practical deployment. We observe that top-$k$ and random sampling are complementary: top-$k$ performs well when attention scores are dominated by a few tokens, whereas random sampling provides better estimates when attention scores are relatively uniform. Building on this insight and leveraging the statistical guarantees of sampling, we introduce vAttention, the first practical sparse attention mechanism with user-specified $(\epsilon, \delta)$ guarantees on approximation accuracy (thus, verified). These guarantees make vAttention a compelling step toward practical, reliable deployment of sparse attention at scale. By unifying top-k and sampling, vAttention outperforms both individually, delivering a superior quality-efficiency trade-off. Our experiments show that vAttention significantly improves the quality of sparse attention (e.g., $\sim$4.5 percentage points for Llama-3.1-8B-Inst and Deepseek-R1-Distill-Llama-8B on RULER-HARD), and effectively bridges the gap between full and sparse attention (e.g., across datasets, it matches full model quality with upto 20x sparsity). We also demonstrate that it can be deployed in reasoning scenarios to achieve fast decoding without compromising model quality (e.g., vAttention achieves full model quality on AIME2024 at 10x sparsity with up to 32K token generations). Code is open-sourced at https://github.com/xAlg-ai/sparse-attention-hub. |
| 2025-10-07 | [ARRC: Advanced Reasoning Robot Control - Knowledge-Driven Autonomous Manipulation Using Retrieval-Augmented Generation](http://arxiv.org/abs/2510.05547v1) | Eugene Vorobiov, Ammar Jaleel Mahmood et al. | We present ARRC (Advanced Reasoning Robot Control), a practical system that connects natural-language instructions to safe local robotic control by combining Retrieval-Augmented Generation (RAG) with RGB-D perception and guarded execution on an affordable robot arm. The system indexes curated robot knowledge (movement patterns, task templates, and safety heuristics) in a vector database, retrieves task-relevant context for each instruction, and conditions a large language model (LLM) to produce JSON-structured action plans. Plans are executed on a UFactory xArm 850 fitted with a Dynamixel-driven parallel gripper and an Intel RealSense D435 camera. Perception uses AprilTag detections fused with depth to produce object-centric metric poses. Execution is enforced via software safety gates: workspace bounds, speed and force caps, timeouts, and bounded retries. We describe the architecture, knowledge design, integration choices, and a reproducible evaluation protocol for tabletop scan, approach, and pick-place tasks. Experimental results demonstrate the efficacy of the proposed approach. Our design shows that RAG-based planning can substantially improve plan validity and adaptability while keeping perception and low-level control local to the robot. |
| 2025-10-07 | [The New Quant: A Survey of Large Language Models in Financial Prediction and Trading](http://arxiv.org/abs/2510.05533v1) | Weilong Fu | Large language models are reshaping quantitative investing by turning unstructured financial information into evidence-grounded signals and executable decisions. This survey synthesizes research with a focus on equity return prediction and trading, consolidating insights from domain surveys and more than fifty primary studies. We propose a task-centered taxonomy that spans sentiment and event extraction, numerical and economic reasoning, multimodal understanding, retrieval-augmented generation, time series prompting, and agentic systems that coordinate tools for research, backtesting, and execution. We review empirical evidence for predictability, highlight design patterns that improve faithfulness such as retrieval first prompting and tool-verified numerics, and explain how signals feed portfolio construction under exposure, turnover, and capacity controls. We assess benchmarks and datasets for prediction and trading and outline desiderata-for time safe and economically meaningful evaluation that reports costs, latency, and capacity. We analyze challenges that matter in production, including temporal leakage, hallucination, data coverage and structure, deployment economics, interpretability, governance, and safety. The survey closes with recommendations for standardizing evaluation, building auditable pipelines, and advancing multilingual and cross-market research so that language-driven systems deliver robust and risk-controlled performance in practice. |
| 2025-10-07 | [KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG for Safety-Critical Aviation Maintenance](http://arxiv.org/abs/2510.05524v1) | Kuangshi Ai, Jonathan A. Karr Jr et al. | We present Knowledge Extraction on OMIn (KEO), a domain-specific knowledge extraction and reasoning framework with large language models (LLMs) in safety-critical contexts. Using the Operations and Maintenance Intelligence (OMIn) dataset, we construct a QA benchmark spanning global sensemaking and actionable maintenance tasks. KEO builds a structured Knowledge Graph (KG) and integrates it into a retrieval-augmented generation (RAG) pipeline, enabling more coherent, dataset-wide reasoning than traditional text-chunk RAG. We evaluate locally deployable LLMs (Gemma-3, Phi-4, Mistral-Nemo) and employ stronger models (GPT-4o, Llama-3.3) as judges. Experiments show that KEO markedly improves global sensemaking by revealing patterns and system-level insights, while text-chunk RAG remains effective for fine-grained procedural tasks requiring localized retrieval. These findings underscore the promise of KG-augmented LLMs for secure, domain-specific QA and their potential in high-stakes reasoning. |
| 2025-10-07 | [Assessing Human Rights Risks in AI: A Framework for Model Evaluation](http://arxiv.org/abs/2510.05519v1) | Vyoma Raman, Camille Chabot et al. | The Universal Declaration of Human Rights and other international agreements outline numerous inalienable rights that apply across geopolitical boundaries. As generative AI becomes increasingly prevalent, it poses risks to human rights such as non-discrimination, health, and security, which are also central concerns for AI researchers focused on fairness and safety. We contribute to the field of algorithmic auditing by presenting a framework to computationally assess human rights risk. Drawing on the UN Guiding Principles on Business and Human Rights, we develop an approach to evaluating a model to make grounded claims about the level of risk a model poses to particular human rights. Our framework consists of three parts: selecting tasks that are likely to pose human rights risks within a given context, designing metrics to measure the scope, scale, and likelihood of potential risks from that task, and analyzing rights with respect to the values of those metrics. Because a human rights approach centers on real-world harms, it requires evaluating AI systems in the specific contexts in which they are deployed. We present a case study of large language models in political news journalism, demonstrating how our framework helps to design an evaluation and benchmarking different models. We then discuss the implications of the results for the rights of access to information and freedom of thought and broader considerations for adopting this approach. |

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



