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
| 2026-02-24 | [Not Just How Much, But Where: Decomposing Epistemic Uncertainty into Per-Class Contributions](http://arxiv.org/abs/2602.21160v1) | Mame Diarra Toure, David A. Stephens | In safety-critical classification, the cost of failure is often asymmetric, yet Bayesian deep learning summarises epistemic uncertainty with a single scalar, mutual information (MI), that cannot distinguish whether a model's ignorance involves a benign or safety-critical class. We decompose MI into a per-class vector $C_k(x)=σ_k^{2}/(2μ_k)$, with $μ_k{=}\mathbb{E}[p_k]$ and $σ_k^2{=}\mathrm{Var}[p_k]$ across posterior samples. The decomposition follows from a second-order Taylor expansion of the entropy; the $1/μ_k$ weighting corrects boundary suppression and makes $C_k$ comparable across rare and common classes. By construction $\sum_k C_k \approx \mathrm{MI}$, and a companion skewness diagnostic flags inputs where the approximation degrades. After characterising the axiomatic properties of $C_k$, we validate it on three tasks: (i) selective prediction for diabetic retinopathy, where critical-class $C_k$ reduces selective risk by 34.7\% over MI and 56.2\% over variance baselines; (ii) out-of-distribution detection on clinical and image benchmarks, where $\sum_k C_k$ achieves the highest AUROC and the per-class view exposes asymmetric shifts invisible to MI; and (iii) a controlled label-noise study in which $\sum_k C_k$ shows less sensitivity to injected aleatoric noise than MI under end-to-end Bayesian training, while both metrics degrade under transfer learning. Across all tasks, the quality of the posterior approximation shapes uncertainty at least as strongly as the choice of metric, suggesting that how uncertainty is propagated through the network matters as much as how it is measured. |
| 2026-02-24 | [Automated Detection and Mitigation of Dependability Failures in Healthcare Scenarios through Digital Twins](http://arxiv.org/abs/2602.21037v1) | Bruno Guindani, Matteo Camilli et al. | Medical Cyber-Physical Systems (CPSs) integrating Patients, Devices, and healthcare personnel (Physicians) form safety-critical PDP triads whose dependability is challenged by system heterogeneity and uncertainty in human and physiological behavior. While existing clinical decision support systems support clinical practice, there remains a need for proactive, reliability-oriented methodologies capable of identifying and mitigating failure scenarios before patient safety is compromised. This paper presents M-GENGAR, a methodology based on a closed-loop Digital Twin (DT) paradigm for dependability assurance of medical CPSs. The approach combines Stochastic Hybrid Automata modeling, data-driven learning of patient dynamics, and Statistical Model Checking with an offline critical scenario detection phase that integrates model-space exploration and diversity analysis to systematically identify and classify scenarios violating expert-defined dependability requirements. M-GENGAR also supports the automated synthesis of mitigation strategies, enabling runtime feedback and control within the DT loop. We evaluate M-GENGAR on a representative use case study involving a pulmonary ventilator. Results show that, in 87.5% of the evaluated scenarios, strategies synthesized through formal game-theoretic analysis stabilize patient vital metrics at least as effectively as human decision-making, while maintaining relevant metrics 20% closer to nominal healthy values on average. |
| 2026-02-24 | [International AI Safety Report 2026](http://arxiv.org/abs/2602.21012v1) | Yoshua Bengio, Stephen Clare et al. | The International AI Safety Report 2026 synthesises the current scientific evidence on the capabilities, emerging risks, and safety of general-purpose AI systems. The report series was mandated by the nations attending the AI Safety Summit in Bletchley, UK. 29 nations, the UN, the OECD, and the EU each nominated a representative to the report's Expert Advisory Panel. Over 100 AI experts contributed, representing diverse perspectives and disciplines. Led by the Report's Chair, these independent experts collectively had full discretion over the report's content. |
| 2026-02-24 | [VII: Visual Instruction Injection for Jailbreaking Image-to-Video Generation Models](http://arxiv.org/abs/2602.20999v1) | Bowen Zheng, Yongli Xiang et al. | Image-to-Video (I2V) generation models, which condition video generation on reference images, have shown emerging visual instruction-following capability, allowing certain visual cues in reference images to act as implicit control signals for video generation. However, this capability also introduces a previously overlooked risk: adversaries may exploit visual instructions to inject malicious intent through the image modality. In this work, we uncover this risk by proposing Visual Instruction Injection (VII), a training-free and transferable jailbreaking framework that intentionally disguises the malicious intent of unsafe text prompts as benign visual instructions in the safe reference image. Specifically, VII coordinates a Malicious Intent Reprogramming module to distill malicious intent from unsafe text prompts while minimizing their static harmfulness, and a Visual Instruction Grounding module to ground the distilled intent onto a safe input image by rendering visual instructions that preserve semantic consistency with the original unsafe text prompt, thereby inducing harmful content during I2V generation. Empirically, our extensive experiments on four state-of-the-art commercial I2V models (Kling-v2.5-turbo, Gemini Veo-3.1, Seedance-1.5-pro, and PixVerse-V5) demonstrate that VII achieves Attack Success Rates of up to 83.5% while reducing Refusal Rates to near zero, significantly outperforming existing baselines. |
| 2026-02-24 | [Evaluating Proactive Risk Awareness of Large Language Models](http://arxiv.org/abs/2602.20976v1) | Xuan Luo, Yubin Chen et al. | As large language models (LLMs) are increasingly embedded in everyday decision-making, their safety responsibilities extend beyond reacting to explicit harmful intent toward anticipating unintended but consequential risks. In this work, we introduce a proactive risk awareness evaluation framework that measures whether LLMs can anticipate potential harms and provide warnings before damage occurs. We construct the Butterfly dataset to instantiate this framework in the environmental and ecological domain. It contains 1,094 queries that simulate ordinary solution-seeking activities whose responses may induce latent ecological impact. Through experiments across five widely used LLMs, we analyze the effects of response length, languages, and modality. Experimental results reveal consistent, significant declines in proactive awareness under length-restricted responses, cross-lingual similarities, and persistent blind spots in (multimodal) species protection. These findings highlight a critical gap between current safety alignment and the requirements of real-world ecological responsibility, underscoring the need for proactive safeguards in LLM deployment. |
| 2026-02-24 | [EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations](http://arxiv.org/abs/2602.20958v1) | Luka Šiktar, Branimir Ćaran et al. | Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios. |
| 2026-02-24 | [ParkDiffusion++: Ego Intention Conditioned Joint Multi-Agent Trajectory Prediction for Automated Parking using Diffusion Models](http://arxiv.org/abs/2602.20923v1) | Jiarong Wei, Anna Rehr et al. | Automated parking is a challenging operational domain for advanced driver assistance systems, requiring robust scene understanding and interaction reasoning. The key challenge is twofold: (i) predict multiple plausible ego intentions according to context and (ii) for each intention, predict the joint responses of surrounding agents, enabling effective what-if decision-making. However, existing methods often fall short, typically treating these interdependent problems in isolation. We propose ParkDiffusion++, which jointly learns a multi-modal ego intention predictor and an ego-conditioned multi-agent joint trajectory predictor for automated parking. Our approach makes several key contributions. First, we introduce an ego intention tokenizer that predicts a small set of discrete endpoint intentions from agent histories and vectorized map polylines. Second, we perform ego-intention-conditioned joint prediction, yielding socially consistent predictions of the surrounding agents for each possible ego intention. Third, we employ a lightweight safety-guided denoiser with different constraints to refine joint scenes during training, thus improving accuracy and safety. Fourth, we propose counterfactual knowledge distillation, where an EMA teacher refined by a frozen safety-guided denoiser provides pseudo-targets that capture how agents react to alternative ego intentions. Extensive evaluations demonstrate that ParkDiffusion++ achieves state-of-the-art performance on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Importantly, qualitative what-if visualizations show that other agents react appropriately to different ego intentions. |
| 2026-02-24 | [Transcoder Adapters for Reasoning-Model Diffing](http://arxiv.org/abs/2602.20904v1) | Nathan Hu, Jake Ward et al. | While reasoning models are increasingly ubiquitous, the effects of reasoning training on a model's internal mechanisms remain poorly understood. In this work, we introduce transcoder adapters, a technique for learning an interpretable approximation of the difference in MLP computation before and after fine-tuning. We apply transcoder adapters to characterize the differences between Qwen2.5-Math-7B and its reasoning-distilled variant, DeepSeek-R1-Distill-Qwen-7B. Learned adapters are faithful to the target model's internal computation and next-token predictions. When evaluated on reasoning benchmarks, adapters match the reasoning model's response lengths and typically recover 50-90% of the accuracy gains from reasoning fine-tuning. Adapter features are sparsely activating and interpretable. When examining adapter features, we find that only ~8% have activating examples directly related to reasoning behaviors. We deeply study one such behavior -- the production of hesitation tokens (e.g., "wait"). Using attribution graphs, we trace hesitation to only ~2.4% of adapter features (5.6k total) performing one of two functions. These features are necessary and sufficient for producing hesitation tokens; removing them reduces response length, often without affecting accuracy. Overall, our results provide insight into reasoning training and suggest transcoder adapters may be useful for studying fine-tuning more broadly. |
| 2026-02-24 | [When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](http://arxiv.org/abs/2602.20880v1) | Yongli Xiang, Ziming Hong et al. | Text-to-Image (T2I) diffusion models have demonstrated significant advancements in generating high-quality images, while raising potential safety concerns regarding harmful content generation. Safety-guidance-based methods have been proposed to mitigate harmful outputs by steering generation away from harmful zones, where the zones are averaged across multiple harmful categories based on predefined keywords. However, these approaches fail to capture the complex interplay among different harm categories, leading to "harmful conflicts" where mitigating one type of harm may inadvertently amplify another, thus increasing overall harmful rate. To address this issue, we propose Conflict-aware Adaptive Safety Guidance (CASG), a training-free framework that dynamically identifies and applies the category-aligned safety direction during generation. CASG is composed of two components: (i) Conflict-aware Category Identification (CaCI), which identifies the harmful category most aligned with the model's evolving generative state, and (ii) Conflict-resolving Guidance Application (CrGA), which applies safety steering solely along the identified category to avoid multi-category interference. CASG can be applied to both latent-space and text-space safeguards. Experiments on T2I safety benchmarks demonstrate CASG's state-of-the-art performance, reducing the harmful rate by up to 15.4% compared to existing methods. |
| 2026-02-24 | [DA-Cal: Towards Cross-Domain Calibration in Semantic Segmentation](http://arxiv.org/abs/2602.20860v1) | Wangkai Li, Rui Sun et al. | While existing unsupervised domain adaptation (UDA) methods greatly enhance target domain performance in semantic segmentation, they often neglect network calibration quality, resulting in misalignment between prediction confidence and actual accuracy -- a significant risk in safety-critical applications. Our key insight emerges from observing that performance degrades substantially when soft pseudo-labels replace hard pseudo-labels in cross-domain scenarios due to poor calibration, despite the theoretical equivalence of perfectly calibrated soft pseudo-labels to hard pseudo-labels. Based on this finding, we propose DA-Cal, a dedicated cross-domain calibration framework that transforms target domain calibration into soft pseudo-label optimization. DA-Cal introduces a Meta Temperature Network to generate pixel-level calibration parameters and employs bi-level optimization to establish the relationship between soft pseudo-labels and UDA supervision, while utilizing complementary domain-mixing strategies to prevent overfitting and reduce domain discrepancies. Experiments demonstrate that DA-Cal seamlessly integrates with existing self-training frameworks across multiple UDA segmentation benchmarks, significantly improving target domain calibration while delivering performance gains without inference overhead. The code will be released. |
| 2026-02-24 | [Exploiting Dependency and Parallelism: Real-Time Scheduling and Analysis for GPU Tasks](http://arxiv.org/abs/2602.20826v1) | Yuanhai Zhang, Songyang He et al. | With the rapid advancement of Artificial Intelligence, the Graphics Processing Unit (GPU) has become increasingly essential across a growing number of safety-critical application domains. Applying a GPU is indispensable for parallel computing; however, the complex data dependencies and resource contention across kernels within a GPU task may unpredictably delay its execution time. To address these problems, this paper presents a scheduling and analysis method for Directed Acyclic Graph (DAG)-structured GPU tasks. Given a DAG representation, the proposed scheduling scales the kernel-level parallelism and establishes inter-kernel dependencies to provide a reduced and predictable DAG response time. The corresponding timing analysis yields a safe yet nonpessimistic makespan bound without any assumption on kernel priorities. The proposed method is implemented using the standard CUDA API, requiring no additional software or hardware support. Experimental results under synthetic and real-world benchmarks demonstrate that the proposed approach effectively reduces the worst-case makespan and measured task execution time compared to the existing methods up to 32.8% and 21.3%, respectively. |
| 2026-02-24 | [Pressure Reveals Character: Behavioural Alignment Evaluation at Depth](http://arxiv.org/abs/2602.20813v1) | Nora Petrova, John Burden | Evaluating alignment in language models requires testing how they behave under realistic pressure, not just what they claim they would do. While alignment failures increasingly cause real-world harm, comprehensive evaluation frameworks with realistic multi-turn scenarios remain lacking. We introduce an alignment benchmark spanning 904 scenarios across six categories -- Honesty, Safety, Non-Manipulation, Robustness, Corrigibility, and Scheming -- validated as realistic by human raters. Our scenarios place models under conflicting instructions, simulated tool access, and multi-turn escalation to reveal behavioural tendencies that single-turn evaluations miss. Evaluating 24 frontier models using LLM judges validated against human annotations, we find that even top-performing models exhibit gaps in specific categories, while the majority of models show consistent weaknesses across the board. Factor analysis reveals that alignment behaves as a unified construct (analogous to the g-factor in cognitive research) with models scoring high on one category tending to score high on others. We publicly release the benchmark and an interactive leaderboard to support ongoing evaluation, with plans to expand scenarios in areas where we observe persistent weaknesses and to add new models as they are released. |
| 2026-02-24 | [POMDPPlanners: Open-Source Package for POMDP Planning](http://arxiv.org/abs/2602.20810v1) | Yaacov Pariente, Vadim Indelman | We present POMDPPlanners, an open-source Python package for empirical evaluation of Partially Observable Markov Decision Process (POMDP) planning algorithms. The package integrates state-of-the-art planning algorithms, a suite of benchmark environments with safety-critical variants, automated hyperparameter optimization via Optuna, persistent caching with failure recovery, and configurable parallel simulation -- reducing the overhead of extensive simulation studies. POMDPPlanners is designed to enable scalable, reproducible research on decision-making under uncertainty, with particular emphasis on risk-sensitive settings where standard toolkits fall short. |
| 2026-02-24 | [Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty](http://arxiv.org/abs/2602.20729v1) | Xu Wan, Chao Yang et al. | Safe Reinforcement Learning (RL) is crucial for achieving high performance while ensuring safety in real-world applications. However, the complex interplay of multiple uncertainty sources in real environments poses significant challenges for interpretable risk assessment and robust decision-making. To address these challenges, we propose Fuz-RL, a fuzzy measure-guided robust framework for safe RL. Specifically, our framework develops a novel fuzzy Bellman operator for estimating robust value functions using Choquet integrals. Theoretically, we prove that solving the Fuz-RL problem (in Constrained Markov Decision Process (CMDP) form) is equivalent to solving distributionally robust safe RL problems (in robust CMDP form), effectively avoiding min-max optimization. Empirical analyses on safe-control-gym and safety-gymnasium scenarios demonstrate that Fuz-RL effectively integrates with existing safe RL baselines in a model-free manner, significantly improving both safety and control performance under various types of uncertainties in observation, action, and dynamics. |
| 2026-02-24 | [ID-LoRA: Efficient Low-Rank Adaptation Inspired by Matrix Interpolative Decomposition](http://arxiv.org/abs/2602.20727v1) | Xindian Ma, Rundong Kong et al. | LoRA has become a universal Parameter-Efficient Fine-Tuning (PEFT) technique that equips Large Language Models (LLMs) to adapt quickly to new tasks. However, when these models are scaled up, even the latest LoRA variants still introduce considerable overhead in trainable parameters. Conversely, aggressively lowering the rank to curb this overhead markedly degrades performance in complex multi-task settings. We propose ID-LoRA, a novel PEFT framework that breaks the trade-off. Its core innovation lies in extracting and reusing clustered parameter groups from the pretrained weight matrix. These groups are then used to form multiple low-rank components, all of which share only a single initialized trainable low-rank matrix. This approach cuts the number of trainable parameters while keeping the model's capacity intact. We evaluate ID-LoRA on five diverse benchmarks: Mathematical Reasoning, Code Generation, MMLU, CommonsenseQA, and Safety Alignment. ID-LoRA outperforms both full fine-tuning and existing PEFT baselines (e.g., LoRA, DoRA, HydraLoRA) while using up to 46% fewer trainable parameters than the standard LoRA. In multi-task scenarios, it surpasses LoRA and its recent variants (e.g., DoRA and HydraLoRA) on both Code and MMLU tasks, yet requires only 54% of the trainable parameters demanded by the conventional LoRA. |
| 2026-02-24 | [Autonomous Laboratory Agent via Customized Domain-Specific Language Model and Modular AI Interface](http://arxiv.org/abs/2602.20669v1) | Zhuo Diao, Kouma Matsumoto et al. | We introduce a system architecture that addresses a fundamental challenge in deploying language-model agents for autonomous control of scientific instrumentation: ensuring reliability in safety-critical environments. The framework integrates probabilistic reasoning by domain-specialized language models with deterministic execution layers that enforce constraints through structured validation and modular orchestration. By separating intent interpretation, experimental planning, and command verification, the architecture translates high-level scientific goals into verifiable experimental actions. We demonstrate this approach in real-time atomic-resolution scanning probe microscopy experiments operated at room temperature, where the system autonomously generates control strategies, invokes corrective modules, and maintains stable operation under experimentally challenging conditions. Quantitative evaluations show that domain-adapted small language models achieve high routing robustness and command accuracy while operating on consumer-grade hardware. Beyond a specific instrument, the framework establishes a general computational principle for deploying language-model agents in safety-critical experimental workflows, providing a pathway toward scalable autonomous laboratories. |
| 2026-02-24 | [Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments](http://arxiv.org/abs/2602.20645v1) | Keisuke Takeshita, Takahiro Yamazaki et al. | The objective of this study is to enable fast and safe manipulation tasks in home environments. Specifically, we aim to develop a system that can recognize its surroundings and identify target objects while in motion, enabling it to plan and execute actions accordingly. We propose a periodic sampling-based whole-body trajectory planning method, called the "Robot Local Planner (RLP)." This method leverages unique features of home environments to enhance computational efficiency, motion optimality, and robustness against recognition and control errors, all while ensuring safety. The RLP minimizes computation time by planning with minimal waypoints and generating safe trajectories. Furthermore, overall motion optimality is improved by periodically executing trajectory planning to select more optimal motions. This approach incorporates inverse kinematics that are robust to base position errors, further enhancing robustness. Evaluation experiments demonstrated that the RLP outperformed existing methods in terms of motion planning time, motion duration, and robustness, confirming its effectiveness in home environments. Moreover, application experiments using a tidy-up task achieved high success rates and short operation times, thereby underscoring its practical feasibility. |
| 2026-02-24 | [An LLM-driven Scenario Generation Pipeline Using an Extended Scenic DSL for Autonomous Driving Safety Validation](http://arxiv.org/abs/2602.20644v1) | Fida Khandaker Safa, Yupeng Jiang et al. | Real-world crash reports, which combine textual summaries and sketches, are valuable for scenario-based testing of autonomous driving systems (ADS). However, current methods cannot effectively translate this multimodal data into precise, executable simulation scenarios, hindering the scalability of ADS safety validation. In this work, we propose a scalable and verifiable pipeline that uses a large language model (GPT-4o mini) and a probabilistic intermediate representation (an Extended Scenic domain-specific language) to automatically extract semantic scenario configurations from crash reports and generate corresponding simulation-ready scenarios. Unlike earlier approaches such as ScenicNL and LCTGen (which generate scenarios directly from text) or TARGET (which uses deterministic mappings from traffic rules), our method introduces an intermediate Scenic DSL layer to separate high-level semantic understanding from low-level scenario rendering, reducing errors and capturing real-world variability. We evaluated the pipeline on cases from the NHTSA CIREN database. The results show high accuracy in knowledge extraction: 100% correctness for environmental and road network attributes, and 97% and 98% for oracle and actor trajectories, respectively, compared to human-derived ground truth. We executed the generated scenarios in the CARLA simulator using the Autoware driving stack, and they consistently triggered the intended traffic-rule violations (such as opposite-lane crossing and red-light running) across 2,000 scenario variations. These findings demonstrate that the proposed pipeline provides a legally grounded, scalable, and verifiable approach to ADS safety validation. |
| 2026-02-24 | [When can we trust untrusted monitoring? A safety case sketch across collusion strategies](http://arxiv.org/abs/2602.20628v1) | Nelson Gardner-Challis, Jonathan Bostock et al. | AIs are increasingly being deployed with greater autonomy and capabilities, which increases the risk that a misaligned AI may be able to cause catastrophic harm. Untrusted monitoring -- using one untrusted model to oversee another -- is one approach to reducing risk. Justifying the safety of an untrusted monitoring deployment is challenging because developers cannot safely deploy a misaligned model to test their protocol directly. In this paper, we develop upon existing methods for rigorously demonstrating safety based on pre-deployment testing. We relax assumptions that previous AI control research made about the collusion strategies a misaligned AI might use to subvert untrusted monitoring. We develop a taxonomy covering passive self-recognition, causal collusion (hiding pre-shared signals), acausal collusion (hiding signals via Schelling points), and combined strategies. We create a safety case sketch to clearly present our argument, explicitly state our assumptions, and highlight unsolved challenges. We identify conditions under which passive self-recognition could be a more effective collusion strategy than those studied previously. Our work builds towards more robust evaluations of untrusted monitoring. |
| 2026-02-24 | [Towards Secure and Efficient DNN Accelerators via Hardware-Software Co-Design](http://arxiv.org/abs/2602.20521v1) | Wei Xuan, Zihao Xuan et al. | The rapid deployment of deep neural network (DNN) accelerators in safety-critical domains such as autonomous vehicles, healthcare systems, and financial infrastructure necessitates robust mechanisms to safeguard data confidentiality and computational integrity. Existing security solutions for DNN accelerators, however, suffer from excessive hardware resource demands and frequent off-chip memory access overheads, which degrade performance and scalability.   To address these challenges, this paper presents a secure and efficient memory protection framework for DNN accelerators with minimal overhead. First, we propose a bandwidth-aware cryptographic scheme that adapts encryption granularity based on memory traffic patterns, striking a balance between security and resource efficiency. Second, we observe that both the overlapping regions in the intra-layer tiling's sliding window pattern and those resulting from inter-layer tiling strategy discrepancies introduce substantial redundant memory accesses and repeated computational overhead in cryptography. Third, we introduce a multi-level authentication mechanism that effectively eliminates unnecessary off-chip memory accesses, enhancing performance and energy efficiency. Experimental results show that this work decreases performance overhead by over 12% and achieves 87% energy efficiency improvement for both server and edge neural processing units (NPUs), while ensuring robust scalability. |

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



