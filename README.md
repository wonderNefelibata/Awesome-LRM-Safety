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
| 2025-10-03 | [TITE-Safety: Time-to-event Safety Monitoring for Clinical Trials](http://arxiv.org/abs/2510.03175v1) | Michael J. Martens, Qinghua Lian et al. | Safety evaluation is an essential component of clinical trials. To protect study participants, these studies often implement safety stopping rules that will halt the trial if an excessive number of toxicity events occur. Existing safety monitoring methods often treat these events as binary outcomes. A strategy that instead handles these as time-to-event endpoints can offer higher power and a reduced time to signal of excess risk, but must manage additional complexities including censoring and competing risks. We propose the TITE-Safety approach for safety monitoring, which incorporates time-to-event information while handling censored observations and competing risks appropriately. This strategy is applied to develop stopping rules using score tests, Bayesian beta-extended binomial models, and sequential probability ratio tests. The operating characteristics of these methods are studied via simulation for common phase 2 and 3 trial scenarios. Across simulation settings, the proposed techniques offer reductions in expected toxicities of 20% or more compared to binary data methods and maintain the type I error rate near the nominal level across various event time distributions. These methods are demonstrated through a redesign of the safety monitoring scheme for BMT CTN 0601, a single arm, phase 2 trial that evaluated bone marrow transplant as treatment for severe sickle cell disease. Our R package "stoppingrule" offers functions to construct and evaluate these stopping rules, providing valuable tools for trial design to investigators. |
| 2025-10-03 | [Learning Stability Certificate for Robotics in Real-World Environments](http://arxiv.org/abs/2510.03123v1) | Zhe Shen | Stability certificates play a critical role in ensuring the safety and reliability of robotic systems. However, deriving these certificates for complex, unknown systems has traditionally required explicit knowledge of system dynamics, often making it a daunting task. This work introduces a novel framework that learns a Lyapunov function directly from trajectory data, enabling the certification of stability for autonomous systems without needing detailed system models. By parameterizing the Lyapunov candidate using a neural network and ensuring positive definiteness through Cholesky factorization, our approach automatically identifies whether the system is stable under the given trajectory. To address the challenges posed by noisy, real-world data, we allow for controlled violations of the stability condition, focusing on maintaining high confidence in the stability certification process. Our results demonstrate that this framework can provide data-driven stability guarantees, offering a robust method for certifying the safety of robotic systems in dynamic, real-world environments. This approach works without access to the internal control algorithms, making it applicable even in situations where system behavior is opaque or proprietary. The tool for learning the stability proof is open-sourced by this research: https://github.com/HansOersted/stability. |
| 2025-10-03 | [Whisker-based Tactile Flight for Tiny Drones](http://arxiv.org/abs/2510.03119v1) | Chaoxiang Ye, Guido de Croon et al. | Tiny flying robots hold great potential for search-and-rescue, safety inspections, and environmental monitoring, but their small size limits conventional sensing-especially with poor-lighting, smoke, dust or reflective obstacles. Inspired by nature, we propose a lightweight, 3.2-gram, whisker-based tactile sensing apparatus for tiny drones, enabling them to navigate and explore through gentle physical interaction. Just as rats and moles use whiskers to perceive surroundings, our system equips drones with tactile perception in flight, allowing obstacle sensing even in pitch-dark conditions. The apparatus uses barometer-based whisker sensors to detect obstacle locations while minimising destabilisation. To address sensor noise and drift, we develop a tactile depth estimation method achieving sub-6 mm accuracy. This enables drones to navigate, contour obstacles, and explore confined spaces solely through touch-even in total darkness along both soft and rigid surfaces. Running fully onboard a 192-KB RAM microcontroller, the system supports autonomous tactile flight and is validated in both simulation and real-world tests. Our bio-inspired approach redefines vision-free navigation, opening new possibilities for micro aerial vehicles in extreme environments. |
| 2025-10-03 | [InsideOut: An EfficientNetV2-S Based Deep Learning Framework for Robust Multi-Class Facial Emotion Recognition](http://arxiv.org/abs/2510.03066v1) | Ahsan Farabi, Israt Khandaker et al. | Facial Emotion Recognition (FER) is a key task in affective computing, enabling applications in human-computer interaction, e-learning, healthcare, and safety systems. Despite advances in deep learning, FER remains challenging due to occlusions, illumination and pose variations, subtle intra-class differences, and dataset imbalance that hinders recognition of minority emotions. We present InsideOut, a reproducible FER framework built on EfficientNetV2-S with transfer learning, strong data augmentation, and imbalance-aware optimization. The approach standardizes FER2013 images, applies stratified splitting and augmentation, and fine-tunes a lightweight classification head with class-weighted loss to address skewed distributions. InsideOut achieves 62.8% accuracy with a macro averaged F1 of 0.590 on FER2013, showing competitive results compared to conventional CNN baselines. The novelty lies in demonstrating that efficient architectures, combined with tailored imbalance handling, can provide practical, transparent, and reproducible FER solutions. |
| 2025-10-03 | [Economic zone data-enabled predictive control for connected open water systems](http://arxiv.org/abs/2510.03043v1) | Xiaoqiao Chen, Xuewen Zhang et al. | Real-time regulation of water distribution in connected open water systems is critical for ensuring system safety and meeting operational requirements. In this work, we consider a connected open water system that includes linkage hydraulic structures such as weirs, pumps and sluice gates. We propose a mixed-integer economic zone data-enabled predictive control (DeePC) approach, which is used to maintain the water levels of the branches within desired zones to avoid floods and reduce the energy consumption of the pumps in the considered water system. The proposed DeePC-based approach predicts the future dynamics of the system water levels, and generates optimal control actions based on system input and output data, thereby eliminating the need for both first-principles modeling and explicit data-driven modeling. To achieve multiple control objectives in order of priority, we utilize lexicographic optimization and adapt traditional DeePC cost function for zone tracking and energy consumption minimization. Additionally, Bayesian optimization is utilized to determine the control target zone, which effectively balances zone tracking and energy consumption in the presence of external disturbances. Comprehensive simulations and comparative analyses demonstrate the effectiveness of the proposed method. The proposed method maintains water levels within the desired zone for 97.04% of the operating time, with an average energy consumption of 33.5 kWh per 0.5 h. Compared to baseline methods, the proposed approach reduces the zone-tracking mean square error by 98.82% relative to economic zone DeePC without Bayesian optimization, and lowers energy consumption by 44.08% relative to economic set-point tracking DeePC. As compared to passive pump/gate control, the proposed method lowers the frequency of zone violations by 86.94% and the average energy consumption by 4.69%. |
| 2025-10-03 | [Long-Term Human Motion Prediction Using Spatio-Temporal Maps of Dynamics](http://arxiv.org/abs/2510.03031v1) | Yufei Zhu, Andrey Rudenko et al. | Long-term human motion prediction (LHMP) is important for the safe and efficient operation of autonomous robots and vehicles in environments shared with humans. Accurate predictions are important for applications including motion planning, tracking, human-robot interaction, and safety monitoring. In this paper, we exploit Maps of Dynamics (MoDs), which encode spatial or spatio-temporal motion patterns as environment features, to achieve LHMP for horizons of up to 60 seconds. We propose an MoD-informed LHMP framework that supports various types of MoDs and includes a ranking method to output the most likely predicted trajectory, improving practical utility in robotics. Further, a time-conditioned MoD is introduced to capture motion patterns that vary across different times of day. We evaluate MoD-LHMP instantiated with three types of MoDs. Experiments on two real-world datasets show that MoD-informed method outperforms learning-based ones, with up to 50\% improvement in average displacement error, and the time-conditioned variant achieves the highest accuracy overall. Project code is available at https://github.com/test-bai-cpu/LHMP-with-MoDs.git |
| 2025-10-03 | [Physics-Constrained Inc-GAN for Tunnel Propagation Modeling from Sparse Line Measurements](http://arxiv.org/abs/2510.03019v1) | Yang Zhou, Haochang Wu et al. | High-speed railway tunnel communication systems require reliable radio wave propagation prediction to ensure operational safety. However, conventional simulation methods face challenges of high computational complexity and inability to effectively process sparse measurement data collected during actual railway operations. This letter proposes an inception-enhanced generative adversarial network (Inc-GAN) that can reconstruct complete electric field distributions across tunnel cross-sections using sparse value lines measured during actual train operations as input. This directly addresses practical railway measurement constraints. Through an inception-based generator architecture and progressive training strategy, the method achieves robust reconstruction from single measurement signal lines to complete field distributions. Numerical simulation validation demonstrates that Inc-GAN can accurately predict electric fields based on measured data collected during actual train operations, with significantly improved computational efficiency compared to traditional methods, providing a novel solution for railway communication system optimization based on real operational data. |
| 2025-10-03 | [Untargeted Jailbreak Attack](http://arxiv.org/abs/2510.02999v1) | Xinzhe Huang, Wenjing Hu et al. | Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that \textsc{UJA} can achieve over 80\% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%. |
| 2025-10-03 | [Real-Time Nonlinear Model Predictive Control of Heavy-Duty Skid-Steered Mobile Platform for Trajectory Tracking Tasks](http://arxiv.org/abs/2510.02976v1) | Alvaro Paz, Pauli Mustalahti et al. | This paper presents a framework for real-time optimal controlling of a heavy-duty skid-steered mobile platform for trajectory tracking. The importance of accurate real-time performance of the controller lies in safety considerations of situations where the dynamic system under control is affected by uncertainties and disturbances, and the controller should compensate for such phenomena in order to provide stable performance. A multiple-shooting nonlinear model-predictive control framework is proposed in this paper. This framework benefits from suitable algorithm along with readings from various sensors for genuine real-time performance with extremely high accuracy. The controller is then tested for tracking different trajectories where it demonstrates highly desirable performance in terms of both speed and accuracy. This controller shows remarkable improvement when compared to existing nonlinear model-predictive controllers in the literature that were implemented on skid-steered mobile platforms. |
| 2025-10-03 | [External Data Extraction Attacks against Retrieval-Augmented Large Language Models](http://arxiv.org/abs/2510.02964v1) | Yu He, Yifei Chen et al. | In recent years, RAG has emerged as a key paradigm for enhancing large language models (LLMs). By integrating externally retrieved information, RAG alleviates issues like outdated knowledge and, crucially, insufficient domain expertise. While effective, RAG introduces new risks of external data extraction attacks (EDEAs), where sensitive or copyrighted data in its knowledge base may be extracted verbatim. These risks are particularly acute when RAG is used to customize specialized LLM applications with private knowledge bases. Despite initial studies exploring these risks, they often lack a formalized framework, robust attack performance, and comprehensive evaluation, leaving critical questions about real-world EDEA feasibility unanswered.   In this paper, we present the first comprehensive study to formalize EDEAs against retrieval-augmented LLMs. We first formally define EDEAs and propose a unified framework decomposing their design into three components: extraction instruction, jailbreak operator, and retrieval trigger, under which prior attacks can be considered instances within our framework. Guided by this framework, we develop SECRET: a Scalable and EffeCtive exteRnal data Extraction aTtack. Specifically, SECRET incorporates (1) an adaptive optimization process using LLMs as optimizers to generate specialized jailbreak prompts for EDEAs, and (2) cluster-focused triggering, an adaptive strategy that alternates between global exploration and local exploitation to efficiently generate effective retrieval triggers. Extensive evaluations across 4 models reveal that SECRET significantly outperforms previous attacks, and is highly effective against all 16 tested RAG instances. Notably, SECRET successfully extracts 35% of the data from RAG powered by Claude 3.7 Sonnet for the first time, whereas other attacks yield 0% extraction. Our findings call for attention to this emerging threat. |
| 2025-10-03 | [Metrics vs Surveys: Can Quantitative Measures Replace Human Surveys in Social Robot Navigation? A Correlation Analysis](http://arxiv.org/abs/2510.02941v1) | Stefano Trepella, Mauro Martini et al. | Social, also called human-aware, navigation is a key challenge for the integration of mobile robots into human environments. The evaluation of such systems is complex, as factors such as comfort, safety, and legibility must be considered. Human-centered assessments, typically conducted through surveys, provide reliable insights but are costly, resource-intensive, and difficult to reproduce or compare across systems. Alternatively, numerical social navigation metrics are easy to compute and facilitate comparisons, yet the community lacks consensus on a standard set of metrics.   This work explores the relationship between numerical metrics and human-centered evaluations to identify potential correlations. If specific quantitative measures align with human perceptions, they could serve as standardized evaluation tools, reducing the dependency on surveys. Our results indicate that while current metrics capture some aspects of robot navigation behavior, important subjective factors remain insufficiently represented and new metrics are necessary. |
| 2025-10-03 | [Self-Reflective Generation at Test Time](http://arxiv.org/abs/2510.02919v1) | Jian Mu, Qixin Zhang et al. | Large language models (LLMs) increasingly solve complex reasoning tasks via long chain-of-thought, but their forward-only autoregressive generation process is fragile; early token errors can cascade, which creates a clear need for self-reflection mechanisms. However, existing self-reflection either performs revisions over full drafts or learns self-correction via expensive training, both fundamentally reactive and inefficient. To address this, we propose Self-Reflective Generation at Test Time (SRGen), a lightweight test-time framework that reflects before generating at uncertain points. During token generation, SRGen utilizes dynamic entropy thresholding to identify high-uncertainty tokens. For each identified token, it trains a specific corrective vector, which fully exploits the already generated context for a self-reflective generation to correct the token probability distribution. By retrospectively analyzing the partial output, this self-reflection enables more trustworthy decisions, thereby significantly reducing the probability of errors at highly uncertain points. Evaluated on challenging mathematical reasoning benchmarks and a diverse set of LLMs, SRGen can consistently strengthen model reasoning: improvements in single-pass quality also translate into stronger self-consistency voting. Especially, on AIME2024 with DeepSeek-R1-Distill-Qwen-7B, SRGen yields absolute improvements of +12.0% on Pass@1 and +13.3% on Cons@5. Moreover, our findings position SRGen as a plug-and-play method that integrates reflection into the generation process for reliable LLM reasoning, achieving consistent gains with bounded overhead and broad composability with other training-time (e.g., RLHF) and test-time (e.g., SLOT) techniques. |
| 2025-10-03 | [Training-Free Out-Of-Distribution Segmentation With Foundation Models](http://arxiv.org/abs/2510.02909v1) | Laith Nayal, Hadi Salloum et al. | Detecting unknown objects in semantic segmentation is crucial for safety-critical applications such as autonomous driving. Large vision foundation models, includ- ing DINOv2, InternImage, and CLIP, have advanced visual representation learn- ing by providing rich features that generalize well across diverse tasks. While their strength in closed-set semantic tasks is established, their capability to detect out- of-distribution (OoD) regions in semantic segmentation remains underexplored. In this work, we investigate whether foundation models fine-tuned on segmen- tation datasets can inherently distinguish in-distribution (ID) from OoD regions without any outlier supervision. We propose a simple, training-free approach that utilizes features from the InternImage backbone and applies K-Means clustering alongside confidence thresholding on raw decoder logits to identify OoD clusters. Our method achieves 50.02 Average Precision on the RoadAnomaly benchmark and 48.77 on the benchmark of ADE-OoD with InternImage-L, surpassing several supervised and unsupervised baselines. These results suggest a promising direc- tion for generic OoD segmentation methods that require minimal assumptions or additional data. |
| 2025-10-03 | [Point Cloud-Based Control Barrier Functions for Model Predictive Control in Safety-Critical Navigation of Autonomous Mobile Robots](http://arxiv.org/abs/2510.02885v1) | Faduo Liang, Yunfeng Yang et al. | In this work, we propose a novel motion planning algorithm to facilitate safety-critical navigation for autonomous mobile robots. The proposed algorithm integrates a real-time dynamic obstacle tracking and mapping system that categorizes point clouds into dynamic and static components. For dynamic point clouds, the Kalman filter is employed to estimate and predict their motion states. Based on these predictions, we extrapolate the future states of dynamic point clouds, which are subsequently merged with static point clouds to construct the forward-time-domain (FTD) map. By combining control barrier functions (CBFs) with nonlinear model predictive control, the proposed algorithm enables the robot to effectively avoid both static and dynamic obstacles. The CBF constraints are formulated based on risk points identified through collision detection between the predicted future states and the FTD map. Experimental results from both simulated and real-world scenarios demonstrate the efficacy of the proposed algorithm in complex environments. In simulation experiments, the proposed algorithm is compared with two baseline approaches, showing superior performance in terms of safety and robustness in obstacle avoidance. The source code is released for the reference of the robotics community. |
| 2025-10-03 | [ELMF4EggQ: Ensemble Learning with Multimodal Feature Fusion for Non-Destructive Egg Quality Assessment](http://arxiv.org/abs/2510.02876v1) | Md Zahim Hassan, Md. Osama et al. | Accurate, non-destructive assessment of egg quality is critical for ensuring food safety, maintaining product standards, and operational efficiency in commercial poultry production. This paper introduces ELMF4EggQ, an ensemble learning framework that employs multimodal feature fusion to classify egg grade and freshness using only external attributes - image, shape, and weight. A novel, publicly available dataset of 186 brown-shelled eggs was constructed, with egg grade and freshness levels determined through laboratory-based expert assessments involving internal quality measurements, such as yolk index and Haugh unit. To the best of our knowledge, this is the first study to apply machine learning methods for internal egg quality assessment using only external, non-invasive features, and the first to release a corresponding labeled dataset. The proposed framework integrates deep features extracted from external egg images with structural characteristics such as egg shape and weight, enabling a comprehensive representation of each egg. Image feature extraction is performed using top-performing pre-trained CNN models (ResNet152, DenseNet169, and ResNet152V2), followed by PCA-based dimensionality reduction, SMOTE augmentation, and classification using multiple machine learning algorithms. An ensemble voting mechanism combines predictions from the best-performing classifiers to enhance overall accuracy. Experimental results demonstrate that the multimodal approach significantly outperforms image-only and tabular (shape and weight) only baselines, with the multimodal ensemble approach achieving 86.57% accuracy in grade classification and 70.83% in freshness prediction. All code and data are publicly available at https://github.com/Kenshin-Keeps/Egg_Quality_Prediction_ELMF4EggQ, promoting transparency, reproducibility, and further research in this domain. |
| 2025-10-03 | [Knowledge-Aware Modeling with Frequency Adaptive Learning for Battery Health Prognostics](http://arxiv.org/abs/2510.02839v1) | Vijay Babu Pamshetti, Wei Zhang et al. | Battery health prognostics are critical for ensuring safety, efficiency, and sustainability in modern energy systems. However, it has been challenging to achieve accurate and robust prognostics due to complex battery degradation behaviors with nonlinearity, noise, capacity regeneration, etc. Existing data-driven models capture temporal degradation features but often lack knowledge guidance, which leads to unreliable long-term health prognostics. To overcome these limitations, we propose Karma, a knowledge-aware model with frequency-adaptive learning for battery capacity estimation and remaining useful life prediction. The model first performs signal decomposition to derive battery signals in different frequency bands. A dual-stream deep learning architecture is developed, where one stream captures long-term low-frequency degradation trends and the other models high-frequency short-term dynamics. Karma regulates the prognostics with knowledge, where battery degradation is modeled as a double exponential function based on empirical studies. Our dual-stream model is used to optimize the parameters of the knowledge with particle filters to ensure physically consistent and reliable prognostics and uncertainty quantification. Experimental study demonstrates Karma's superior performance, achieving average error reductions of 50.6% and 32.6% over state-of-the-art algorithms for battery health prediction on two mainstream datasets, respectively. These results highlight Karma's robustness, generalizability, and potential for safer and more reliable battery management across diverse applications. |
| 2025-10-03 | [Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](http://arxiv.org/abs/2510.02833v1) | Zhixin Xie, Xurui Song et al. | Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign. |
| 2025-10-03 | [A Granular Study of Safety Pretraining under Model Abliteration](http://arxiv.org/abs/2510.02768v1) | Shashank Agnihotri, Jonas Jakubassa et al. | Open-weight LLMs can be modified at inference time with simple activation edits, which raises a practical question for safety: do common safety interventions like refusal training or metatag training survive such edits? We study model abliteration, a lightweight projection technique designed to remove refusal-sensitive directions, and conduct a controlled evaluation across a granular sequence of Safety Pretraining checkpoints for SmolLM2-1.7B, alongside widely used open baselines. For each of 20 systems, original and abliterated, we issue 100 prompts with balanced harmful and harmless cases, classify responses as **Refusal** or **Non-Refusal** using multiple judges, and validate judge fidelity on a small human-labeled subset. We also probe whether models can identify refusal in their own outputs. Our study produces a checkpoint-level characterization of which data-centric safety components remain robust under abliteration, quantifies how judge selection influences evaluation outcomes, and outlines a practical protocol for integrating inference-time edits into safety assessments. Code: https://github.com/shashankskagnihotri/safety_pretraining. |
| 2025-10-03 | [Retrv-R1: A Reasoning-Driven MLLM Framework for Universal and Efficient Multimodal Retrieval](http://arxiv.org/abs/2510.02745v1) | Lanyun Zhu, Deyi Ji et al. | The success of DeepSeek-R1 demonstrates the immense potential of using reinforcement learning (RL) to enhance LLMs' reasoning capabilities. This paper introduces Retrv-R1, the first R1-style MLLM specifically designed for multimodal universal retrieval, achieving higher performance by employing step-by-step reasoning to produce more accurate retrieval results. We find that directly applying the methods of DeepSeek-R1 to retrieval tasks is not feasible, mainly due to (1) the high computational cost caused by the large token consumption required for multiple candidates with reasoning processes, and (2) the instability and suboptimal results when directly applying RL to train for retrieval tasks. To address these issues, Retrv-R1 introduces an information compression module with a details inspection mechanism, which enhances computational efficiency by reducing the number of tokens while ensuring that critical information for challenging candidates is preserved. Furthermore, a new training paradigm is proposed, including an activation stage using a retrieval-tailored synthetic CoT dataset for more effective optimization, followed by RL with a novel curriculum reward to improve both performance and efficiency. Incorporating these novel designs, Retrv-R1 achieves SOTA performance, high efficiency, and strong generalization ability, as demonstrated by experiments across multiple benchmarks and tasks. |
| 2025-10-03 | [RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization](http://arxiv.org/abs/2510.02695v1) | Kai Fukazawa, Kunal Mundada et al. | In safety-critical domains where online data collection is infeasible, offline reinforcement learning (RL) offers an attractive alternative but only if policies deliver high returns without incurring catastrophic lower-tail risk. Prior work on risk-averse offline RL achieves safety at the cost of value conservatism and restricted policy classes, whereas expressive policies are only used in risk-neutral settings. Here, we address this gap by introducing the \textbf{Risk-Aware Multimodal Actor-Critic (RAMAC)} framework, which couples an \emph{expressive generative actor} with a distributional critic. The RAMAC differentiates composite objective combining distributional risk and BC loss through the generative path, achieving risk-sensitive learning in complex multimodal scenarios. We instantiate RAMAC with diffusion and flow-matching actors and observe consistent gains in $\mathrm{CVaR}_{0.1}$ while maintaining strong returns on most Stochastic-D4RL tasks. Code: https://github.com/KaiFukazawa/RAMAC.git |

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



