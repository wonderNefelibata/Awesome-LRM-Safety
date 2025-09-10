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
| 2025-09-09 | [Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search](http://arxiv.org/abs/2509.07969v1) | Xin Lai, Junyi Li et al. | Recent advances in large multimodal models have leveraged image-based tools with reinforcement learning to tackle visual problems. However, existing open-source approaches often exhibit monotonous reasoning patterns and allow only a limited number of interaction turns, making them inadequate for difficult tasks that require trial-and-error exploration. In this work, we address this limitation by scaling up tool-based interactions and introduce Mini-o3, a system that executes deep, multi-turn reasoning -- spanning tens of steps -- and achieves state-of-the-art performance on challenging visual search tasks. Our recipe for reproducing OpenAI o3-style behaviors comprises three key components. First, we construct the Visual Probe Dataset, a collection of thousands of challenging visual search problems designed for exploratory reasoning. Second, we develop an iterative data collection pipeline to obtain cold-start trajectories that exhibit diverse reasoning patterns, including depth-first search, trial-and-error, and goal maintenance. Third, we propose an over-turn masking strategy that prevents penalization of over-turn responses (those that hit the maximum number of turns) during reinforcement learning, thereby balancing training-time efficiency with test-time scalability. Despite training with an upper bound of only six interaction turns, our model generates trajectories that naturally scale to tens of turns at inference time, with accuracy improving as the number of turns increases. Extensive experiments demonstrate that Mini-o3 produces rich reasoning patterns and deep thinking paths, effectively solving challenging visual search problems. |
| 2025-09-09 | [ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation](http://arxiv.org/abs/2509.07941v1) | Kai Ye, Liangcai Su et al. | Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.   In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is https://importsnare.github.io. |
| 2025-09-09 | [Predicting person-level injury severity using crash narratives: A balanced approach with roadway classification and natural language process techniques](http://arxiv.org/abs/2509.07845v1) | Mohammad Zana Majidi, Sajjad Karimi et al. | Predicting injuries and fatalities in traffic crashes plays a critical role in enhancing road safety, improving emergency response, and guiding public health interventions. This study investigates the added value of unstructured crash narratives (written by police officers at the scene) when combined with structured crash data to predict injury severity. Two widely used Natural Language Processing (NLP) techniques, Term Frequency-Inverse Document Frequency (TF-IDF) and Word2Vec, were employed to extract semantic meaning from the narratives, and their effectiveness was compared. To address the challenge of class imbalance, a K-Nearest Neighbors-based oversampling method was applied to the training data prior to modeling. The dataset consists of crash records from Kentucky spanning 2019 to 2023. To account for roadway heterogeneity, three road classification schemes were used: (1) eight detailed functional classes (e.g., Urban Two-Lane, Rural Interstate, Urban Multilane Divided), (2) four broader paired categories (e.g., Urban vs. Rural, Freeway vs. Non-Freeway), and (3) a unified dataset without classification. A total of 102 machine learning models were developed by combining structured features and narrative-based features using the two NLP techniques alongside three ensemble algorithms: XGBoost, Random Forest, and AdaBoost. Results demonstrate that models incorporating narrative data consistently outperform those relying solely on structured data. Among all combinations, TF-IDF coupled with XGBoost yielded the most accurate predictions in most subgroups. The findings highlight the power of integrating textual and structured crash information to enhance person-level injury prediction. This work offers a practical and adaptable framework for transportation safety professionals to improve crash severity modeling, guide policy decisions, and design more effective countermeasures. |
| 2025-09-09 | [Nuclear Data Adjustment for Nonlinear Applications in the OECD/NEA WPNCS SG14 Benchmark -- A Bayesian Inverse UQ-based Approach for Data Assimilation](http://arxiv.org/abs/2509.07790v1) | Christopher Brady, Xu Wu | The Organization for Economic Cooperation and Development (OECD) Working Party on Nuclear Criticality Safety (WPNCS) proposed a benchmark exercise to assess the performance of current nuclear data adjustment techniques applied to nonlinear applications and experiments with low correlation to applications. This work introduces Bayesian Inverse Uncertainty Quantification (IUQ) as a method for nuclear data adjustments in this benchmark, and compares IUQ to the more traditional methods of Generalized Linear Least Squares (GLLS) and Monte Carlo Bayes (MOCABA). Posterior predictions from IUQ showed agreement with GLLS and MOCABA for linear applications. When comparing GLLS, MOCABA, and IUQ posterior predictions to computed model responses using adjusted parameters, we observe that GLLS predictions fail to replicate computed response distributions for nonlinear applications, while MOCABA shows near agreement, and IUQ uses computed model responses directly. We also discuss observations on why experiments with low correlation to applications can be informative to nuclear data adjustments and identify some properties useful in selecting experiments for inclusion in nuclear data adjustment. Performance in this benchmark indicates potential for Bayesian IUQ in nuclear data adjustments. |
| 2025-09-09 | [Factuality Beyond Coherence: Evaluating LLM Watermarking Methods for Medical Texts](http://arxiv.org/abs/2509.07755v1) | Rochana Prih Hastuti, Rian Adam Rajagede et al. | As large language models (LLMs) adapted to sensitive domains such as medicine, their fluency raises safety risks, particularly regarding provenance and accountability. Watermarking embeds detectable patterns to mitigate these risks, yet its reliability in medical contexts remains untested. Existing benchmarks focus on detection-quality tradeoffs, overlooking factual risks under low-entropy settings often exploited by watermarking's reweighting strategy. We propose a medical-focused evaluation workflow that jointly assesses factual accuracy and coherence. Using GPT-Judger and further human validation, we introduce the Factuality-Weighted Score (FWS), a composite metric prioritizing factual accuracy beyond coherence to guide watermarking deployment in medical domains. Our evaluation shows current watermarking methods substantially compromise medical factuality, with entropy shifts degrading medical entity representation. These findings underscore the need for domain-aware watermarking approaches that preserve the integrity of medical content. |
| 2025-09-09 | [Fault Tolerant Control of a Quadcopter using Reinforcement Learning](http://arxiv.org/abs/2509.07707v1) | Muzaffar Habib, Adnan Maqsood et al. | This study presents a novel reinforcement learning (RL)-based control framework aimed at enhancing the safety and robustness of the quadcopter, with a specific focus on resilience to in-flight one propeller failure. Addressing the critical need of a robust control strategy for maintaining a desired altitude for the quadcopter to safe the hardware and the payload in physical applications. The proposed framework investigates two RL methodologies Dynamic Programming (DP) and Deep Deterministic Policy Gradient (DDPG), to overcome the challenges posed by the rotor failure mechanism of the quadcopter. DP, a model-based approach, is leveraged for its convergence guarantees, despite high computational demands, whereas DDPG, a model-free technique, facilitates rapid computation but with constraints on solution duration. The research challenge arises from training RL algorithms on large dimensions and action domains. With modifications to the existing DP and DDPG algorithms, the controllers were trained not only to cater for large continuous state and action domain and also achieve a desired state after an inflight propeller failure. To verify the robustness of the proposed control framework, extensive simulations were conducted in a MATLAB environment across various initial conditions and underscoring its viability for mission-critical quadcopter applications. A comparative analysis was performed between both RL algorithms and their potential for applications in faulty aerial systems. |
| 2025-09-09 | [What's in the Box: Ergonomic and Expressive Capture Tracking over Generic Data Structures (Extended Version)](http://arxiv.org/abs/2509.07609v1) | Yichen Xu, Oliver Bračevac et al. | Capturing types in Scala unify static effect and resource tracking with object capabilities, enabling lightweight effect polymorphism with minimal notational overhead. However, their expressiveness has been insufficient for tracking capabilities embedded in generic data structures, preventing them from scaling to the standard collections library -- an essential prerequisite for broader adoption. This limitation stems from the inability to name capabilities within the system's notion of box types.   This paper develops System Capless, a new foundation for capturing types that provides the theoretical basis for reach capabilities (rcaps), a novel mechanism for naming "what's in the box." The calculus refines the universal capability notion into a new scheme with existential and universal capture set quantification. Intuitively, rcaps witness existentially quantified capture sets inside the boxes of generic types in a way that does not require exposing existential capture types in the surface language. We have fully mechanized the formal metatheory of System Capless in Lean, including proofs of type soundness and scope safety. System Capless supports the same lightweight notation of capturing types plus rcaps, as certified by a type-preserving translation, and also enables fully optional explicit capture-set quantification to increase expressiveness.   Finally, we present a full reimplementation of capture checking in Scala 3 based on System Capless and migrate the entire Scala collections library and an asynchronous programming library to evaluate its practicality and ergonomics. Our results demonstrate that reach capabilities enable the adoption of capture checking in production code with minimal changes and minimal-to-zero notational overhead in a vast majority of cases. |
| 2025-09-09 | [Can SSD-Mamba2 Unlock Reinforcement Learning for End-to-End Motion Control?](http://arxiv.org/abs/2509.07593v1) | Gavin Tao, Yinuo Wang et al. | End-to-end reinforcement learning for motion control promises unified perception-action policies that scale across embodiments and tasks, yet most deployed controllers are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for scalable, foresightful, and efficient end-to-end motion control. |
| 2025-09-09 | [Flexible Morphing Aerial Robot with Inflatable Structure for Perching-based Human-Robot Interaction](http://arxiv.org/abs/2509.07496v1) | Ayano Miyamichi, Moju Zhao et al. | Birds in nature perform perching not only for rest but also for interaction with human such as the relationship with falconers. Recently, researchers achieve perching-capable aerial robots as a way to save energy, and deformable structure demonstrate significant advantages in efficiency of perching and compactness of configuration. However, ensuring flight stability remains challenging for deformable aerial robots due to the difficulty of controlling flexible arms. Furthermore, perching for human interaction requires high compliance along with safety. Thus, this study aims to develop a deformable aerial robot capable of perching on humans with high flexibility and grasping ability. To overcome the challenges of stability of both flight and perching, we propose a hybrid morphing structure that combines a unilateral flexible arm and a pneumatic inflatable actuators. This design allows the robot's arms to remain rigid during flight and soft while perching for more effective grasping. We also develop a pneumatic control system that optimizes pressure regulation while integrating shock absorption and adjustable grasping forces, enhancing interaction capabilities and energy efficiency. Besides, we focus on the structural characteristics of the unilateral flexible arm and identify sufficient conditions under which standard quadrotor modeling and control remain effective in terms of flight stability. Finally, the developed prototype demonstrates the feasibility of compliant perching maneuvers on humans, as well as the robust recovery even after arm deformation caused by thrust reductions during flight. To the best of our knowledge, this work is the first to achieve an aerial robot capable of perching on humans for interaction. |
| 2025-09-09 | [HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention](http://arxiv.org/abs/2509.07475v1) | Saumya Goswami, Siddharth Kurra | Detecting content that contradicts or is unsupported by a given source text is a critical challenge for the safe deployment of generative language models. We introduce HALT-RAG, a post-hoc verification system designed to identify hallucinations in the outputs of Retrieval-Augmented Generation (RAG) pipelines. Our flexible and task-adaptable framework uses a universal feature set derived from an ensemble of two frozen, off-the-shelf Natural Language Inference (NLI) models and lightweight lexical signals. These features are used to train a simple, calibrated, and task-adapted meta-classifier. Using a rigorous 5-fold out-of-fold (OOF) training protocol to prevent data leakage and produce unbiased estimates, we evaluate our system on the HaluEval benchmark. By pairing our universal feature set with a lightweight, task-adapted classifier and a precision-constrained decision policy, HALT-RAG achieves strong OOF F1-scores of 0.7756, 0.9786, and 0.7391 on the summarization, QA, and dialogue tasks, respectively. The system's well-calibrated probabilities enable a practical abstention mechanism, providing a reliable tool for balancing model performance with safety requirements. |
| 2025-09-09 | [Safe and Non-Conservative Contingency Planning for Autonomous Vehicles via Online Learning-Based Reachable Set Barriers](http://arxiv.org/abs/2509.07464v1) | Rui Yang, Lei Zheng et al. | Autonomous vehicles must navigate dynamically uncertain environments while balancing the safety and driving efficiency. This challenge is exacerbated by the unpredictable nature of surrounding human-driven vehicles (HVs) and perception inaccuracies, which require planners to adapt to evolving uncertainties while maintaining safe trajectories. Overly conservative planners degrade driving efficiency, while deterministic approaches may encounter serious issues and risks of failure when faced with sudden and unexpected maneuvers. To address these issues, we propose a real-time contingency trajectory optimization framework in this paper. By employing event-triggered online learning of HV control-intent sets, our method dynamically quantifies multi-modal HV uncertainties and refines the forward reachable set (FRS) incrementally. Crucially, we enforce invariant safety through FRS-based barrier constraints that ensure safety without reliance on accurate trajectory prediction of HVs. These constraints are embedded in contingency trajectory optimization and solved efficiently through consensus alternative direction method of multipliers (ADMM). The system continuously adapts to the uncertainties in HV behaviors, preserving feasibility and safety without resorting to excessive conservatism. High-fidelity simulations on highway and urban scenarios, as well as a series of real-world experiments demonstrate significant improvements in driving efficiency and passenger comfort while maintaining safety under uncertainty. The project page is available at https://pathetiue.github.io/frscp.github.io/. |
| 2025-09-09 | [DepthVision: Robust Vision-Language Understanding through GAN-Based LiDAR-to-RGB Synthesis](http://arxiv.org/abs/2509.07463v1) | Sven Kirchner, Nils Purschke et al. | Ensuring reliable robot operation when visual input is degraded or insufficient remains a central challenge in robotics. This letter introduces DepthVision, a framework for multimodal scene understanding designed to address this problem. Unlike existing Vision-Language Models (VLMs), which use only camera-based visual input alongside language, DepthVision synthesizes RGB images from sparse LiDAR point clouds using a conditional generative adversarial network (GAN) with an integrated refiner network. These synthetic views are then combined with real RGB data using a Luminance-Aware Modality Adaptation (LAMA), which blends the two types of data dynamically based on ambient lighting conditions. This approach compensates for sensor degradation, such as darkness or motion blur, without requiring any fine-tuning of downstream vision-language models. We evaluate DepthVision on real and simulated datasets across various models and tasks, with particular attention to safety-critical tasks. The results demonstrate that our approach improves performance in low-light conditions, achieving substantial gains over RGB-only baselines while preserving compatibility with frozen VLMs. This work highlights the potential of LiDAR-guided RGB synthesis for achieving robust robot operation in real-world environments. |
| 2025-09-09 | [Bias-Aware Machine Unlearning: Towards Fairer Vision Models via Controllable Forgetting](http://arxiv.org/abs/2509.07456v1) | Sai Siddhartha Chary Aylapuram, Veeraraju Elluru et al. | Deep neural networks often rely on spurious correlations in training data, leading to biased or unfair predictions in safety-critical domains such as medicine and autonomous driving. While conventional bias mitigation typically requires retraining from scratch or redesigning data pipelines, recent advances in machine unlearning provide a promising alternative for post-hoc model correction. In this work, we investigate \textit{Bias-Aware Machine Unlearning}, a paradigm that selectively removes biased samples or feature representations to mitigate diverse forms of bias in vision models. Building on privacy-preserving unlearning techniques, we evaluate various strategies including Gradient Ascent, LoRA, and Teacher-Student distillation. Through empirical analysis on three benchmark datasets, CUB-200-2011 (pose bias), CIFAR-10 (synthetic patch bias), and CelebA (gender bias in smile detection), we demonstrate that post-hoc unlearning can substantially reduce subgroup disparities, with improvements in demographic parity of up to \textbf{94.86\%} on CUB-200, \textbf{30.28\%} on CIFAR-10, and \textbf{97.37\%} on CelebA. These gains are achieved with minimal accuracy loss and with methods scoring an average of 0.62 across the 3 settings on the joint evaluation of utility, fairness, quality, and privacy. Our findings establish machine unlearning as a practical framework for enhancing fairness in deployed vision systems without necessitating full retraining. |
| 2025-09-09 | [Attention and Risk-Aware Decision Framework for Safe Autonomous Driving](http://arxiv.org/abs/2509.07412v1) | Zhen Tian, Fujiang Yuan et al. | Autonomous driving has attracted great interest due to its potential capability in full-unsupervised driving. Model-based and learning-based methods are widely used in autonomous driving. Model-based methods rely on pre-defined models of the environment and may struggle with unforeseen events. Proximal policy optimization (PPO), an advanced learning-based method, can adapt to the above limits by learning from interactions with the environment. However, existing PPO faces challenges with poor training results, and low training efficiency in long sequences. Moreover, the poor training results are equivalent to collisions in driving tasks. To solve these issues, this paper develops an improved PPO by introducing the risk-aware mechanism, a risk-attention decision network, a balanced reward function, and a safety-assisted mechanism. The risk-aware mechanism focuses on highlighting areas with potential collisions, facilitating safe-driving learning of the PPO. The balanced reward function adjusts rewards based on the number of surrounding vehicles, promoting efficient exploration of the control strategy during training. Additionally, the risk-attention network enhances the PPO to hold channel and spatial attention for the high-risk areas of input images. Moreover, the safety-assisted mechanism supervises and prevents the actions with risks of collisions during the lane keeping and lane changing. Simulation results on a physical engine demonstrate that the proposed algorithm outperforms benchmark algorithms in collision avoidance, achieving higher peak reward with less training time, and shorter driving time remaining on the risky areas among multiple testing traffic flow scenarios. |
| 2025-09-09 | [Adaptive Evolutionary Framework for Safe, Efficient, and Cooperative Autonomous Vehicle Interactions](http://arxiv.org/abs/2509.07411v1) | Zhen Tian, Zhihao Lin | Modern transportation systems face significant challenges in ensuring road safety, given serious injuries caused by road accidents. The rapid growth of autonomous vehicles (AVs) has prompted new traffic designs that aim to optimize interactions among AVs. However, effective interactions between AVs remains challenging due to the absence of centralized control. Besides, there is a need for balancing multiple factors, including passenger demands and overall traffic efficiency. Traditional rule-based, optimization-based, and game-theoretic approaches each have limitations in addressing these challenges. Rule-based methods struggle with adaptability and generalization in complex scenarios, while optimization-based methods often require high computational resources. Game-theoretic approaches, such as Stackelberg and Nash games, suffer from limited adaptability and potential inefficiencies in cooperative settings. This paper proposes an Evolutionary Game Theory (EGT)-based framework for AV interactions that overcomes these limitations by utilizing a decentralized and adaptive strategy evolution mechanism. A causal evaluation module (CEGT) is introduced to optimize the evolutionary rate, balancing mutation and evolution by learning from historical interactions. Simulation results demonstrate the proposed CEGT outperforms EGT and popular benchmark games in terms of lower collision rates, improved safety distances, higher speeds, and overall better performance compared to Nash and Stackelberg games across diverse scenarios and parameter settings. |
| 2025-09-09 | [PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions](http://arxiv.org/abs/2509.07370v1) | Yixuan Tang, Yi Yang et al. | Recent advancements in Large Language Models (LLMs) demonstrate remarkable capabilities across various fields. These developments have led to more direct communication between humans and LLMs in various situations, such as social companionship and psychological support. However, LLMs often exhibit limitations in emotional perception and social competence during real-world conversations. These limitations partly originate from their inability to adapt their communication style and emotional expression to different social and task contexts. In this work, we introduce PersonaFuse, a novel LLM post-training framework that enables LLMs to adapt and express different personalities for varying situations. Inspired by Trait Activation Theory and the Big Five personality model, PersonaFuse employs a Mixture-of-Expert architecture that combines persona adapters with a dynamic routing network, enabling contextual trait expression. Experimental results show that PersonaFuse substantially outperforms baseline models across multiple dimensions of social-emotional intelligence. Importantly, these gains are achieved without sacrificing general reasoning ability or model safety, which remain common limitations of direct prompting and supervised fine-tuning approaches. PersonaFuse also delivers consistent improvements in downstream human-centered applications, such as mental health counseling and review-based customer service. Finally, human preference evaluations against leading LLMs, including GPT-4o and DeepSeek, demonstrate that PersonaFuse achieves competitive response quality despite its comparatively smaller model size. These findings demonstrate that PersonaFuse~offers a theoretically grounded and practical approach for developing social-emotional enhanced LLMs, marking a significant advancement toward more human-centric AI systems. |
| 2025-09-09 | [Distributed Frequency Control for Multi-Area Power Systems Considering Transient Frequency Safety](http://arxiv.org/abs/2509.07345v1) | Xiemin Mo, Tao Liu | High penetration of renewable energy sources intensifies frequency fluctuations in multi-area power systems, challenging both stability and operational safety. This paper proposes a novel distributed frequency control method that ensures transient frequency safety and enforces generation capacity constraints, while achieving steady-state frequency restoration and optimal economic operation. The method integrates a feedback optimization (FO)-based controller and a safety corrector. The FO-based controller generates reference setpoints by solving an optimization problem, driving the system to the steady state corresponding to the optimal solution of this problem. The safety corrector then modifies these references using control barrier functions to maintain frequencies within prescribed safe bounds during transients while respecting capacity constraints. The proposed method combines low computational burden with improved regulation performance and enhanced practical applicability. Theoretical analysis establishes optimality, asymptotic stability, and transient frequency safety for the closed-loop system. Simulation studies show that, compared with conventional FO-based schemes, the method consistently enforces frequency safety and capacity limits, achieves smaller frequency deviations and faster recovery, thereby demonstrating its practical effectiveness and advantages. |
| 2025-09-09 | [The bound orbits and gravitational waveforms of timelike particles around renormalization group improved Kerr black holes](http://arxiv.org/abs/2509.07333v1) | Yong-Zhuang Li, Xiao-Mei Kuang | In this article, we investigate the bound orbits of the timelike particles and the gravitational waveforms emitted from these orbits around a renormalization group improved Kerr black hole in the framework of the asymptotic safety approach. The running Newton coupling in the metric is characterized by two free quantum parameters $(\omega,\,\gamma)$ arsing from the non-perturbative renormalization group theory and the appropriate cutoff identification, respectively. As expected, the radii of the horizon, the marginally bound orbits and the innermost stable orbit are all decrease as the quantum parameters increase. Under the extreme mass-ratio inspirals approximation the deviation of gravitational waveforms radiated by the periodic orbits from those in the classical Kerr background increases with the two quantum parameter. However, this effect is much smaller in the retrograde case compared to the prograde case. Especially, by comparing the characteristic strain of those gravitational wave with the sensitivity curve of several potential detectors, we find that their characteristic frequencies can fall within the sensitivity ranges of several planned gravitational wave observatories, suggesting that such signals may be detectable with sufficient instrumental sensitivity. |
| 2025-09-09 | [SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs](http://arxiv.org/abs/2509.07315v1) | Hongfei Xia, Hongru Wang et al. | Large Language Models (LLMs) have exhibited great performance in autonomously calling various tools in external environments, leading to better problem solving and task automation capabilities. However, these external tools also amplify potential risks such as financial loss or privacy leakage with ambiguous or malicious user instructions. Compared to previous studies, which mainly assess the safety awareness of LLMs after obtaining the tool execution results (i.e., retrospective evaluation), this paper focuses on prospective ways to assess the safety of LLM tool utilization, aiming to avoid irreversible harm caused by directly executing tools. To this end, we propose SafeToolBench, the first benchmark to comprehensively assess tool utilization security in a prospective manner, covering malicious user instructions and diverse practical toolsets. Additionally, we propose a novel framework, SafeInstructTool, which aims to enhance LLMs' awareness of tool utilization security from three perspectives (i.e., \textit{User Instruction, Tool Itself, and Joint Instruction-Tool}), leading to nine detailed dimensions in total. We experiment with four LLMs using different methods, revealing that existing approaches fail to capture all risks in tool utilization. In contrast, our framework significantly enhances LLMs' self-awareness, enabling a more safe and trustworthy tool utilization. |
| 2025-09-09 | [Distributed Leader-Follower Consensus for Uncertain Multiagent Systems with Time-Triggered Switching of the Communication Network](http://arxiv.org/abs/2509.07304v1) | Armel Koulong, Ali Pakniyat | A distributed adaptive control strategy is developed for heterogeneous multiagent systems in nonlinear Brunovsky form with \({\pd}\)-dimensional $n^{\text{th}}$-order dynamics, operating under time-triggered switching communication topologies. The approach uses repulsive potential functions to ensure agent-agent and obstacle safety, while neural network estimators compensate for system uncertainties and disturbances. A high-order control barrier function framework is then employed to certify the positive invariance of the safe sets and the boundedness of the proposed control inputs. The resulting distributed control and adaptive laws, together with dwell-time requirements for topology transitions, achieve leader-following consensus. This integrated design provides synchronized formation and robust disturbance rejection in evolving network configurations, and its effectiveness is demonstrated through numerical simulations. |

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



