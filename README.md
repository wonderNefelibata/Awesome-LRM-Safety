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
| 2026-03-03 | [Understanding and Mitigating Dataset Corruption in LLM Steering](http://arxiv.org/abs/2603.03206v1) | Cullen Anderson, Narmeen Oozeer et al. | Contrastive steering has been shown as a simple and effective method to adjust the generative behavior of LLMs at inference time. It uses examples of prompt responses with and without a trait to identify a direction in an intermediate activation layer, and then shifts activations in this 1-dimensional subspace. However, despite its growing use in AI safety applications, the robustness of contrastive steering to noisy or adversarial data corruption is poorly understood. We initiate a study of the robustness of this process with respect to corruption of the dataset of examples used to train the steering direction. Our first observation is that contrastive steering is quite robust to a moderate amount of corruption, but unwanted side effects can be clearly and maliciously manifested when a non-trivial fraction of the training data is altered. Second, we analyze the geometry of various types of corruption, and identify some safeguards. Notably, a key step in learning the steering direction involves high-dimensional mean computation, and we show that replacing this step with a recently developed robust mean estimator often mitigates most of the unwanted effects of malicious corruption. |
| 2026-03-03 | [Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use](http://arxiv.org/abs/2603.03205v1) | Aradhye Agarwal, Gurdit Siyan et al. | Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings. |
| 2026-03-03 | [FEAST: Retrieval-Augmented Multi-Hierarchical Food Classification for the FoodEx2 System](http://arxiv.org/abs/2603.03176v1) | Lorenzo Molfetta, Alessio Cocchieri et al. | Hierarchical text classification (HTC) and extreme multi-label classification (XML) tasks face compounded challenges from complex label interdependencies, data sparsity, and extreme output dimensions. These challenges are exemplified in the European Food Safety Authority's FoodEx2 system-a standardized food classification framework essential for food consumption monitoring and contaminant exposure assessment across Europe. FoodEx2 coding transforms natural language food descriptions into a set of codes from multiple standardized hierarchies, but faces implementation barriers due to its complex structure. Given a food description (e.g., "organic yogurt''), the system identifies its base term ("yogurt''), all the applicable facet categories (e.g., "production method''), and then, every relevant facet descriptors to each category (e.g., "organic production''). While existing models perform adequately on well-balanced and semantically dense hierarchies, no work has been applied on the practical constraints imposed by the FoodEx2 system. The limited literature addressing such real-world scenarios further compounds these challenges. We propose FEAST (Food Embedding And Semantic Taxonomy), a novel retrieval-augmented framework that decomposes FoodEx2 classification into a three-stage approach: (1) base term identification, (2) multi-label facet prediction, and (3) facet descriptor assignment. By leveraging the system's hierarchical structure to guide training and performing deep metric learning, FEASTlearns discriminative embeddings that mitigate data sparsity and improve generalization on rare and fine-grained labels. Evaluated on the multilingual FoodEx2 benchmark, FEAST outperforms the prior European's CNN baseline F1 scores by 12-38 % on rare classes. |
| 2026-03-03 | [Conditioned Activation Transport for T2I Safety Steering](http://arxiv.org/abs/2603.03163v1) | Maciej Chrabąszcz, Aleksander Szymczyk et al. | Despite their impressive capabilities, current Text-to-Image (T2I) models remain prone to generating unsafe and toxic content. While activation steering offers a promising inference-time intervention, we observe that linear activation steering frequently degrades image quality when applied to benign prompts. To address this trade-off, we first construct SafeSteerDataset, a contrastive dataset containing 2300 safe and unsafe prompt pairs with high cosine similarity. Leveraging this data, we propose Conditioned Activation Transport (CAT), a framework that employs a geometry-based conditioning mechanism and nonlinear transport maps. By conditioning transport maps to activate only within unsafe activation regions, we minimize interference with benign queries. We validate our approach on two state-of-the-art architectures: Z-Image and Infinity. Experiments demonstrate that CAT generalizes effectively across these backbones, significantly reducing Attack Success Rate while maintaining image fidelity compared to unsteered generations. Warning: This paper contains potentially offensive text and images. |
| 2026-03-03 | [Deep Q-Learning-Based Gain Scheduling for Nonlinear Quadcopter Dynamics](http://arxiv.org/abs/2603.03127v1) | Hossein Rastgoftar, Muhammad J. H. Zahed | This paper presents a deep Q-network (DQN)-based gain-scheduling framework for safety-critical quadcopter trajectory tracking. Instead of directly learning control inputs, the proposed approach selects from a finite set of pre-certified stabilizing gain vectors, enabling reinforcement learning to operate within a structured and stability-preserving control architecture. By exploiting the isotropic structure of the translational dynamics, feedback gains are shared across spatial axes to reduce dimensionality while preserving performance. The learned policy adapts feedback aggressiveness in real time, applying high authority during large transients and reducing gains near convergence to limit control effort. Simulation results using a high-fidelity nonlinear quadcopter model demonstrate accurate trajectory tracking, bounded attitude excursions, smooth transition to hover after the final time, and consistent reward improvement, validating the effectiveness and robustness of the proposed learning-based gain scheduling strategy. |
| 2026-03-03 | [TAO-Attack: Toward Advanced Optimization-Based Jailbreak Attacks for Large Language Models](http://arxiv.org/abs/2603.03081v1) | Zhi Xu, Jiaqi Li et al. | Large language models (LLMs) have achieved remarkable success across diverse applications but remain vulnerable to jailbreak attacks, where attackers craft prompts that bypass safety alignment and elicit unsafe responses. Among existing approaches, optimization-based attacks have shown strong effectiveness, yet current methods often suffer from frequent refusals, pseudo-harmful outputs, and inefficient token-level updates. In this work, we propose TAO-Attack, a new optimization-based jailbreak method. TAO-Attack employs a two-stage loss function: the first stage suppresses refusals to ensure the model continues harmful prefixes, while the second stage penalizes pseudo-harmful outputs and encourages the model toward more harmful completions. In addition, we design a direction-priority token optimization (DPTO) strategy that improves efficiency by aligning candidates with the gradient direction before considering update magnitude. Extensive experiments on multiple LLMs demonstrate that TAO-Attack consistently outperforms state-of-the-art methods, achieving higher attack success rates and even reaching 100\% in certain scenarios. |
| 2026-03-03 | [TrustMH-Bench: A Comprehensive Benchmark for Evaluating the Trustworthiness of Large Language Models in Mental Health](http://arxiv.org/abs/2603.03047v1) | Zixin Xiong, Ziteng Wang et al. | While Large Language Models (LLMs) demonstrate significant potential in providing accessible mental health support, their practical deployment raises critical trustworthiness concerns due to the domains high-stakes and safety-sensitive nature. Existing evaluation paradigms for general-purpose LLMs fail to capture mental health-specific requirements, highlighting an urgent need to prioritize and enhance their trustworthiness. To address this, we propose TrustMH-Bench, a holistic framework designed to systematically quantify the trustworthiness of mental health LLMs. By establishing a deep mapping from domain-specific norms to quantitative evaluation metrics, TrustMH-Bench evaluates models across eight core pillars: Reliability, Crisis Identification and Escalation, Safety, Fairness, Privacy, Robustness, Anti-sycophancy, and Ethics. We conduct extensive experiments across six general-purpose LLMs and six specialized mental health models. Experimental results indicate that the evaluated models underperform across various trustworthiness dimensions in mental health scenarios, revealing significant deficiencies. Notably, even generally powerful models (e.g., GPT-5.1) fail to maintain consistently high performance across all dimensions. Consequently, systematically improving the trustworthiness of LLMs has become a critical task. Our data and code are released. |
| 2026-03-03 | [Why Does RLAIF Work At All?](http://arxiv.org/abs/2603.03000v1) | Robin Young | Reinforcement Learning from AI Feedback (RLAIF) enables language models to improve by training on their own preference judgments, yet no theoretical account explains why this self-improvement seemingly works for value learning. We propose the latent value hypothesis, that pretraining on internet-scale data encodes human values as directions in representation space, and constitutional prompts elicit these latent values into preference judgments. We formalize this intuition under a linear model where the constitution acts as a projection operator selecting value-relevant directions. Our analysis yields several results. RLAIF improves alignment when the constitution-activated direction correlates with true values better than the model's default generation direction thus explaining the generation-judgment gap; the ceiling on RLAIF quality is determined by how well representations encode values, which scales with model capacity; and adversarial constitutions exist that can activate anti-social value directions encoded from harmful pretraining data. Our account unifies scattered empirical findings including the refusal direction, low-rank safety subspaces, and RLAIF scaling behavior. |
| 2026-03-03 | [Grid-Forming Control with Assignable Voltage Regulation Guarantees and Safety-Critical Current Limiting](http://arxiv.org/abs/2603.02975v1) | Bhathiya Rathnayake, Sijia Geng | This paper develops a nonlinear grid-forming (GFM) controller with provable voltage-formation guarantees, with over-current limiting enforced via a control-barrier-function (CBF)-based safety filter. The nominal controller follows a droop-based inner-outer architecture, in which voltage references and frequency are generated by droop laws, an outer-loop voltage controller produces current references using backstepping (BS), and an inner-loop current controller synthesizes the terminal voltage. The grid voltage is treated as an unknown bounded disturbance, without requiring knowledge of its bound, and the controller design does not rely on any network parameters beyond the point of common coupling (PCC). To robustify voltage formation against the grid voltage, a deadzone-adapted disturbance suppression (DADS) framework is incorporated, yielding practical voltage regulation characterized by asymptotic convergence of the PCC voltage errors to an assignably small and known residual set. Furthermore, the closed-loop system is proven to be globally well posed, with all physical and adaptive states bounded and voltage error transients (due to initial conditions) decaying exponentially at an assignable rate. On top of the nominal controller, hard over-current protection is achieved through a minimally invasive CBF-based safety filter that enforces strict current limits via a single-constraint quadratic program. The safety filter is compatible with any locally Lipschitz nominal controller. Rigorous analysis establishes forward invariance of the safe-current set and boundedness of all states under current limiting. Numerical results demonstrate improved transient performance and faster recovery during current-limiting events when the proposed DADS-BS controller is used as the nominal control law, compared with conventional PI-based GFM control. |
| 2026-03-03 | [Speech recognition assisted by large language models to command software orally -- Application to an augmented and virtual reality web app for immersive molecular graphics](http://arxiv.org/abs/2603.02901v1) | Fabio Cortes Rodriguez, Luciano Abriata | This project successfully developed, evaluated and integrated a Voice User Interface (VUI) into a web application that we are developing for immersive molecular graphics. Said app provides augmented and virtual reality (AR and VR) environments where users manipulate molecules with their hands, but this means the hands can't be used to control the app through a regular mouse- and keyboard-based GUI. The speech-based VUI system developed here alleviates this problem, making it easy to control the app via natural spoken (or typed) commands. To achieve this VUI we evaluated two distinct Automated Speech Recognition (ASR) systems: Chrome's native Speech API and OpenAI's Whisper v3. While Whisper offered broader browser compatibility, its tendency to "hallucinate" with specialized scientific jargon proved very problematic. Consequently, we selected Chrome's ASR for its stability, speed, and reliability. For translating transcribed speech into software commands, we tested two Large Language Model (LLM)-driven approaches: either generating executable code, or calling predefined functions. The function call method, powered by OpenAI's GPT-4o-mini, was ultimately adopted due to its superior safety, efficiency, and reliability over the more complex and error-prone code-generation approach. The resulting VUI is then based on an integration of Chrome's ASR with our LLM-based function-calling module, enabling users to command the application using natural language as shown in a video linked inside this report. We provide links to live examples demonstrating all the intermediate components, and details on how we crafted the LLM's prompt in order to teach it the function calls as well as ways to clean up the transcribed speech and to explain itself while generating function calls. For best demonstration of the final system, we provide a video example. |
| 2026-03-03 | [SIGMark: Scalable In-Generation Watermark with Blind Extraction for Video Diffusion](http://arxiv.org/abs/2603.02882v1) | Xinjie Zhu, Zijing Zhao et al. | Artificial Intelligence Generated Content (AIGC), particularly video generation with diffusion models, has been advanced rapidly. Invisible watermarking is a key technology for protecting AI-generated videos and tracing harmful content, and thus plays a crucial role in AI safety. Beyond post-processing watermarks which inevitably degrade video quality, recent studies have proposed distortion-free in-generation watermarking for video diffusion models. However, existing in-generation approaches are non-blind: they require maintaining all the message-key pairs and performing template-based matching during extraction, which incurs prohibitive computational costs at scale. Moreover, when applied to modern video diffusion models with causal 3D Variational Autoencoders (VAEs), their robustness against temporal disturbance becomes extremely weak. To overcome these challenges, we propose SIGMark, a Scalable In-Generation watermarking framework with blind extraction for video diffusion. To achieve blind-extraction, we propose to generate watermarked initial noise using a Global set of Frame-wise PseudoRandom Coding keys (GF-PRC), reducing the cost of storing large-scale information while preserving noise distribution and diversity for distortion-free watermarking. To enhance robustness, we further design a Segment Group-Ordering module (SGO) tailored to causal 3D VAEs, ensuring robust watermark inversion during extraction under temporal disturbance. Comprehensive experiments on modern diffusion models show that SIGMark achieves very high bit-accuracy during extraction under both temporal and spatial disturbances with minimal overhead, demonstrating its scalability and robustness. Our project is available at https://jeremyzhao1998.github.io/SIGMark-release/. |
| 2026-03-03 | [Emerging trends in Cislunar Space for Lunar Science Exploration and Space Robotics aiding Human Spaceflight Safety](http://arxiv.org/abs/2603.02878v1) | Arsalan Muhammad, Yue Wang et al. | In recent years, the Moon has emerged as an unparalleled extraterrestrial testbed for advancing cuttingedge technological and scientific research critical to enabling sustained human presence on its surface and supporting future interplanetary exploration. This study identifies and investigates two pivotal research domains with substantial transformative potential for accelerating humanity interplanetary aspirations. First is Lunar Science Exploration with Artificial Intelligence and Space Robotics which focusses on AI and Space Robotics redefining the frontiers of space exploration. Second being Space Robotics aid in manned spaceflight to the Moon serving as critical assets for pre-deployment infrastructure development, In-Situ Resource Utilization, surface operations support, and astronaut safety assurance. By integrating autonomy, machine learning, and realtime sensor fusion, space robotics not only augment human capabilities but also serve as force multipliers in achieving sustainable lunar exploration, paving the way for future crewed missions to Mars and beyond. |
| 2026-03-03 | [An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification](http://arxiv.org/abs/2603.02719v1) | L. Julián Lechuga López, Farah E. Shamout et al. | As artificial intelligence systems move toward clinical deployment, ensuring reliable prediction behavior is fundamental for safety-critical decision-making tasks. One proposed safeguard is selective prediction, where models can defer uncertain predictions to human experts for review. In this work, we empirically evaluate the reliability of uncertainty-based selective prediction in multilabel clinical condition classification using multimodal ICU data. Across a range of state-of-the-art unimodal and multimodal models, we find that selective prediction can substantially degrade performance despite strong standard evaluation metrics. This failure is driven by severe class-dependent miscalibration, whereby models assign high uncertainty to correct predictions and low uncertainty to incorrect ones, particularly for underrepresented clinical conditions. Our results show that commonly used aggregate metrics can obscure these effects, limiting their ability to assess selective prediction behavior in this setting. Taken together, our findings characterize a task-specific failure mode of selective prediction in multimodal clinical condition classification and highlight the need for calibration-aware evaluation to provide strong guarantees of safety and robustness in clinical AI. |
| 2026-03-03 | [Retrieval-Augmented Robots via Retrieve-Reason-Act](http://arxiv.org/abs/2603.02688v1) | Izat Temiraliev, Diji Yang et al. | To achieve general-purpose utility, we argue that robots must evolve from passive executors into active Information Retrieval users. In strictly zero-shot settings where no prior demonstrations exist, robots face a critical information gap, such as the exact sequence required to assemble a complex furniture kit, that cannot be satisfied by internal parametric knowledge (common sense) or past internal memory. While recent robotic works attempt to use search before action, they primarily focus on retrieving past kinematic trajectories (analogous to searching internal memory) or text-based safety rules (searching for constraints). These approaches fail to address the core information need of active task construction: acquiring unseen procedural knowledge from external, unstructured documentation. In this paper, we define the paradigm as Retrieval-Augmented Robotics (RAR), empowering the robot with the information-seeking capability that bridges the gap between visual documentation and physical actuation. We formulate the task execution as an iterative Retrieve-Reason-Act loop: the robot or embodied agent actively retrieves relevant visual procedural manuals from an unstructured corpus, grounds the abstract 2D diagrams to 3D physical parts via cross-modal alignment, and synthesizes executable plans. We validate this paradigm on a challenging long-horizon assembly benchmark. Our experiments demonstrate that grounding robotic planning in retrieved visual documents significantly outperforms baselines relying on zero-shot reasoning or few-shot example retrieval. This work establishes the basis of RAR, extending the scope of Information Retrieval from answering user queries to driving embodied physical actions. |
| 2026-03-03 | [HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse](http://arxiv.org/abs/2603.02684v1) | Sai Kartheek Reddy Kasu, Shankar Biradar et al. | Subtle and indirect hate speech remains an underexplored challenge in online safety research, particularly when harmful intent is embedded within misleading or manipulative narratives. Existing hate speech datasets primarily capture overt toxicity, underrepresenting the nuanced ways misinformation can incite or normalize hate. To address this gap, we present HateMirage, a novel dataset of Faux Hate comments designed to advance reasoning and explainability research on hate emerging from fake or distorted narratives. The dataset was constructed by identifying widely debunked misinformation claims from fact-checking sources and tracing related YouTube discussions, resulting in 4,530 user comments. Each comment is annotated along three interpretable dimensions: Target (who is affected), Intent (the underlying motivation or goal behind the comment), and Implication (its potential social impact). Unlike prior explainability datasets such as HateXplain and HARE, which offer token-level or single-dimensional reasoning, HateMirage introduces a multi-dimensional explanation framework that captures the interplay between misinformation, harm, and social consequence. We benchmark multiple open-source language models on HateMirage using ROUGE-L F1 and Sentence-BERT similarity to assess explanation coherence. Results suggest that explanation quality may depend more on pretraining diversity and reasoning-oriented data rather than on model scale alone. By coupling misinformation reasoning with harm attribution, HateMirage establishes a new benchmark for interpretable hate detection and responsible AI research. |
| 2026-03-03 | [MMH-Planner: Multi-Mode Hybrid Trajectory Planning Method for UAV Efficient Flight Based on Real-Time Spatial Awareness](http://arxiv.org/abs/2603.02683v1) | Yinghao Zhao, Chenguang Dai et al. | Motion planning is a critical component of intelligent unmanned systems, enabling their complex autonomous operations. However, current planning algorithms still face limitations in planning efficiency due to inflexible strategies and weak adaptability. To address this, this paper proposes a multi-mode hybrid trajectory planning method for UAVs based on real-time environmental awareness, which dynamically selects the optimal planning model for high-quality trajectory generation in response to environmental changes. First, we introduce a goal-oriented spatial awareness method that rapidly assesses flight safety in the upcoming environments. Second, a multi-mode hybrid trajectory planning mechanism is proposed, which can enhance the planning efficiency by selecting the optimal planning model for trajectory generation based on prior spatial awareness. Finally, we design a lazy replanning strategy that triggers replanning only when necessary to reduce computational resource consumption while maintaining flight quality. To validate the performance of the proposed method, we conducted comprehensive comparative experiments in simulation environments. Results demonstrate that our approach outperforms existing state-of-the-art (SOTA) algorithms across multiple metrics, achieving the best performance particularly in terms of the average number of planning iterations and computational cost per iteration. Furthermore, the effectiveness of our approach is further verified through real-world flight experiments integrated with a self-developed intelligent UAV platform. |
| 2026-03-03 | [From Shallow to Deep: Pinning Semantic Intent via Causal GRPO](http://arxiv.org/abs/2603.02675v1) | Shuyi Zhou, Zeen Song et al. | Large Language Models remain vulnerable to adversarial prefix attacks (e.g., ``Sure, here is'') despite robust standard safety. We diagnose this vulnerability as Shallow Safety Alignment, stemming from a pathology we term semantic representation decay: as the model generates compliant prefixes, its internal malicious intent signal fades. To address this, we propose Two-Stage Causal-GRPO (TSC-GRPO), a framework designed to achieve intent pinning. First, grounded in causal identifiability theory, we train a causal intent probe to disentangle invariant intent from stylistic perturbations. Second, we internalize this causal awareness into the policy via Group Relative Policy Optimization. By employing a cumulative causal penalty within ``fork-in-the-road'' training scenarios, we force the model to learn that accumulating harmful tokens monotonically decreases reward, enabling robust late-stage refusals. Experiments show that TSC-GRPO significantly outperforms baselines in defending against jailbreak attacks while preserving general utility. |
| 2026-03-03 | [Watch Your Step: Learning Semantically-Guided Locomotion in Cluttered Environment](http://arxiv.org/abs/2603.02657v1) | Denan Liang, Yuan Zhu et al. | Although legged robots demonstrate impressive mobility on rough terrain, using them safely in cluttered environments remains a challenge. A key issue is their inability to avoid stepping on low-lying objects, such as high-cost small devices or cables on flat ground. This limitation arises from a disconnection between high-level semantic understanding and low-level control, combined with errors in elevation maps during real-world operation. To address this, we introduce SemLoco, a Reinforcement Learning (RL) framework designed to avoid obstacles precisely in densely cluttered environments. SemLoco uses a two-stage RL approach that combines both soft and hard constraints and performs pixel-wise foothold safety inference, enabling more accurate foot placement. Additionally, SemLoco integrates a semantic map to assign traversability costs rather than relying solely on geometric data. SemLoco significantly reduces collisions and improves safety around sensitive objects, enabling reliable navigation in situations where traditional controllers would likely cause damage. Experimental results further demonstrate that SemLoco can be effectively applied to more complex, unstructured real-world environments. |
| 2026-03-03 | [SaFeR-ToolKit: Structured Reasoning via Virtual Tool Calling for Multimodal Safety](http://arxiv.org/abs/2603.02635v1) | Zixuan Xu, Tiancheng He et al. | Vision-language models remain susceptible to multimodal jailbreaks and over-refusal because safety hinges on both visual evidence and user intent, while many alignment pipelines supervise only the final response. To address this, we present SaFeR-ToolKit, which formalizes safety decision-making as a checkable protocol. Concretely, a planner specifies a persona, a Perception $\to$ Reasoning $\to$ Decision tool set, and a constrained transition graph, while a responder outputs a typed key-value tool trace before the final answer. To make the protocol reliably followed in practice, we train a single policy with a three-stage curriculum (SFT $\to$ DPO $\to$ GRPO), where GRPO directly supervises tool usage beyond answer-level feedback. Our contributions are two-fold: I. Dataset. The first tool-based safety reasoning dataset, comprising 31,654 examples (SFT 6k, DPO 18.6k, GRPO 6k) plus 1k held-out evaluation. II. Experiments. On Qwen2.5-VL, SaFeR-ToolKit significantly improves Safety/Helpfulness/Reasoning Rigor on 3B (29.39/45.04/4.98 $\to$ 84.40/71.13/78.87) and 7B (53.21/52.92/19.26 $\to$ 86.34/80.79/85.34), while preserving general capabilities (3B: 58.67 $\to$ 59.21; 7B: 66.39 $\to$ 66.81). Codes are available at https://github.com/Duebassx/SaFeR_ToolKit. |
| 2026-03-03 | [ExpGuard: LLM Content Moderation in Specialized Domains](http://arxiv.org/abs/2603.02588v1) | Minseok Choi, Dongjin Kim et al. | With the growing deployment of large language models (LLMs) in real-world applications, establishing robust safety guardrails to moderate their inputs and outputs has become essential to ensure adherence to safety policies. Current guardrail models predominantly address general human-LLM interactions, rendering LLMs vulnerable to harmful and adversarial content within domain-specific contexts, particularly those rich in technical jargon and specialized concepts. To address this limitation, we introduce ExpGuard, a robust and specialized guardrail model designed to protect against harmful prompts and responses across financial, medical, and legal domains. In addition, we present ExpGuardMix, a meticulously curated dataset comprising 58,928 labeled prompts paired with corresponding refusal and compliant responses, from these specific sectors. This dataset is divided into two subsets: ExpGuardTrain, for model training, and ExpGuardTest, a high-quality test set annotated by domain experts to evaluate model robustness against technical and domain-specific content. Comprehensive evaluations conducted on ExpGuardTest and eight established public benchmarks reveal that ExpGuard delivers competitive performance across the board while demonstrating exceptional resilience to domain-specific adversarial attacks, surpassing state-of-the-art models such as WildGuard by up to 8.9% in prompt classification and 15.3% in response classification. To encourage further research and development, we open-source our code, data, and model, enabling adaptation to additional domains and supporting the creation of increasingly robust guardrail models. |

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



