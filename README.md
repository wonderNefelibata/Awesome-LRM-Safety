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
| 2026-01-30 | [IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models](http://arxiv.org/abs/2601.23266v1) | Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian et al. | This paper proposes a novel inverse reinforcement learning framework using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals. Reinforcement learning (RL) is then performed with a hybrid reward that combines diffuse environmental feedback and targeted IRL rewards. A conditional diffusion model, which acts as a safety supervisor, plans safe paths. It stays in its lane, avoids obstacles, and moves smoothly. Then, a learnable adaptive mask (LAM) improves perception. It shifts visual attention based on vehicle speed and nearby hazards. After FSM-based imitation, the policy is fine-tuned with Proximal Policy Optimization (PPO). Training is run in the Webots simulator with a two-stage curriculum. A 96\% success rate is reached, and collisions are reduced to 0.05 per 1k steps, marking a new benchmark for safe navigation. By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness.We make our code publicly available. |
| 2026-01-30 | [Now You Hear Me: Audio Narrative Attacks Against Large Audio-Language Models](http://arxiv.org/abs/2601.23255v1) | Ye Yu, Haibo Jin et al. | Large audio-language models increasingly operate on raw speech inputs, enabling more seamless integration across domains such as voice assistants, education, and clinical triage. This transition, however, introduces a distinct class of vulnerabilities that remain largely uncharacterized. We examine the security implications of this modality shift by designing a text-to-audio jailbreak that embeds disallowed directives within a narrative-style audio stream. The attack leverages an advanced instruction-following text-to-speech (TTS) model to exploit structural and acoustic properties, thereby circumventing safety mechanisms primarily calibrated for text. When delivered through synthetic speech, the narrative format elicits restricted outputs from state-of-the-art models, including Gemini 2.0 Flash, achieving a 98.26% success rate that substantially exceeds text-only baselines. These results highlight the need for safety frameworks that jointly reason over linguistic and paralinguistic representations, particularly as speech-based interfaces become more prevalent. |
| 2026-01-30 | [Video-o3: Native Interleaved Clue Seeking for Long Video Multi-Hop Reasoning](http://arxiv.org/abs/2601.23224v1) | Xiangyu Zeng, Zhiqiu Zhang et al. | Existing multimodal large language models for long-video understanding predominantly rely on uniform sampling and single-turn inference, limiting their ability to identify sparse yet critical evidence amid extensive redundancy. We introduce Video-o3, a novel framework that supports iterative discovery of salient visual clues, fine-grained inspection of key segments, and adaptive termination once sufficient evidence is acquired. Technically, we address two core challenges in interleaved tool invocation. First, to mitigate attention dispersion induced by the heterogeneity of reasoning and tool-calling, we propose Task-Decoupled Attention Masking, which isolates per-step concentration while preserving shared global context. Second, to control context length growth in multi-turn interactions, we introduce a Verifiable Trajectory-Guided Reward that balances exploration coverage with reasoning efficiency. To support training at scale, we further develop a data synthesis pipeline and construct Seeker-173K, comprising 173K high-quality tool-interaction trajectories for effective supervised and reinforcement learning. Extensive experiments show that Video-o3 substantially outperforms state-of-the-art methods, achieving 72.1% accuracy on MLVU and 46.5% on Video-Holmes. These results demonstrate Video-o3's strong multi-hop evidence-seeking and reasoning capabilities, and validate the effectiveness of native tool invocation in long-video scenarios. |
| 2026-01-30 | [TriSpec: Ternary Speculative Decoding via Lightweight Proxy Verification](http://arxiv.org/abs/2601.23180v1) | Haoyun Jiang, Junqi He et al. | Inference efficiency in Large Language Models (LLMs) is fundamentally limited by their serial, autoregressive generation, especially as reasoning becomes a key capability and response sequences grow longer. Speculative decoding (SD) offers a powerful solution, providing significant speed-ups through its lightweight drafting and parallel verification mechanism. While existing work has nearly saturated improvements in draft effectiveness and efficiency, this paper advances SD from a new yet critical perspective: the verification cost. We propose TriSpec, a novel ternary SD framework that, at its core, introduces a lightweight proxy to significantly reduce computational cost by approving easily verifiable draft sequences and engaging the full target model only when encountering uncertain tokens. TriSpec can be integrated with state-of-the-art SD methods like EAGLE-3 to further reduce verification costs, achieving greater acceleration. Extensive experiments on the Qwen3 and DeepSeek-R1-Distill-Qwen/LLaMA families show that TriSpec achieves up to 35\% speedup over standard SD, with up to 50\% fewer target model invocations while maintaining comparable accuracy. |
| 2026-01-30 | [On Safer Reinforcement Learning Policies for Sedation and Analgesia in Intensive Care](http://arxiv.org/abs/2601.23154v1) | Joel Romero-Hernandez, Oscar Camara | Pain management in intensive care usually involves complex trade-offs between therapeutic goals and patient safety, since both inadequate and excessive treatment may induce serious sequelae. Reinforcement learning can help address this challenge by learning medication dosing policies from retrospective data. However, prior work on sedation and analgesia has optimized for objectives that do not value patient survival while relying on algorithms unsuitable for imperfect information settings. We investigated the risks of these design choices by implementing a deep reinforcement learning framework to suggest hourly medication doses under partial observability. Using data from 47,144 ICU stays in the MIMIC-IV database, we trained policies to prescribe opioids, propofol, benzodiazepines, and dexmedetomidine according to two goals: reduce pain or jointly reduce pain and mortality. We found that, although the two policies were associated with lower pain, actions from the first policy were positively correlated with mortality, while those proposed by the second policy were negatively correlated. This suggests that valuing long-term outcomes could be critical for safer treatment policies, even if a short-term goal remains the primary objective. |
| 2026-01-30 | [THINKSAFE: Self-Generated Safety Alignment for Reasoning Models](http://arxiv.org/abs/2601.23143v1) | Seanie Lee, Sangwoo Park et al. | Large reasoning models (LRMs) achieve remarkable performance by leveraging reinforcement learning (RL) on reasoning tasks to generate long chain-of-thought (CoT) reasoning. However, this over-optimization often prioritizes compliance, making models vulnerable to harmful prompts. To mitigate this safety degradation, recent approaches rely on external teacher distillation, yet this introduces a distributional discrepancy that degrades native reasoning. We propose ThinkSafe, a self-generated alignment framework that restores safety alignment without external teachers. Our key insight is that while compliance suppresses safety mechanisms, models often retain latent knowledge to identify harm. ThinkSafe unlocks this via lightweight refusal steering, guiding the model to generate in-distribution safety reasoning traces. Fine-tuning on these self-generated responses effectively realigns the model while minimizing distribution shift. Experiments on DeepSeek-R1-Distill and Qwen3 show ThinkSafe significantly improves safety while preserving reasoning proficiency. Notably, it achieves superior safety and comparable reasoning to GRPO, with significantly reduced computational cost. Code, models, and datasets are available at https://github.com/seanie12/ThinkSafe.git. |
| 2026-01-30 | [How should AI Safety Benchmarks Benchmark Safety?](http://arxiv.org/abs/2601.23112v1) | Cheng Yu, Severin Engelmann et al. | AI safety benchmarks are pivotal for safety in advanced AI systems; however, they have significant technical, epistemic, and sociotechnical shortcomings. We present a review of 210 safety benchmarks that maps out common challenges in safety benchmarking, documenting failures and limitations by drawing from engineering sciences and long-established theories of risk and safety. We argue that adhering to established risk management principles, mapping the space of what can(not) be measured, developing robust probabilistic metrics, and efficiently deploying measurement theory to connect benchmarking objectives with the world can significantly improve the validity and usefulness of AI safety benchmarks. The review provides a roadmap on how to improve AI safety benchmarking, and we illustrate the effectiveness of these recommendations through quantitative and qualitative evaluation. We also introduce a checklist that can help researchers and practitioners develop robust and epistemologically sound safety benchmarks. This study advances the science of benchmarking and helps practitioners deploy AI systems more responsibly. |
| 2026-01-30 | [FlowCalib: LiDAR-to-Vehicle Miscalibration Detection using Scene Flows](http://arxiv.org/abs/2601.23107v1) | Ilir Tahiraj, Peter Wittal et al. | Accurate sensor-to-vehicle calibration is essential for safe autonomous driving. Angular misalignments of LiDAR sensors can lead to safety-critical issues during autonomous operation. However, current methods primarily focus on correcting sensor-to-sensor errors without considering the miscalibration of individual sensors that cause these errors in the first place. We introduce FlowCalib, the first framework that detects LiDAR-to-vehicle miscalibration using motion cues from the scene flow of static objects. Our approach leverages the systematic bias induced by rotational misalignment in the flow field generated from sequential 3D point clouds, eliminating the need for additional sensors. The architecture integrates a neural scene flow prior for flow estimation and incorporates a dual-branch detection network that fuses learned global flow features with handcrafted geometric descriptors. These combined representations allow the system to perform two complementary binary classification tasks: a global binary decision indicating whether misalignment is present and separate, axis-specific binary decisions indicating whether each rotational axis is misaligned. Experiments on the nuScenes dataset demonstrate FlowCalib's ability to robustly detect miscalibration, establishing a benchmark for sensor-to-vehicle miscalibration detection. |
| 2026-01-30 | [Exploring Sidewalk Sheds in New York City through Chatbot Surveys and Human Computer Interaction](http://arxiv.org/abs/2601.23095v1) | Junyi Li, Zhaoxi Zhang et al. | Sidewalk sheds are a common feature of the streetscape in New York City, reflecting ongoing construction and maintenance activities. However, policymakers and local business owners have raised concerns about reduced storefront visibility and altered pedestrian navigation. Although sidewalk sheds are widely used for safety, their effects on pedestrian visibility and movement are not directly measured in current planning practices. To address this, we developed an AI-based chatbot survey that collects image-based annotations and route choices from pedestrians, linking these responses to specific shed design features, including clearance height, post spacing, and color. This AI chatbot survey integrates a large language model (e.g., Google's Gemini-1.5-flash-001 model) with an image-annotation interface, allowing users to interact with street images, mark visual elements, and provide structured feedback through guided dialogue. To explore pedestrian perceptions and behaviors, this paper conducts a grid-based analysis of entrance annotations and applies logistic mixed-effects modeling to assess sidewalk choice patterns. Analysis of the dataset (n = 25) shows that: (1) the presence of scaffolding significantly reduces pedestrians' ability to identify ground-floor retail entrances, and (2) variations in weather conditions and shed design features significantly influence sidewalk selection behavior. By integrating generative AI into urban research, this study demonstrates a novel method for evaluating sidewalk shed designs and provides empirical evidence to support adjustments to shed guidelines that improve the pedestrian experience without compromising safety. |
| 2026-01-30 | [Safer Policy Compliance with Dynamic Epistemic Fallback](http://arxiv.org/abs/2601.23094v1) | Joseph Marvin Imperial, Harish Tayyar Madabushi | Humans develop a series of cognitive defenses, known as epistemic vigilance, to combat risks of deception and misinformation from everyday interactions. Developing safeguards for LLMs inspired by this mechanism might be particularly helpful for their application in high-stakes tasks such as automating compliance with data privacy laws. In this paper, we introduce Dynamic Epistemic Fallback (DEF), a dynamic safety protocol for improving an LLM's inference-time defenses against deceptive attacks that make use of maliciously perturbed policy texts. Through various levels of one-sentence textual cues, DEF nudges LLMs to flag inconsistencies, refuse compliance, and fallback to their parametric knowledge upon encountering perturbed policy texts. Using globally recognized legal policies such as HIPAA and GDPR, our empirical evaluations report that DEF effectively improves the capability of frontier LLMs to detect and refuse perturbed versions of policies, with DeepSeek-R1 achieving a 100% detection rate in one setting. This work encourages further efforts to develop cognitively inspired defenses to improve LLM robustness against forms of harm and deception that exploit legal artifacts. |
| 2026-01-30 | [Character as a Latent Variable in Large Language Models: A Mechanistic Account of Emergent Misalignment and Conditional Safety Failures](http://arxiv.org/abs/2601.23081v1) | Yanghao Su, Wenbo Zhou et al. | Emergent Misalignment refers to a failure mode in which fine-tuning large language models (LLMs) on narrowly scoped data induces broadly misaligned behavior. Prior explanations mainly attribute this phenomenon to the generalization of erroneous or unsafe content. In this work, we show that this view is incomplete. Across multiple domains and model families, we find that fine-tuning models on data exhibiting specific character-level dispositions induces substantially stronger and more transferable misalignment than incorrect-advice fine-tuning, while largely preserving general capabilities. This indicates that emergent misalignment arises from stable shifts in model behavior rather than from capability degradation or corrupted knowledge. We further show that such behavioral dispositions can be conditionally activated by both training-time triggers and inference-time persona-aligned prompts, revealing shared structure across emergent misalignment, backdoor activation, and jailbreak susceptibility. Overall, our results identify character formation as a central and underexplored alignment risk, suggesting that robust alignment must address behavioral dispositions rather than isolated errors or prompt-level defenses. |
| 2026-01-30 | [One-shot Optimized Steering Vector for Hallucination Mitigation for VLMs](http://arxiv.org/abs/2601.23041v1) | Youxu Shi, Suorong Yang et al. | Vision Language Models (VLMs) achieve strong performance on multimodal tasks but still suffer from hallucination and safety-related failures that persist even at scale. Steering offers a lightweight technique to improve model performance. However, steering, whether input-dependent or input-independent, achieves a meaningful trade-off between efficiency and effectiveness. In this work, we observe that steering vectors can generalize across inputs when tasks share aligned semantic intent. Based on this insight, we propose \textbf{OSGA} (\textbf{O}ne-shot \textbf{S}teering with \textbf{G}enerative \textbf{A}nchor), an input-independent framework that improves model performance with a single optimization instance. OSGA first selects an informative sample via a variance-based data selection strategy and learns a single steering vector with a contrastive objective with generative anchor regularization. The resulting vector can be universally applied at a certain layer during inference time without modifying model parameters. Experiments across multiple benchmarks show that a single OSGA-optimized steering vector consistently improves hallucination mitigation and safety enhancement with negligible overhead, highlighting one-shot steering as a practical and scalable solution for reliable VLMs. |
| 2026-01-30 | [Divide-and-Conquer CoT: RL for Reducing Latency via Parallel Reasoning](http://arxiv.org/abs/2601.23027v1) | Arvind Mahankali, Kaiyue Wen et al. | Long chain-of-thought reasoning (Long CoT) is now fundamental to state-of-the-art LLMs, especially in mathematical reasoning. However, LLM generation is highly sequential, and long CoTs lead to a high latency. We propose to train Divide-and-Conquer CoT (DC-CoT) to reduce the latency. With DC-CoT, the model can act as a director that identifies distinct subtasks that can be performed in parallel in its reasoning process, and then spawns workers to execute the subtasks. Our goal is to achieve high accuracy, with a low longest path length, which is a theoretical measure of the latency needed for the response. We start with a long CoT base model (DeepScaleR-1.5B-Preview), and first use SFT with a small curated demonstration set to initialize its ability to spawn workers in a certain format. Because SFT degrades the accuracy significantly, we design a multi-stage RL algorithm, with various data filtering strategies, to recover the accuracy while decreasing the longest path length. Across several benchmarks including AIME 2024 and HMMT 2025, DC-CoT achieves similar accuracy as DeepScaleR-1.5B-Preview while decreasing longest path length by 35-40%. Our code, SFT dataset and models are publicly available at https://github.com/amahankali10/DC_CoT_RL_for_Low_Latency_CoT_with_Parallel_Reasoning. |
| 2026-01-30 | [Toward Fully Autonomous Driving: AI, Challenges, Opportunities, and Needs](http://arxiv.org/abs/2601.22927v1) | Lars Ullrich, Michael Buchholz et al. | Automated driving (AD) is promising, but the transition to fully autonomous driving is, among other things, subject to the real, ever-changing open world and the resulting challenges. However, research in the field of AD demonstrates the ability of artificial intelligence (AI) to outperform classical approaches, handle higher complexities, and reach a new level of autonomy. At the same time, the use of AI raises further questions of safety and transferability. To identify the challenges and opportunities arising from AI concerning autonomous driving functionalities, we have analyzed the current state of AD, outlined limitations, and identified foreseeable technological possibilities. Thereby, various further challenges are examined in the context of prospective developments. In this way, this article reconsiders fully autonomous driving with respect to advancements in the field of AI and carves out the respective needs and resulting research questions. |
| 2026-01-30 | [A Serverless Edge-Native Data Processing Architecture for Autonomous Driving Training](http://arxiv.org/abs/2601.22919v1) | Fabian Bally, Michael Schötz et al. | Data is both the key enabler and a major bottleneck for machine learning in autonomous driving. Effective model training requires not only large quantities of sensor data but also balanced coverage that includes rare yet safety-critical scenarios. Capturing such events demands extensive driving time and efficient selection. This paper introduces the Lambda framework, an edge-native platform that enables on-vehicle data filtering and processing through user-defined functions. The framework provides a serverless-inspired abstraction layer that separates application logic from low-level execution concerns such as scheduling, deployment, and isolation. By adapting Function-as-a-Service (FaaS) principles to resource-constrained automotive environments, it allows developers to implement modular, event-driven filtering algorithms while maintaining compatibility with ROS 2 and existing data recording pipelines. We evaluate the framework on an NVIDIA Jetson Orin Nano and compare it against native ROS 2 deployments. Results show competitive performance, reduced latency and jitter, and confirm that lambda-based abstractions can support real-time data processing in embedded autonomous driving systems. The source code is available at https://github.com/LASFAS/jblambda. |
| 2026-01-30 | [A Comparative Evaluation of Large Vision-Language Models for 2D Object Detection under SOTIF Conditions](http://arxiv.org/abs/2601.22830v1) | Ji Zhou, Yilin Ding et al. | Reliable environmental perception remains one of the main obstacles for safe operation of automated vehicles. Safety of the Intended Functionality (SOTIF) concerns safety risks from perception insufficiencies, particularly under adverse conditions where conventional detectors often falter. While Large Vision-Language Models (LVLMs) demonstrate promising semantic reasoning, their quantitative effectiveness for safety-critical 2D object detection is underexplored. This paper presents a systematic evaluation of ten representative LVLMs using the PeSOTIF dataset, a benchmark specifically curated for long-tail traffic scenarios and environmental degradations. Performance is quantitatively compared against the classical perception approach, a YOLO-based detector. Experimental results reveal a critical trade-off: top-performing LVLMs (e.g., Gemini 3, Doubao) surpass the YOLO baseline in recall by over 25% in complex natural scenarios, exhibiting superior robustness to visual degradation. Conversely, the baseline retains an advantage in geometric precision for synthetic perturbations. These findings highlight the complementary strengths of semantic reasoning versus geometric regression, supporting the use of LVLMs as high-level safety validators in SOTIF-oriented automated driving systems. |
| 2026-01-30 | [Constructing Safety Cases for AI Systems: A Reusable Template Framework](http://arxiv.org/abs/2601.22773v1) | Sung Une Lee, Liming Zhu et al. | Safety cases, structured arguments that a system is acceptably safe, are becoming central to the governance of AI systems. Yet, traditional safety-case practices from aviation or nuclear engineering rely on well-specified system boundaries, stable architectures, and known failure modes. Modern AI systems such as generative and agentic AI are the opposite. Their capabilities emerge unpredictably from low-level training objectives, their behaviour varies with prompts, and their risk profiles shift through fine-tuning, scaffolding, or deployment context. This study examines how safety cases are currently constructed for AI systems and why classical approaches fail to capture these dynamics. It then proposes a framework of reusable safety-case templates, each following a predefined structure of claims, arguments, and evidence tailored for AI systems. The framework introduces comprehensive taxonomies for AI-specific claim types (assertion-based, constrained-based, capability-based), argument types (demonstrative, comparative, causal/explanatory, risk-based, and normative), and evidence families (empirical, mechanistic, comparative, expert-driven, formal methods, operational/field data, and model-based). Each template is illustrated through end-to-end patterns addressing distinctive challenges such as evaluation without ground truth, dynamic model updates, and threshold-based risk decisions. The result is a systematic, composable, and reusable approach to constructing and maintaining safety cases that are credible, auditable, and adaptive to the evolving behaviour of generative and frontier AI systems. |
| 2026-01-30 | [Qualitative Evaluation of LLM-Designed GUI](http://arxiv.org/abs/2601.22759v1) | Bartosz Sawicki, Tomasz Les et al. | As generative artificial intelligence advances, Large Language Models (LLMs) are being explored for automated graphical user interface (GUI) design. This study investigates the usability and adaptability of LLM-generated interfaces by analysing their ability to meet diverse user needs. The experiments included utilization of three state-of-the-art models from January 2025 (OpenAI GPT o3-mini-high, DeepSeek R1, and Anthropic Claude 3.5 Sonnet) generating mockups for three interface types: a chat system, a technical team panel, and a manager dashboard. Expert evaluations revealed that while LLMs are effective at creating structured layouts, they face challenges in meeting accessibility standards and providing interactive functionality. Further testing showed that LLMs could partially tailor interfaces for different user personas but lacked deeper contextual understanding. The results suggest that while LLMs are promising tools for early-stage UI prototyping, human intervention remains critical to ensure usability, accessibility, and user satisfaction. |
| 2026-01-30 | [Lingua-SafetyBench: A Benchmark for Safety Evaluation of Multilingual Vision-Language Models](http://arxiv.org/abs/2601.22737v1) | Enyi Shi, Pengyang Shao et al. | Robust safety of vision-language large models (VLLMs) under joint multilingual and multimodal inputs remains underexplored. Existing benchmarks are typically multilingual but text-only, or multimodal but monolingual. Recent multilingual multimodal red-teaming efforts render harmful prompts into images, yet rely heavily on typography-style visuals and lack semantically grounded image-text pairs, limiting coverage of realistic cross-modal interactions. We introduce Lingua-SafetyBench, a benchmark of 100,440 harmful image-text pairs across 10 languages, explicitly partitioned into image-dominant and text-dominant subsets to disentangle risk sources. Evaluating 11 open-source VLLMs reveals a consistent asymmetry: image-dominant risks yield higher ASR in high-resource languages, while text-dominant risks are more severe in non-high-resource languages. A controlled study on the Qwen series shows that scaling and version upgrades reduce Attack Success Rate (ASR) overall but disproportionately benefit HRLs, widening the gap between HRLs and Non-HRLs under text-dominant risks. This underscores the necessity of language- and modality-aware safety alignment beyond mere scaling.To facilitate reproducibility and future research, we will publicly release our benchmark, model checkpoints, and source code.The code and dataset will be available at https://github.com/zsxr15/Lingua-SafetyBench.Warning: this paper contains examples with unsafe content. |
| 2026-01-30 | [Statistical Estimation of Adversarial Risk in Large Language Models under Best-of-N Sampling](http://arxiv.org/abs/2601.22636v1) | Mingqian Feng, Xiaodong Liu et al. | Large Language Models (LLMs) are typically evaluated for safety under single-shot or low-budget adversarial prompting, which underestimates real-world risk. In practice, attackers can exploit large-scale parallel sampling to repeatedly probe a model until a harmful response is produced. While recent work shows that attack success increases with repeated sampling, principled methods for predicting large-scale adversarial risk remain limited. We propose a scaling-aware Best-of-N estimation of risk, SABER, for modeling jailbreak vulnerability under Best-of-N sampling. We model sample-level success probabilities using a Beta distribution, the conjugate prior of the Bernoulli distribution, and derive an analytic scaling law that enables reliable extrapolation of large-N attack success rates from small-budget measurements. Using only n=100 samples, our anchored estimator predicts ASR@1000 with a mean absolute error of 1.66, compared to 12.04 for the baseline, which is an 86.2% reduction in estimation error. Our results reveal heterogeneous risk scaling profiles and show that models appearing robust under standard evaluation can experience rapid nonlinear risk amplification under parallel adversarial pressure. This work provides a low-cost, scalable methodology for realistic LLM safety assessment. We will release our code and evaluation scripts upon publication to future research. |

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



