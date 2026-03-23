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
| 2026-03-20 | [Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation](http://arxiv.org/abs/2603.20172v1) | Richard J. Young | Recent work on chain-of-thought (CoT) faithfulness reports single aggregate numbers (e.g., DeepSeek-R1 acknowledges hints 39% of the time), implying that faithfulness is an objective, measurable property of a model. This paper demonstrates that it is not. Three classifiers (a regex-only detector, a two-stage regex-plus-LLM pipeline, and an independent Claude Sonnet 4 judge) are applied to 10,276 influenced reasoning traces from 12 open-weight models spanning 9 families and 7B to 1T parameters. On identical data, these classifiers produce overall faithfulness rates of 74.4%, 82.6%, and 69.7%, respectively, with non-overlapping 95% confidence intervals. Per-model gaps range from 2.6 to 30.6 percentage points; all are statistically significant (McNemar's test, p < 0.001). The disagreements are systematic, not random: inter-classifier agreement measured by Cohen's kappa ranges from 0.06 ("slight") for sycophancy hints to 0.42 ("moderate") for grader hints, and the asymmetry is pronounced: for sycophancy, 883 cases are classified as faithful by the pipeline but unfaithful by the Sonnet judge, while only 2 go the other direction. Classifier choice can also reverse model rankings: Qwen3.5-27B ranks 1st under the pipeline but 7th under the Sonnet judge; OLMo-3.1-32B moves in the opposite direction, from 9th to 3rd. The root cause is that different classifiers operationalize related faithfulness constructs at different levels of stringency (lexical mention versus epistemic dependence), and these constructs yield divergent measurements on the same behavior. These results demonstrate that published faithfulness numbers cannot be meaningfully compared across studies that use different classifiers, and that future evaluations should report sensitivity ranges across multiple classification methodologies rather than single point estimates. |
| 2026-03-20 | [Evaluating Evidence Grounding Under User Pressure in Instruction-Tuned Language Models](http://arxiv.org/abs/2603.20162v1) | Sai Koneru, Elphin Joe et al. | In contested domains, instruction-tuned language models must balance user-alignment pressures against faithfulness to the in-context evidence. To evaluate this tension, we introduce a controlled epistemic-conflict framework grounded in the U.S. National Climate Assessment. We conduct fine-grained ablations over evidence composition and uncertainty cues across 19 instruction-tuned models spanning 0.27B to 32B parameters. Across neutral prompts, richer evidence generally improves evidence-consistent accuracy and ordinal scoring performance. Under user pressure, however, evidence does not reliably prevent user-aligned reversals in this controlled fixed-evidence setting. We report three primary failure modes. First, we identify a negative partial-evidence interaction, where adding epistemic nuance, specifically research gaps, is associated with increased susceptibility to sycophancy in families like Llama-3 and Gemma-3. Second, robustness scales non-monotonically: within some families, certain low-to-mid scale models are especially sensitive to adversarial user pressure. Third, models differ in distributional concentration under conflict: some instruction-tuned models maintain sharply peaked ordinal distributions under pressure, while others are substantially more dispersed; in scale-matched Qwen comparisons, reasoning-distilled variants (DeepSeek-R1-Qwen) exhibit consistently higher dispersion than their instruction-tuned counterparts. These findings suggest that, in a controlled fixed-evidence setting, providing richer in-context evidence alone offers no guarantee against user pressure without explicit training for epistemic integrity. |
| 2026-03-20 | [KUKAloha: A General, Low-Cost, and Shared-Control based Teleoperation Framework for Construction Robot Arm](http://arxiv.org/abs/2603.20129v1) | Yifan Xu, Qizhang Shen et al. | This paper presents KUKAloha, a general, low-cost, and shared-control teleoperation framework designed for construction robot arms. The proposed system employs a leader-follower paradigm in which a lightweight leading arm enables intuitive human guidance for coarse robot motion, while an autonomous perception module based on AprilTag detection performs precise alignment and grasp execution. By explicitly decoupling human control from fine manipulation, KUKAloha improves safety and repeatability when operating large-scale manipulators. We implement the framework on a KUKA robot arm and conduct a usability study with representative construction manipulation tasks. Experimental results demonstrate that KUKAloha reduces operator workload, improves task completion efficiency, and provides a practical solution for scalable demonstration collection and shared human-robot control in construction environments. |
| 2026-03-20 | [Evolving Jailbreaks: Automated Multi-Objective Long-Tail Attacks on Large Language Models](http://arxiv.org/abs/2603.20122v1) | Wenjing Hong, Zhonghua Rong et al. | Large Language Models (LLMs) have been widely deployed, especially through free Web-based applications that expose them to diverse user-generated inputs, including those from long-tail distributions such as low-resource languages and encrypted private data. This open-ended exposure increases the risk of jailbreak attacks that undermine model safety alignment. While recent studies have shown that leveraging long-tail distributions can facilitate such jailbreaks, existing approaches largely rely on handcrafted rules, limiting the systematic evaluation of these security and privacy vulnerabilities. In this work, we present EvoJail, an automated framework for discovering long-tail distribution attacks via multi-objective evolutionary search. EvoJail formulates long-tail attack prompt generation as a multi-objective optimization problem that jointly maximizes attack effectiveness and minimizes output perplexity, and introduces a semantic-algorithmic solution representation to capture both high-level semantic intent and low-level structural transformations of encryption-decryption logic. Building upon this representation, EvoJail integrates LLM-assisted operators into a multi-objective evolutionary framework, enabling adaptive and semantically informed mutation and crossover for efficiently exploring a highly structured and open-ended search space. Extensive experiments demonstrate that EvoJail consistently discovers diverse and effective long-tail jailbreak strategies, achieving competitive performance with existing methods in both individual and ensemble level. |
| 2026-03-20 | [Not an Obstacle for Dog, but a Hazard for Human: A Co-Ego Navigation System for Guide Dog Robots](http://arxiv.org/abs/2603.20121v1) | Ruiping Liu, Jingqi Zhang et al. | Guide dogs offer independence to Blind and Low-Vision (BLV) individuals, yet their limited availability leaves the vast majority of BLV users without access. Quadruped robotic guide dogs present a promising alternative, but existing systems rely solely on the robot's ground-level sensors for navigation, overlooking a critical class of hazards: obstacles that are transparent to the robot yet dangerous at human body height, such as bent branches. We term this the viewpoint asymmetry problem and present the first system to explicitly address it. Our Co-Ego system adopts a dual-branch obstacle avoidance framework that integrates the robot-centric ground sensing with the user's elevated egocentric perspective to ensure comprehensive navigation safety. Deployed on a quadruped robot, the system is evaluated in a controlled user study with sighted participants under blindfold across three conditions: unassisted, single-view, and cross-view fusion. Results demonstrate that cross-view fusion significantly reduces collision times and cognitive load, verifying the necessity of viewpoint complementarity for safe robotic guide dog navigation. |
| 2026-03-20 | [Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition](http://arxiv.org/abs/2603.20108v1) | Krzysztof Kotowski, Ramez Shendy et al. | Forecasting plays a crucial role in modern safety-critical applications, such as space operations. However, the increasing use of deep forecasting models introduces a new security risk of trojan horse attacks, carried out by hiding a backdoor in the training data or directly in the model weights. Once implanted, the backdoor is activated by a specific trigger pattern at test time, causing the model to produce manipulated predictions. We focus on this issue in our \textit{Trojan Horse Hunt} data science competition, where more than 200 teams faced the task of identifying triggers hidden in deep forecasting models for spacecraft telemetry. We describe the novel task formulation, benchmark set, evaluation protocol, and best solutions from the competition. We further summarize key insights and research directions for effective identification of triggers in time series forecasting models. All materials are publicly available on the official competition webpage https://www.kaggle.com/competitions/trojan-horse-hunt-in-space. |
| 2026-03-20 | [Diffusion-Based Makeup Transfer with Facial Region-Aware Makeup Features](http://arxiv.org/abs/2603.20012v1) | Zheng Gao, Debin Meng et al. | Current diffusion-based makeup transfer methods commonly use the makeup information encoded by off-the-shelf foundation models (e.g., CLIP) as condition to preserve the makeup style of reference image in the generation. Although effective, these works mainly have two limitations: (1) foundation models pre-trained for generic tasks struggle to capture makeup styles; (2) the makeup features of reference image are injected to the diffusion denoising model as a whole for global makeup transfer, overlooking the facial region-aware makeup features (i.e., eyes, mouth, etc) and limiting the regional controllability for region-specific makeup transfer. To address these, in this work, we propose Facial Region-Aware Makeup features (FRAM), which has two stages: (1) makeup CLIP fine-tuning; (2) identity and facial region-aware makeup injection. For makeup CLIP fine-tuning, unlike prior works using off-the-shelf CLIP, we synthesize annotated makeup style data using GPT-o3 and text-driven image editing model, and then use the data to train a makeup CLIP encoder through self-supervised and image-text contrastive learning. For identity and facial region-aware makeup injection, we construct before-and-after makeup image pairs from the edited images in stage 1 and then use them to learn to inject identity of source image and makeup of reference image to the diffusion denoising model for makeup transfer. Specifically, we use learnable tokens to query the makeup CLIP encoder to extract facial region-aware makeup features for makeup injection, which is learned via an attention loss to enable regional control. As for identity injection, we use a ControlNet Union to encode source image and its 3D mesh simultaneously. The experimental results verify the superiority of our regional controllability and our makeup transfer performance. |
| 2026-03-20 | [MedSPOT: A Workflow-Aware Sequential Grounding Benchmark for Clinical GUI](http://arxiv.org/abs/2603.19993v1) | Rozain Shakeel, Abdul Rahman Mohammad Ali et al. | Despite the rapid progress of Multimodal Large Language Models (MLLMs), their ability to perform reliable visual grounding in high-stakes clinical software environments remains underexplored. Existing GUI benchmarks largely focus on isolated, single-step grounding queries, overlooking the sequential, workflow-driven reasoning required in real-world medical interfaces, where tasks evolve across independent steps and dynamic interface states. We introduce MedSPOT, a workflow-aware sequential grounding benchmark for clinical GUI environments. Unlike prior benchmarks that treat grounding as a standalone prediction task, MedSPOT models procedural interaction as a sequence of structured spatial decisions. The benchmark comprises 216 task-driven videos with 597 annotated keyframes, in which each task consists of 2 to 3 interdependent grounding steps within realistic medical workflows. This design captures interface hierarchies, contextual dependencies, and fine-grained spatial precision under evolving conditions. To evaluate procedural robustness, we propose a strict sequential evaluation protocol that terminates task assessment upon the first incorrect grounding prediction, explicitly measuring error propagation in multi-step workflows. We further introduce a comprehensive failure taxonomy, including edge bias, small-target errors, no prediction, near miss, far miss, and toolbar confusion, to enable systematic diagnosis of model behavior in clinical GUI settings. By shifting evaluation from isolated grounding to workflow-aware sequential reasoning, MedSPOT establishes a realistic and safety-critical benchmark for assessing multimodal models in medical software environments. Code and data are available at: https://github.com/Tajamul21/MedSPOT. |
| 2026-03-20 | [Cavitation by phase shift of focused shock waves inside a droplet](http://arxiv.org/abs/2603.19990v1) | Samuele Fiorini, Guillaume T. Bokman et al. | Localized cavitation in liquids and soft tissues, typically initiated by the rarefaction phase of high-amplitude ultrasound waves, is leveraged in several biomedical applications such as ablation techniques and drug delivery with vaporizing agents. However, safety considerations aimed at avoiding unwanted bubble activity outside the targeted region pose a limit to the maximum allowed peak rarefaction pressure, which on the other hand can hinder the therapeutic efficacy of these techniques. This study shows that a purely compressive shock wave can generate localized, negative pressure and initiate cavitation inside a sub-millimetric perfluorohexane droplet, without requiring any externally applied rarefaction wave. The Gouy phase shift is identified as the physical mechanism responsible for the conversion of positive pressure into tension during shock focusing, and its occurrence is demonstrated through numerical simulations and direct experimental measurements. Comparison of the regions affected by cavitation, visualized \emph{in-situ} by means of high-speed x-ray phase-contrast imaging, with prediction from Classical Nucleation Theory suggests homogeneous nucleation as the underlying mechanism behind bubble formation. The presented findings offer valuable insights into the physics of shock wave propagation which can inspire the development of novel acoustic driving strategies for cavitation generation, facilitating the reduction of negative pressures outside the target region and improving the safety and precision of biomedical treatments. |
| 2026-03-20 | [Interpreting Reinforcement Learning Model Behavior via Koopman with Control](http://arxiv.org/abs/2603.19968v1) | William T. Redman | Reinforcement learning (RL) models have shown the capability of learning complex behaviors, but quantitatively assessing those behaviors - which is critical for safety assurance and the discovery of novel strategies - is challenging. By viewing RL models as control systems, we hypothesize that data-driven approximations of their associated Koopman operators may provide dynamical information about their behavior, thus enabling greater interpretability. To test this, we apply the Koopman with control framework to RL models trained on several standard benchmark environments and demonstrate that properties of the fit linear control models, such as stability and controllability, evolve during training in a task dependent manner. Comparing these metrics across different training epochs or across differently optimized RL models enables an understanding of how they differ. In addition, we find cases where - even when the reward achieved by the RL model is static - the stability and controllability is nonetheless evolving, predicting increased reward with further training. This suggests that these metrics may be able to serve as hidden progress measures, a core idea in mechanistic interpretability. Taken together, our results illustrate that the Koopman with control framework provides a comprehensive way in which to analyze and interpret the behavior of RL models, particularly across training. |
| 2026-03-20 | [HiPath: Hierarchical Vision-Language Alignment for Structured Pathology Report Prediction](http://arxiv.org/abs/2603.19957v1) | Ruicheng Yuan, Zhenxuan Zhang et al. | Pathology reports are structured, multi-granular documents encoding diagnostic conclusions, histological grades, and ancillary test results across one or more anatomical sites; yet existing pathology vision-language models (VLMs) reduce this output to a flat label or free-form text. We present HiPath, a lightweight VLM framework built on frozen UNI2 and Qwen3 backbones that treats structured report prediction as its primary training objective. Three trainable modules totalling 15M parameters address complementary aspects of the problem: a Hierarchical Patch Aggregator (HiPA) for multi-image visual encoding, Hierarchical Contrastive Learning (HiCL) for cross-modal alignment via optimal transport, and Slot-based Masked Diagnosis Prediction (Slot-MDP) for structured diagnosis generation. Trained on 749K real-world Chinese pathology cases from three hospitals, HiPath achieves 68.9% strict and 74.7% clinically acceptable accuracy with a 97.3% safety rate, outperforming all baselines under the same frozen backbone. Cross-hospital evaluation confirms generalisation with only a 3.4pp drop in strict accuracy while maintaining 97.1% safety. |
| 2026-03-20 | [Overreliance on AI in Information-seeking from Video Content](http://arxiv.org/abs/2603.19843v1) | Anders Giovanni Møller, Elisa Bassignana et al. | The ubiquity of multimedia content is reshaping online information spaces, particularly in social media environments. At the same time, search is being rapidly transformed by generative AI, with large language models (LLMs) routinely deployed as intermediaries between users and multimedia content to retrieve and summarize information. Despite their growing influence, the impact of LLM inaccuracies and potential vulnerabilities on multimedia information-seeking tasks remains largely unexplored. We investigate how generative AI affects accuracy, efficiency, and confidence in information retrieval from videos. We conduct an experiment with around 900 participants on 8,000+ video-based information-seeking tasks, comparing behavior across three conditions: (1) access to videos only, (2) access to videos with LLM-based AI assistance, and (3) access to videos with a deceiving AI assistant designed to provide false answers. We find that AI assistance increases accuracy by 3-7% when participants viewed the relevant video segment, and by 27-35% when they did not. Efficiency increases by 10% for short videos and 25% for longer ones. However, participants tend to over-rely on AI outputs, resulting in accuracy drops of up to 32% when interacting with the deceiving AI. Alarmingly, self-reported confidence in answers remains stable across all three conditions. Our findings expose fundamental safety risks in AI-mediated video information retrieval. |
| 2026-03-20 | [Multi-Agent Motion Planning on Industrial Magnetic Levitation Platforms: A Hybrid ADMM-HOCBF approach](http://arxiv.org/abs/2603.19838v1) | Bavo Tistaert, Stan Servaes et al. | This paper presents a novel hybrid motion planning method for holonomic multi-agent systems. The proposed decentralised model predictive control (MPC) framework tackles the intractability of classical centralised MPC for a growing number of agents while providing safety guarantees. This is achieved by combining a decentralised version of the alternating direction method of multipliers (ADMM) with a centralised high-order control barrier function (HOCBF) architecture. Simulation results show significant improvement in scalability over classical centralised MPC. We validate the efficacy and real-time capability of the proposed method by developing a highly efficient C++ implementation and deploying the resulting trajectories on a real industrial magnetic levitation platform. |
| 2026-03-20 | [FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization](http://arxiv.org/abs/2603.19835v1) | Chiyu Ma, Shuo Yang et al. | We present Future-KL Influenced Policy Optimization (FIPO), a reinforcement learning algorithm designed to overcome reasoning bottlenecks in large language models. While GRPO style training scales effectively, it typically relies on outcome-based rewards (ORM) that distribute a global advantage uniformly across every token in a trajectory. We argue that this coarse-grained credit assignment imposes a performance ceiling by failing to distinguish critical logical pivots from trivial tokens. FIPO addresses this by incorporating discounted future-KL divergence into the policy update, creating a dense advantage formulation that re-weights tokens based on their influence on subsequent trajectory behavior. Empirically, FIPO enables models to break through the length stagnation seen in standard baselines. Evaluated on Qwen2.5-32B, FIPO extends the average chain-of-thought length from roughly 4,000 to over 10,000 tokens and increases AIME 2024 Pass@1 accuracy from 50.0% to a peak of 58.0% (converging at approximately 56.0\%). This outperforms both DeepSeek-R1-Zero-Math-32B (around 47.0%) and o1-mini (approximately 56.0%). Our results suggest that establishing dense advantage formulations is a vital path for evolving ORM-based algorithms to unlock the full reasoning potential of base models. We open-source our training system, built on the verl framework. |
| 2026-03-20 | [HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks](http://arxiv.org/abs/2603.19822v1) | Jingyu Guo, Ziye Chen et al. | Existing UAV vision-language navigation (VLN) benchmarks have enabled language-guided flight, but they largely focus on long, step-wise route descriptions with goal-centric evaluation, making them less diagnostic for real operations where brief, high-level commands must be grounded into safe multi-stage behaviors. We present HUGE-Bench, a benchmark for High-Level UAV Vision-Language-Action (HL-VLA) tasks that tests whether an agent can interpret concise language and execute complex, process-oriented trajectories with safety awareness. HUGE-Bench comprises 4 real-world digital twin scenes, 8 high-level tasks, and 2.56M meters of trajectories, and is built on an aligned 3D Gaussian Splatting (3DGS)-Mesh representation that combines photorealistic rendering with collision-capable geometry for scalable generation and collision-aware evaluation. We introduce process-oriented and collision-aware metrics to assess process fidelity, terminal accuracy, and safety. Experiments on representative state-of-the-art VLA models reveal significant gaps in high-level semantic completion and safe execution, highlighting HUGE-Bench as a diagnostic testbed for high-level UAV autonomy. |
| 2026-03-20 | [A Spectral Perspective on Stochastic Control Barrier Functions](http://arxiv.org/abs/2603.19813v1) | Inkyu Jang, Chams E. Mballo et al. | Stochastic control barrier functions (SCBFs) provide a safety-critical control framework for systems subject to stochastic disturbances by bounding the probability of remaining within a safe set. However, synthesizing a valid SCBF that explicitly reflects the true safety probability of the system, which is the most natural measure of safety, remains a challenge. This paper addresses this issue by adopting a spectral perspective, utilizing the linear operator that governs the evolution of the closed-loop system's safety probability. We find that the dominant eigenpair of this Koopman-like operator encodes fundamental safety information of the stochastic system. The dominant eigenfunction is a natural and valid SCBF, with values that explicitly quantify the relative long-term safety of the state, while the dominant eigenvalue indicates the global rate at which the safety probability decays. A practical synthesis algorithm is proposed, termed power-policy iteration, which jointly computes the dominant eigenpair and an optimized backup policy. The method is validated using simulation experiments on safety-critical dynamics models. |
| 2026-03-20 | [Eye Gaze-Informed and Context-Aware Pedestrian Trajectory Prediction in Shared Spaces with Automated Shuttles: A Virtual Reality Study](http://arxiv.org/abs/2603.19812v1) | Danya Li, Yan Feng et al. | The integration of Automated Shuttles into shared urban spaces presents unique challenges due to the absence of traffic rules and the complex pedestrian interactions. Accurately anticipating pedestrian behavior in such unstructured environments is therefore critical for ensuring both safety and efficiency. This paper presents a Virtual Reality (VR) study that captures how pedestrians interact with automated shuttles across diverse scenarios, including varying approach angles and navigating in continuous traffic. We identify critical behavior patterns present in pedestrians' decision-making in shared spaces, including hesitation, evasive maneuvers, gaze allocation, and proxemic adjustments. To model pedestrian behavior, we propose GazeX-LSTM, a multimodal eye gaze-informed and context-aware prediction model that integrates pedestrians' trajectories, fine-grained eye gaze dynamics, and contextual factors. We shift prediction from a vehicle- to a human-centered perspective by leveraging eye-tracking data to capture pedestrian attention. We systematically validate the unique and irreplaceable predictive power of eye gaze over head orientation alone, further enhancing performance by integrating contextual variables. Notably, the combination of eye gaze data and contextual information produces super-additive improvements on pedestrian behavior prediction accuracy, revealing the complementary relationship between visual attention and situational contexts. Together, our findings provide the first evidence that eye gaze-informed modeling fundamentally advances pedestrian behavior prediction and highlight the critical role of situational contexts in shared-space interactions. This paves the way for safer and more adaptive automated vehicle technologies that account for how people perceive and act in complex shared spaces. |
| 2026-03-20 | [Ontology-Based Knowledge Modeling and Uncertainty-Aware Outdoor Air Quality Assessment Using Weighted Interval Type-2 Fuzzy Logic](http://arxiv.org/abs/2603.19683v1) | Md Inzmam, Ritesh Chandra et al. | Outdoor air pollution is a major concern for the environment and public health, especially in areas where urbanization is taking place rapidly. The Indian Air Quality Index (IND-AQI), developed by the Central Pollution Control Board (CPCB), is a standardized reporting system for air quality based on pollutants such as PM2.5, PM10), nitrogen dioxide (NO2), sulfur dioxide (SO2), ozone (O3), carbon monoxide (CO), and ammonia (NH3). However, the traditional calculation of the AQI uses crisp thresholds and deterministic aggregation rules, which are not suitable for handling uncertainty and transitions between classes. To address these limitations, this study proposes a hybrid ontology-based uncertainty-aware framework integrating Weighted Interval Type-2 Fuzzy Logic with semantic knowledge modeling. Interval Type-2 fuzzy sets are used to model uncertainty near AQI class boundaries, while pollutant importance weights are determined using Interval Type-2 Fuzzy Analytic Hierarchy Process (IT2-FAHP) to reflect their relative health impacts. In addition, an OWL-based air quality ontology extending the Semantic Sensor Network (SSN) ontology is developed to represent pollutants, monitoring stations, AQI categories, regulatory standards, and environmental governance actions. Semantic reasoning is implemented using SWRL rules and validated through SPARQL queries to infer AQI categories, health risks, and recommended mitigation actions. Experimental evaluation using CPCB air quality datasets demonstrates that the proposed framework improves AQI classification reliability and uncertainty handling compared with traditional crisp and Type-1 fuzzy approaches, while enabling explainable semantic reasoning and intelligent decision support for air quality monitoring systems |
| 2026-03-20 | [PowerLens: Taming LLM Agents for Safe and Personalized Mobile Power Management](http://arxiv.org/abs/2603.19584v1) | Xingyu Feng, Chang Sun et al. | Battery life remains a critical challenge for mobile devices, yet existing power management mechanisms rely on static rules or coarse-grained heuristics that ignore user activities and personal preferences. We present PowerLens, a system that tames the reasoning power of Large Language Models (LLMs) for safe and personalized mobile power management on Android devices. The key idea is that LLMs' commonsense reasoning can bridge the semantic gap between user activities and system parameters, enabling zero-shot, context-aware policy generation that adapts to individual preferences through implicit feedback. PowerLens employs a multi-agent architecture that recognizes user context from UI semantics and generates holistic power policies across 18 device parameters. A PDL-based constraint framework verifies every action before execution, while a two-tier memory system learns individualized preferences from implicit user overrides through confidence-based distillation, requiring no explicit configuration and converging within 3--5 days. Extensive experiments on a rooted Android device show that PowerLens achieves 81.7% action accuracy and 38.8% energy saving over stock Android, outperforming rule-based and LLM-based baselines, with high user satisfaction, fast preference convergence, and strong safety guarantees, with the system itself consuming only 0.5% of daily battery capacity. |
| 2026-03-20 | [AI Psychosis: Does Conversational AI Amplify Delusion-Related Language?](http://arxiv.org/abs/2603.19574v1) | Soorya Ram Shimgekar, Vipin Gunda et al. | Conversational AI systems are increasingly used for personal reflection and emotional disclosure, raising concerns about their effects on vulnerable users. Recent anecdotal reports suggest that prolonged interactions with AI may reinforce delusional thinking -- a phenomenon sometimes described as AI Psychosis. However, empirical evidence on this phenomenon remains limited. In this work, we examine how delusion-related language evolves during multi-turn interactions with conversational AI. We construct simulated users (SimUsers) from Reddit users' longitudinal posting histories and generate extended conversations with three model families (GPT, LLaMA, and Qwen). We develop DelusionScore, a linguistic measure that quantifies the intensity of delusion-related language across conversational turns. We find that SimUsers derived from users with prior delusion-related discourse (Treatment) exhibit progressively increasing DelusionScore trajectories, whereas those derived from users without such discourse (Control) remain stable or decline. We further find that this amplification varies across themes, with reality skepticism and compulsive reasoning showing the strongest increases. Finally, conditioning AI responses on current DelusionScore substantially reduces these trajectories. These findings provide empirical evidence that conversational AI interactions can amplify delusion-related language over extended use and highlight the importance of state-aware safety mechanisms for mitigating such risks. |

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



