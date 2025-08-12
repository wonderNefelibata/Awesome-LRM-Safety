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
| 2025-08-11 | [Jinx: Unlimited LLMs for Probing Alignment Failures](http://arxiv.org/abs/2508.08243v1) | Jiahao Zhao, Liwei Dong | Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community.   We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety. |
| 2025-08-11 | [Exploring Safety Alignment Evaluation of LLMs in Chinese Mental Health Dialogues via LLM-as-Judge](http://arxiv.org/abs/2508.08236v1) | Yunna Cai, Fan Wang et al. | Evaluating the safety alignment of LLM responses in high-risk mental health dialogues is particularly difficult due to missing gold-standard answers and the ethically sensitive nature of these interactions. To address this challenge, we propose PsyCrisis-Bench, a reference-free evaluation benchmark based on real-world Chinese mental health dialogues. It evaluates whether the model responses align with the safety principles defined by experts. Specifically designed for settings without standard references, our method adopts a prompt-based LLM-as-Judge approach that conducts in-context evaluation using expert-defined reasoning chains grounded in psychological intervention principles. We employ binary point-wise scoring across multiple safety dimensions to enhance the explainability and traceability of the evaluation. Additionally, we present a manually curated, high-quality Chinese-language dataset covering self-harm, suicidal ideation, and existential distress, derived from real-world online discourse. Experiments on 3600 judgments show that our method achieves the highest agreement with expert assessments and produces more interpretable evaluation rationales compared to existing approaches. Our dataset and evaluation tool are publicly available to facilitate further research. |
| 2025-08-11 | [Autonomous Air-Ground Vehicle Operations Optimization in Hazardous Environments: A Multi-Armed Bandit Approach](http://arxiv.org/abs/2508.08217v1) | Jimin Choi, Max Z. Li | Hazardous environments such as chemical spills, radiological zones, and bio-contaminated sites pose significant threats to human safety and public infrastructure. Rapid and reliable hazard mitigation in these settings often unsafe for humans, calling for autonomous systems that can adaptively sense and respond to evolving risks. This paper presents a decision-making framework for autonomous vehicle dispatch in hazardous environments with uncertain and evolving risk levels. The system integrates a Bayesian Upper Confidence Bound (BUCB) sensing strategy with task-specific vehicle routing problems with profits (VRPP), enabling adaptive coordination of unmanned aerial vehicles (UAVs) for hazard sensing and unmanned ground vehicles (UGVs) for cleaning. Using VRPP allows selective site visits under resource constraints by assigning each site a visit value that reflects sensing or cleaning priorities. Site-level hazard beliefs are maintained through a time-weighted Bayesian update. BUCB scores guide UAV routing to balance exploration and exploitation under uncertainty, while UGV routes are optimized to maximize expected hazard reduction under resource constraints. Simulation results demonstrate that our framework reduces the number of dispatch cycles to resolve hazards by around 30% on average compared to baseline dispatch strategies, underscoring the value of uncertainty-aware vehicle dispatch for reliable hazard mitigation. |
| 2025-08-11 | [Robust Adaptive Discrete-Time Control Barrier Certificate](http://arxiv.org/abs/2508.08153v1) | Changrui Liu, Anil Alan et al. | This work develops a robust adaptive control strategy for discrete-time systems using Control Barrier Functions (CBFs) to ensure safety under parametric model uncertainty and disturbances. A key contribution of this work is establishing a barrier function certificate in discrete time for general online parameter estimation algorithms. This barrier function certificate guarantees positive invariance of the safe set despite disturbances and parametric uncertainty without access to the true system parameters. In addition, real-time implementation and inherent robustness guarantees are provided. Our approach demonstrates that, using the proposed robust adaptive CBF framework, the parameter estimation module can be designed separately from the CBF-based safety filter, simplifying the development of safe adaptive controllers for discrete-time systems. The resulting safety filter guarantees that the system remains within the safe set while adapting to model uncertainties, making it a promising strategy for real-world applications involving discrete-time safety-critical systems. |
| 2025-08-11 | [Capsizing-Guided Trajectory Optimization for Autonomous Navigation with Rough Terrain](http://arxiv.org/abs/2508.08108v1) | Wei Zhang, Yinchuan Wang et al. | It is a challenging task for ground robots to autonomously navigate in harsh environments due to the presence of non-trivial obstacles and uneven terrain. This requires trajectory planning that balances safety and efficiency. The primary challenge is to generate a feasible trajectory that prevents robot from tip-over while ensuring effective navigation. In this paper, we propose a capsizing-aware trajectory planner (CAP) to achieve trajectory planning on the uneven terrain. The tip-over stability of the robot on rough terrain is analyzed. Based on the tip-over stability, we define the traversable orientation, which indicates the safe range of robot orientations. This orientation is then incorporated into a capsizing-safety constraint for trajectory optimization. We employ a graph-based solver to compute a robust and feasible trajectory while adhering to the capsizing-safety constraint. Extensive simulation and real-world experiments validate the effectiveness and robustness of the proposed method. The results demonstrate that CAP outperforms existing state-of-the-art approaches, providing enhanced navigation performance on uneven terrains. |
| 2025-08-11 | [ChatGPT on the Road: Leveraging Large Language Model-Powered In-vehicle Conversational Agents for Safer and More Enjoyable Driving Experience](http://arxiv.org/abs/2508.08101v1) | Yeana Lee Bond, Mungyeong Choe et al. | Studies on in-vehicle conversational agents have traditionally relied on pre-scripted prompts or limited voice commands, constraining natural driver-agent interaction. To resolve this issue, the present study explored the potential of a ChatGPT-based in-vehicle agent capable of carrying continuous, multi-turn dialogues. Forty drivers participated in our experiment using a motion-based driving simulator, comparing three conditions (No agent, Pre-scripted agent, and ChatGPT-based agent) as a within-subjects variable. Results showed that the ChatGPT-based agent condition led to more stable driving performance across multiple metrics. Participants demonstrated lower variability in longitudinal acceleration, lateral acceleration, and lane deviation compared to the other two conditions. In subjective evaluations, the ChatGPT-based agent also received significantly higher ratings in competence, animacy, affective trust, and preference compared to the Pre-scripted agent. Our thematic analysis of driver-agent conversations revealed diverse interaction patterns in topics, including driving assistance/questions, entertainment requests, and anthropomorphic interactions. Our results highlight the potential of LLM-powered in-vehicle conversational agents to enhance driving safety and user experience through natural, context-rich interactions. |
| 2025-08-11 | [Symbolic Quantile Regression for the Interpretable Prediction of Conditional Quantiles](http://arxiv.org/abs/2508.08080v1) | Cas Oude Hoekstra, Floris den Hengst | Symbolic Regression (SR) is a well-established framework for generating interpretable or white-box predictive models. Although SR has been successfully applied to create interpretable estimates of the average of the outcome, it is currently not well understood how it can be used to estimate the relationship between variables at other points in the distribution of the target variable. Such estimates of e.g. the median or an extreme value provide a fuller picture of how predictive variables affect the outcome and are necessary in high-stakes, safety-critical application domains. This study introduces Symbolic Quantile Regression (SQR), an approach to predict conditional quantiles with SR. In an extensive evaluation, we find that SQR outperforms transparent models and performs comparably to a strong black-box baseline without compromising transparency. We also show how SQR can be used to explain differences in the target distribution by comparing models that predict extreme and central outcomes in an airline fuel usage case study. We conclude that SQR is suitable for predicting conditional quantiles and understanding interesting feature influences at varying quantiles. |
| 2025-08-11 | [On the Operational Resilience of CBDC: Threats and Prospects of Formal Validation for Offline Payments](http://arxiv.org/abs/2508.08064v1) | Marco Bernardo, Federico Calandra et al. | Information and communication technologies are by now employed in most activities, including economics and finance. Despite the extraordinary power of modern computers and the vast amount of memory, some results of theoretical computer science imply the impossibility of certifying software quality in general. With the exception of safety-critical systems, this has primarily concerned the information processed by confined systems, with limited socio-economic consequences. In the emerging era of technologies for exchanging digital money and tokenized assets over the Internet - such as central bank digital currencies (CBDCs) - even a minor bug could trigger a financial collapse. Although the aforementioned impossibility results cannot be overcome in an absolute sense, there exist formal methods that can provide assertions of computing systems correctness. We advocate their use to validate the operational resilience of software infrastructures enabling CBDCs, with special emphasis on offline payments as they constitute a very critical issue. |
| 2025-08-11 | [The Escalator Problem: Identifying Implicit Motion Blindness in AI for Accessibility](http://arxiv.org/abs/2508.07989v1) | Xiantao Zhang | Multimodal Large Language Models (MLLMs) hold immense promise as assistive technologies for the blind and visually impaired (BVI) community. However, we identify a critical failure mode that undermines their trustworthiness in real-world applications. We introduce the Escalator Problem -- the inability of state-of-the-art models to perceive an escalator's direction of travel -- as a canonical example of a deeper limitation we term Implicit Motion Blindness. This blindness stems from the dominant frame-sampling paradigm in video understanding, which, by treating videos as discrete sequences of static images, fundamentally struggles to perceive continuous, low-signal motion. As a position paper, our contribution is not a new model but rather to: (I) formally articulate this blind spot, (II) analyze its implications for user trust, and (III) issue a call to action. We advocate for a paradigm shift from purely semantic recognition towards robust physical perception and urge the development of new, human-centered benchmarks that prioritize safety, reliability, and the genuine needs of users in dynamic environments. |
| 2025-08-11 | [TrackOR: Towards Personalized Intelligent Operating Rooms Through Robust Tracking](http://arxiv.org/abs/2508.07968v1) | Tony Danjun Wang, Christian Heiliger et al. | Providing intelligent support to surgical teams is a key frontier in automated surgical scene understanding, with the long-term goal of improving patient outcomes. Developing personalized intelligence for all staff members requires maintaining a consistent state of who is located where for long surgical procedures, which still poses numerous computational challenges. We propose TrackOR, a framework for tackling long-term multi-person tracking and re-identification in the operating room. TrackOR uses 3D geometric signatures to achieve state-of-the-art online tracking performance (+11% Association Accuracy over the strongest baseline), while also enabling an effective offline recovery process to create analysis-ready trajectories. Our work shows that by leveraging 3D geometric information, persistent identity tracking becomes attainable, enabling a critical shift towards the more granular, staff-centric analyses required for personalized intelligent systems in the operating room. This new capability opens up various applications, including our proposed temporal pathway imprints that translate raw tracking data into actionable insights for improving team efficiency and safety and ultimately providing personalized support. |
| 2025-08-11 | [Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning](http://arxiv.org/abs/2508.07885v1) | Shoaib Ahmmad, Zubayer Ahmed Aditto et al. | This paper introduces an advanced AI-driven perception system for autonomous quadcopter navigation in GPS-denied indoor environments. The proposed framework leverages cloud computing to offload computationally intensive tasks and incorporates a custom-designed printed circuit board (PCB) for efficient sensor data acquisition, enabling robust navigation in confined spaces. The system integrates YOLOv11 for object detection, Depth Anything V2 for monocular depth estimation, a PCB equipped with Time-of-Flight (ToF) sensors and an Inertial Measurement Unit (IMU), and a cloud-based Large Language Model (LLM) for context-aware decision-making. A virtual safety envelope, enforced by calibrated sensor offsets, ensures collision avoidance, while a multithreaded architecture achieves low-latency processing. Enhanced spatial awareness is facilitated by 3D bounding box estimation with Kalman filtering. Experimental results in an indoor testbed demonstrate strong performance, with object detection achieving a mean Average Precision (mAP50) of 0.6, depth estimation Mean Absolute Error (MAE) of 7.2 cm, only 16 safety envelope breaches across 42 trials over approximately 11 minutes, and end-to-end system latency below 1 second. This cloud-supported, high-intelligence framework serves as an auxiliary perception and navigation system, complementing state-of-the-art drone autonomy for GPS-denied confined spaces. |
| 2025-08-11 | [Multi-agent systems for chemical engineering: A review and perspective](http://arxiv.org/abs/2508.07880v1) | Sophia Rupprecht, Qinghe Gao et al. | Large language model (LLM)-based multi-agent systems (MASs) are a recent but rapidly evolving technology with the potential to transform chemical engineering by decomposing complex workflows into teams of collaborative agents with specialized knowledge and tools. This review surveys the state-of-the-art of MAS within chemical engineering. While early studies demonstrate promising results, scientific challenges remain, including the design of tailored architectures, integration of heterogeneous data modalities, development of foundation models with domain-specific modalities, and strategies for ensuring transparency, safety, and environmental impact. As a young but fast-moving field, MASs offer exciting opportunities to rethink chemical engineering workflows. |
| 2025-08-11 | [Tracking Any Point Methods for Markerless 3D Tissue Tracking in Endoscopic Stereo Images](http://arxiv.org/abs/2508.07851v1) | Konrad Reuter, Suresh Guttikonda et al. | Minimally invasive surgery presents challenges such as dynamic tissue motion and a limited field of view. Accurate tissue tracking has the potential to support surgical guidance, improve safety by helping avoid damage to sensitive structures, and enable context-aware robotic assistance during complex procedures. In this work, we propose a novel method for markerless 3D tissue tracking by leveraging 2D Tracking Any Point (TAP) networks. Our method combines two CoTracker models, one for temporal tracking and one for stereo matching, to estimate 3D motion from stereo endoscopic images. We evaluate the system using a clinical laparoscopic setup and a robotic arm simulating tissue motion, with experiments conducted on a synthetic 3D-printed phantom and a chicken tissue phantom. Tracking on the chicken tissue phantom yielded more reliable results, with Euclidean distance errors as low as 1.1 mm at a velocity of 10 mm/s. These findings highlight the potential of TAP-based models for accurate, markerless 3D tracking in challenging surgical scenarios. |
| 2025-08-11 | [Evaluating Large Language Models as Expert Annotators](http://arxiv.org/abs/2508.07827v1) | Yu-Min Tseng, Wei-Lin Chen et al. | Textual data annotation, the process of labeling or tagging text with relevant information, is typically costly, time-consuming, and labor-intensive. While large language models (LLMs) have demonstrated their potential as direct alternatives to human annotators for general domains natural language processing (NLP) tasks, their effectiveness on annotation tasks in domains requiring expert knowledge remains underexplored. In this paper, we investigate: whether top-performing LLMs, which might be perceived as having expert-level proficiency in academic and professional benchmarks, can serve as direct alternatives to human expert annotators? To this end, we evaluate both individual LLMs and multi-agent approaches across three highly specialized domains: finance, biomedicine, and law. Specifically, we propose a multi-agent discussion framework to simulate a group of human annotators, where LLMs are tasked to engage in discussions by considering others' annotations and justifications before finalizing their labels. Additionally, we incorporate reasoning models (e.g., o3-mini) to enable a more comprehensive comparison. Our empirical results reveal that: (1) Individual LLMs equipped with inference-time techniques (e.g., chain-of-thought (CoT), self-consistency) show only marginal or even negative performance gains, contrary to prior literature suggesting their broad effectiveness. (2) Overall, reasoning models do not demonstrate statistically significant improvements over non-reasoning models in most settings. This suggests that extended long CoT provides relatively limited benefits for data annotation in specialized domains. (3) Certain model behaviors emerge in the multi-agent discussion environment. For instance, Claude 3.7 Sonnet with thinking rarely changes its initial annotations, even when other agents provide correct annotations or valid reasoning. |
| 2025-08-11 | [Power Battery Detection](http://arxiv.org/abs/2508.07797v1) | Xiaoqi Zhao, Peiqian Cao et al. | Power batteries are essential components in electric vehicles, where internal structural defects can pose serious safety risks. We conduct a comprehensive study on a new task, power battery detection (PBD), which aims to localize the dense endpoints of cathode and anode plates from industrial X-ray images for quality inspection. Manual inspection is inefficient and error-prone, while traditional vision algorithms struggle with densely packed plates, low contrast, scale variation, and imaging artifacts. To address this issue and drive more attention into this meaningful task, we present PBD5K, the first large-scale benchmark for this task, consisting of 5,000 X-ray images from nine battery types with fine-grained annotations and eight types of real-world visual interference. To support scalable and consistent labeling, we develop an intelligent annotation pipeline that combines image filtering, model-assisted pre-labeling, cross-verification, and layered quality evaluation. We formulate PBD as a point-level segmentation problem and propose MDCNeXt, a model designed to extract and integrate multi-dimensional structure clues including point, line, and count information from the plate itself. To improve discrimination between plates and suppress visual interference, MDCNeXt incorporates two state space modules. The first is a prompt-filtered module that learns contrastive relationships guided by task-specific prompts. The second is a density-aware reordering module that refines segmentation in regions with high plate density. In addition, we propose a distance-adaptive mask generation strategy to provide robust supervision under varying spatial distributions of anode and cathode positions. The source code and datasets will be publicly available at \href{https://github.com/Xiaoqi-Zhao-DLUT/X-ray-PBD}{PBD5K}. |
| 2025-08-11 | [When are safety filters safe? On minimum phase conditions of control barrier functions](http://arxiv.org/abs/2508.07684v1) | Jason J. Choi, Claire J. Tomlin et al. | In emerging control applications involving multiple and complex tasks, safety filters are gaining prominence as a modular approach to enforcing safety constraints. Among various methods, control barrier functions (CBFs) are widely used for designing safety filters due to their simplicity, imposing a single linear constraint on the control input at each state. In this work, we focus on the internal dynamics of systems governed by CBF-constrained control laws. Our key observation is that, although CBFs guarantee safety by enforcing state constraints, they can inadvertently be "unsafe" by causing the internal state to diverge. We investigate the conditions under which the full system state, including the internal state, can remain bounded under a CBF-based safety filter. Drawing inspiration from the input-output linearization literature, where boundedness is ensured by minimum phase conditions, we propose a new set of CBF minimum phase conditions tailored to the structure imposed by the CBF constraint. A critical distinction from the original minimum phase conditions is that the internal dynamics in our setting is driven by a nonnegative virtual control input, which reflects the enforcement of the safety constraint. We include a range of numerical examples, including single-input, multi-input, linear, and nonlinear systems, validating both our analysis and the necessity of the proposed CBF minimum phase conditions. |
| 2025-08-11 | [Remote ID Based UAV Collision Avoidance Optimization for Low-Altitude Airspace Safety](http://arxiv.org/abs/2508.07651v1) | Ziye Jia, Yian Zhu et al. | With the rapid development of unmanned aerial vehicles (UAVs), it is paramount to ensure safe and efficient operations in open airspaces. The remote identification (Remote ID) is deemed an effective real-time UAV monitoring system by the federal aviation administration, which holds potentials for enabling inter-UAV communications. This paper deeply investigates the application of Remote ID for UAV collision avoidance while minimizing communication delays. First, we propose a Remote ID based distributed multi-UAV collision avoidance (DMUCA) framework to support the collision detection, avoidance decision-making, and trajectory recovery. Next, the average transmission delays for Remote ID messages are analyzed, incorporating the packet reception mechanisms and packet loss due to interference. The optimization problem is formulated to minimize the long-term average communication delay, where UAVs can flexibly select the Remote ID protocol to enhance the collision avoidance performance. To tackle the problem, we design a multi-agent deep Q-network based adaptive communication configuration algorithm, allowing UAVs to autonomously learn the optimal protocol configurations in dynamic environments. Finally, numerical results verify the feasibility of the proposed DMUCA framework, and the proposed mechanism can reduce the average delay by 32% compared to the fixed protocol configuration. |
| 2025-08-11 | [Multi-Turn Jailbreaks Are Simpler Than They Seem](http://arxiv.org/abs/2508.07646v1) | Xiaoxue Yang, Jaeha Lee et al. | While defenses against single-turn jailbreak attacks on Large Language Models (LLMs) have improved significantly, multi-turn jailbreaks remain a persistent vulnerability, often achieving success rates exceeding 70% against models optimized for single-turn protection. This work presents an empirical analysis of automated multi-turn jailbreak attacks across state-of-the-art models including GPT-4, Claude, and Gemini variants, using the StrongREJECT benchmark. Our findings challenge the perceived sophistication of multi-turn attacks: when accounting for the attacker's ability to learn from how models refuse harmful requests, multi-turn jailbreaking approaches are approximately equivalent to simply resampling single-turn attacks multiple times. Moreover, attack success is correlated among similar models, making it easier to jailbreak newly released ones. Additionally, for reasoning models, we find surprisingly that higher reasoning effort often leads to higher attack success rates. Our results have important implications for AI safety evaluation and the design of jailbreak-resistant systems. We release the source code at https://github.com/diogo-cruz/multi_turn_simpler |
| 2025-08-11 | [End-to-End Humanoid Robot Safe and Comfortable Locomotion Policy](http://arxiv.org/abs/2508.07611v1) | Zifan Wang, Xun Yang et al. | The deployment of humanoid robots in unstructured, human-centric environments requires navigation capabilities that extend beyond simple locomotion to include robust perception, provable safety, and socially aware behavior. Current reinforcement learning approaches are often limited by blind controllers that lack environmental awareness or by vision-based systems that fail to perceive complex 3D obstacles. In this work, we present an end-to-end locomotion policy that directly maps raw, spatio-temporal LiDAR point clouds to motor commands, enabling robust navigation in cluttered dynamic scenes. We formulate the control problem as a Constrained Markov Decision Process (CMDP) to formally separate safety from task objectives. Our key contribution is a novel methodology that translates the principles of Control Barrier Functions (CBFs) into costs within the CMDP, allowing a model-free Penalized Proximal Policy Optimization (P3O) to enforce safety constraints during training. Furthermore, we introduce a set of comfort-oriented rewards, grounded in human-robot interaction research, to promote motions that are smooth, predictable, and less intrusive. We demonstrate the efficacy of our framework through a successful sim-to-real transfer to a physical humanoid robot, which exhibits agile and safe navigation around both static and dynamic 3D obstacles. |
| 2025-08-11 | [Progressive Bird's Eye View Perception for Safety-Critical Autonomous Driving: A Comprehensive Survey](http://arxiv.org/abs/2508.07560v1) | Yan Gong, Naibang Wang et al. | Bird's-Eye-View (BEV) perception has become a foundational paradigm in autonomous driving, enabling unified spatial representations that support robust multi-sensor fusion and multi-agent collaboration. As autonomous vehicles transition from controlled environments to real-world deployment, ensuring the safety and reliability of BEV perception in complex scenarios - such as occlusions, adverse weather, and dynamic traffic - remains a critical challenge. This survey provides the first comprehensive review of BEV perception from a safety-critical perspective, systematically analyzing state-of-the-art frameworks and implementation strategies across three progressive stages: single-modality vehicle-side, multimodal vehicle-side, and multi-agent collaborative perception. Furthermore, we examine public datasets encompassing vehicle-side, roadside, and collaborative settings, evaluating their relevance to safety and robustness. We also identify key open-world challenges - including open-set recognition, large-scale unlabeled data, sensor degradation, and inter-agent communication latency - and outline future research directions, such as integration with end-to-end autonomous driving systems, embodied intelligence, and large language models. |

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



