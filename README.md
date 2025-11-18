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
| 2025-11-17 | [A Real-Time Driver Drowsiness Detection System Using MediaPipe and Eye Aspect Ratio](http://arxiv.org/abs/2511.13618v1) | Ashlesha G. Sawant, Shreyash S. Kamble et al. | One of the major causes of road accidents is driver fatigue that causes thousands of fatalities and injuries every year. This study shows development of a Driver Drowsiness Detection System meant to improve the safety of the road by alerting drivers who are showing signs of being drowsy. The system is based on a standard webcam that tracks the facial features of the driver with the main emphasis on the examination of eye movements that can be conducted with the help of the Eye Aspect Ratio (EAR) method. The Face Mesh by MediaPipe is a lightweight framework that can identify facial landmarks with high accuracy and efficiency, which is considered to be important in real time use. The system detects the moments of long eye shutdowns or a very low rate of blinking which are manifestations of drowsiness and alerts the driver through sound to get her attention back. This system achieves a high-performance and low-cost driver monitoring solution with the help of the computational power of OpenCV to process the image and the MediaPipe to identify faces. Test data experimental analyses indicate that the system is very accurate and responds quicker; this confirms that it can be a component of the current Advanced Driving Assistance System (ADAS). |
| 2025-11-17 | [Exploring the Effectiveness of Google Play Store's Privacy Transparency Channels](http://arxiv.org/abs/2511.13576v1) | Anhao Xiang, Weiping Pei et al. | With the requirements and emphases on privacy transparency placed by regulations such as GDPR and CCPA, the Google Play Store requires Android developers to more responsibly communicate their apps' privacy practices to potential users by providing the proper information via the data safety, privacy policy, and permission manifest privacy transparency channels. However, it is unclear how effective those channels are in helping users make informed decisions in the app selection and installation process. In this article, we conducted a study for 190 participants to interact with our simulated privacy transparency channels of mobile apps. We quantitatively analyzed (supplemented by qualitative analysis) participants' responses to five sets of questions. We found that data safety provides the most intuitive user interfaces, privacy policy is most informative and effective, while permission manifest excels at raising participants' concerns about an app's overall privacy risks. These channels complement each other and should all be improved. |
| 2025-11-17 | [Asymptotic analysis of rare events in high dimensions](http://arxiv.org/abs/2511.13559v1) | Anya Katsevich, Alexander Katsevich | Understanding rare events is critical across domains ranging from signal processing to reliability and structural safety, extreme-weather forecasting, and insurance. The analysis of rare events is a computationally challenging problem, particularly in high dimensions $d$. In this work, we develop the first asymptotic high-dimensional theory of rare events. First, we exploit asymptotic integral methods recently developed by the first author to provide an asymptotic expansion of rare event probabilities. The expansion employs the geometry of the rare event boundary and the local behavior of the log probability density. Generically, the expansion is valid if $d^2\llλ$, where $λ$ characterizes the extremity of the event. We prove this condition is necessary by constructing an example in which the first-order remainder is bounded above and below by $d^2/λ$. We also provide a nonasymptotic remainder bound which specifies the precise dependence of the remainder on $d$, $λ$, the density, and the boundary, and which shows that in certain cases, the condition $d^2\ll λ$ can be relaxed. As an application of the theory, we derive asymptotic approximations to rare probabilities under the standard Gaussian density in high dimensions. In the second part of our work, we provide an asymptotic approximation to densities conditional on rare events. This gives rise to simple procedure for approximately sampling conditionally on the rare event using independent Gaussian and exponential random variables. |
| 2025-11-17 | [ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models](http://arxiv.org/abs/2511.13548v1) | Siyang Cheng, Gaotian Liu et al. | The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions. |
| 2025-11-17 | [BootOOD: Self-Supervised Out-of-Distribution Detection via Synthetic Sample Exposure under Neural Collapse](http://arxiv.org/abs/2511.13539v1) | Yuanchao Wang, Tian Qin et al. | Out-of-distribution (OOD) detection is critical for deploying image classifiers in safety-sensitive environments, yet existing detectors often struggle when OOD samples are semantically similar to the in-distribution (ID) classes. We present BootOOD, a fully self-supervised OOD detection framework that bootstraps exclusively from ID data and is explicitly designed to handle semantically challenging OOD samples. BootOOD synthesizes pseudo-OOD features through simple transformations of ID representations and leverages Neural Collapse (NC), where ID features cluster tightly around class means with consistent feature norms. Unlike prior approaches that aim to constrain OOD features into subspaces orthogonal to the collapsed ID means, BootOOD introduces a lightweight auxiliary head that performs radius-based classification on feature norms. This design decouples OOD detection from the primary classifier and imposes a relaxed requirement: OOD samples are learned to have smaller feature norms than ID features, which is easier to satisfy when ID and OOD are semantically close. Experiments on CIFAR-10, CIFAR-100, and ImageNet-200 show that BootOOD outperforms prior post-hoc methods, surpasses training-based methods without outlier exposure, and is competitive with state-of-the-art outlier-exposure approaches while maintaining or improving ID accuracy. |
| 2025-11-17 | [Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew](http://arxiv.org/abs/2511.13535v1) | Farhin Farhad Riya, Shahinul Hoque et al. | As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets. |
| 2025-11-17 | [Minimax Multi-Target Conformal Prediction with Applications to Imaging Inverse Problems](http://arxiv.org/abs/2511.13533v1) | Jeffrey Wen, Rizwan Ahmad et al. | In ill-posed imaging inverse problems, uncertainty quantification remains a fundamental challenge, especially in safety-critical applications. Recently, conformal prediction has been used to quantify the uncertainty that the inverse problem contributes to downstream tasks like image classification, image quality assessment, fat mass quantification, etc. While existing works handle only a scalar estimation target, practical applications often involve multiple targets. In response, we propose an asymptotically minimax approach to multi-target conformal prediction that provides tight prediction intervals while ensuring joint marginal coverage. We then outline how our minimax approach can be applied to multi-metric blind image quality assessment, multi-task uncertainty quantification, and multi-round measurement acquisition. Finally, we numerically demonstrate the benefits of our minimax method, relative to existing multi-target conformal prediction methods, using both synthetic and magnetic resonance imaging (MRI) data. |
| 2025-11-17 | [Uniform Feasibility For Smoothed Backup Control Barrier Functions](http://arxiv.org/abs/2511.13499v1) | Anil Alan, Bart De Schutter | We study feasibility guarantees for safety filters developed using Control Barrier Functions (CBFs) when a safe set is defined using the pointwise minimum of continuously differentiable functions, a construction that is common for the backup CBF method and typically nonsmooth. We replace the minimum by its log-sum-exp (soft-min) smoothing and show that, under a strict safety condition, the smooth function becomes a CBF (or extended CBF) for a range of the smoothing parameter. For compact safe sets, we derive an explicit lower bound on the smoothing parameter that makes the smooth function a CBF and hence renders the corresponding safety filter feasible. For unbounded sets, we introduce tail conditions under which the smooth function satisfies an extended CBF condition uniformly. Finally, we apply these results to backup CBFs. We show that safety of a compact (terminal) backup set under a backup controller, together with a condition ensuring safety of the backup trajectories on the relevant boundary of the safe set, is sufficient for feasibility for backup CBFs. These results provide a recipe for a priori feasibility guarantees for smooth inner approximations of nonsmooth safe sets without the need for additional online certification. |
| 2025-11-17 | [Contact-Safe Reinforcement Learning with ProMP Reparameterization and Energy Awareness](http://arxiv.org/abs/2511.13459v1) | Bingkun Huang, Yuhe Gong et al. | Reinforcement learning (RL) approaches based on Markov Decision Processes (MDPs) are predominantly applied in the robot joint space, often relying on limited task-specific information and partial awareness of the 3D environment. In contrast, episodic RL has demonstrated advantages over traditional MDP-based methods in terms of trajectory consistency, task awareness, and overall performance in complex robotic tasks. Moreover, traditional step-wise and episodic RL methods often neglect the contact-rich information inherent in task-space manipulation, especially considering the contact-safety and robustness. In this work, contact-rich manipulation tasks are tackled using a task-space, energy-safe framework, where reliable and safe task-space trajectories are generated through the combination of Proximal Policy Optimization (PPO) and movement primitives. Furthermore, an energy-aware Cartesian Impedance Controller objective is incorporated within the proposed framework to ensure safe interactions between the robot and the environment. Our experimental results demonstrate that the proposed framework outperforms existing methods in handling tasks on various types of surfaces in 3D environments, achieving high success rates as well as smooth trajectories and energy-safe interactions. |
| 2025-11-17 | [PAST: A Primary-Auxiliary Spatio-Temporal Network for Traffic Time Series Imputation](http://arxiv.org/abs/2511.13414v1) | Hanwen Hu, Zimo Wen et al. | Traffic time series imputation is crucial for the safety and reliability of intelligent transportation systems, while diverse types of missing data, including random, fiber, and block missing make the imputation task challenging. Existing models often focus on disentangling and separately modeling spatial and temporal patterns based on relationships between data points. However, these approaches struggle to adapt to the random missing positions, and fail to learn long-term and large-scale dependencies, which are essential in extensive missing conditions. In this paper, patterns are categorized into two types to handle various missing data conditions: primary patterns, which originate from internal relationships between data points, and auxiliary patterns, influenced by external factors like timestamps and node attributes. Accordingly, we propose the Primary-Auxiliary Spatio-Temporal network (PAST). It comprises a graph-integrated module (GIM) and a cross-gated module (CGM). GIM captures primary patterns via dynamic graphs with interval-aware dropout and multi-order convolutions, and CGM extracts auxiliary patterns through bidirectional gating on embedded external features. The two modules interact via shared hidden vectors and are trained under an ensemble self-supervised framework. Experiments on three datasets under 27 missing data conditions demonstrate that the imputation accuracy of PAST outperforms seven state-of-the-art baselines by up to 26.2% in RMSE and 31.6% in MAE. |
| 2025-11-17 | [Microwave-acoustic-driven power electronics](http://arxiv.org/abs/2511.13412v1) | Liyang Jin, Zichen Xi et al. | Electrical isolation is critical to ensure safety and minimize electromagnetic interference (EMI), yet existing methods struggle to simultaneously transmit power and signals through a unified channel. Here we demonstrate a mechanically-isolated gate driver based on microwave-frequency surface acoustic wave (SAW) device on lithium niobate that achieves galvanic isolation of 2.75 kV with ultralow isolation capacitance (0.032 pF) over 1.25 mm mechanical propagation length, delivering 13.4 V open-circuit voltage and 44.4 mA short-circuit current. We demonstrate isolated gate driving for a gallium nitride (GaN) high-electron-mobility transistor, achieving a turn-on time of 108.8 ns comparable to commercial drivers and validate its operation in a buck converter. In addition, our SAW device operates over an ultrawide temperature range from 0.5 K (-272.6 °C) to 544 K (271 °C). The microwave-frequency SAW devices offer inherent EMI immunity and potential for heterogeneous integration on multiple semiconductor platforms, enabling compact, high-performance isolated power and signal transmission in advanced power electronics. |
| 2025-11-17 | [Descriptor: Distance-Annotated Traffic Perception Question Answering (DTPQA)](http://arxiv.org/abs/2511.13397v1) | Nikos Theodoridis, Tim Brophy et al. | The remarkable progress of Vision-Language Models (VLMs) on a variety of tasks has raised interest in their application to automated driving. However, for these models to be trusted in such a safety-critical domain, they must first possess robust perception capabilities, i.e., they must be capable of understanding a traffic scene, which can often be highly complex, with many things happening simultaneously. Moreover, since critical objects and agents in traffic scenes are often at long distances, we require systems with not only strong perception capabilities at close distances (up to 20 meters), but also at long (30+ meters) range. Therefore, it is important to evaluate the perception capabilities of these models in isolation from other skills like reasoning or advanced world knowledge. Distance-Annotated Traffic Perception Question Answering (DTPQA) is a Visual Question Answering (VQA) benchmark designed specifically for this purpose: it can be used to evaluate the perception systems of VLMs in traffic scenarios using trivial yet crucial questions relevant to driving decisions. It consists of two parts: a synthetic benchmark (DTP-Synthetic) created using a simulator, and a real-world benchmark (DTP-Real) built on top of existing images of real traffic scenes. Additionally, DTPQA includes distance annotations, i.e., how far the object in question is from the camera. More specifically, each DTPQA sample consists of (at least): (a) an image, (b) a question, (c) the ground truth answer, and (d) the distance of the object in question, enabling analysis of how VLM performance degrades with increasing object distance. In this article, we provide the dataset itself along with the Python scripts used to create it, which can be used to generate additional data of the same kind. |
| 2025-11-17 | [Can Large Language Models Function as Qualified Pediatricians? A Systematic Evaluation in Real-World Clinical Contexts](http://arxiv.org/abs/2511.13381v1) | Siyu Zhu, Mouxiao Bian et al. | With the rapid rise of large language models (LLMs) in medicine, a key question is whether they can function as competent pediatricians in real-world clinical settings. We developed PEDIASBench, a systematic evaluation framework centered on a knowledge-system framework and tailored to realistic clinical environments. PEDIASBench assesses LLMs across three dimensions: application of basic knowledge, dynamic diagnosis and treatment capability, and pediatric medical safety and medical ethics. We evaluated 12 representative models released over the past two years, including GPT-4o, Qwen3-235B-A22B, and DeepSeek-V3, covering 19 pediatric subspecialties and 211 prototypical diseases. State-of-the-art models performed well on foundational knowledge, with Qwen3-235B-A22B achieving over 90% accuracy on licensing-level questions, but performance declined ~15% as task complexity increased, revealing limitations in complex reasoning. Multiple-choice assessments highlighted weaknesses in integrative reasoning and knowledge recall. In dynamic diagnosis and treatment scenarios, DeepSeek-R1 scored highest in case reasoning (mean 0.58), yet most models struggled to adapt to real-time patient changes. On pediatric medical ethics and safety tasks, Qwen2.5-72B performed best (accuracy 92.05%), though humanistic sensitivity remained limited. These findings indicate that pediatric LLMs are constrained by limited dynamic decision-making and underdeveloped humanistic care. Future development should focus on multimodal integration and a clinical feedback-model iteration loop to enhance safety, interpretability, and human-AI collaboration. While current LLMs cannot independently perform pediatric care, they hold promise for decision support, medical education, and patient communication, laying the groundwork for a safe, trustworthy, and collaborative intelligent pediatric healthcare system. |
| 2025-11-17 | [Reasoning Shapes Alignment: Investigating Cultural Alignment in Large Reasoning Models with Cultural Norms](http://arxiv.org/abs/2511.13359v1) | Yuhang Wang, Yanxu Zhu et al. | The advanced reasoning capabilities of Large Reasoning Models enable them to thoroughly understand and apply safety policies through deliberate thought processes, thereby improving the models' safety. Beyond safety, these models must also be able to reflect the diverse range of human values across various cultures. This paper presents the Cultural Norm-based Cultural Alignment (CNCA) framework, which enables models to leverage their powerful reasoning ability to align with cultural norms. Specifically, we propose three methods to automatically mine cultural norms from limited survey data and explore ways to effectively utilize these norms for improving cultural alignment. Two alignment paradigms are examined: an in-context alignment method, where cultural norms are explicitly integrated into the user context, and a fine-tuning-based method, which internalizes norms through enhanced Chain-of-Thought training data. Comprehensive experiments demonstrate the effectiveness of these methods, highlighting that models with stronger reasoning capabilities benefit more from cultural norm mining and utilization. Our findings emphasize the potential for reasoning models to better reflect diverse human values through culturally informed alignment strategies. |
| 2025-11-17 | [CorrectAD: A Self-Correcting Agentic System to Improve End-to-end Planning in Autonomous Driving](http://arxiv.org/abs/2511.13297v1) | Enhui Ma, Lijun Zhou et al. | End-to-end planning methods are the de facto standard of the current autonomous driving system, while the robustness of the data-driven approaches suffers due to the notorious long-tail problem (i.e., rare but safety-critical failure cases). In this work, we explore whether recent diffusion-based video generation methods (a.k.a. world models), paired with structured 3D layouts, can enable a fully automated pipeline to self-correct such failure cases. We first introduce an agent to simulate the role of product manager, dubbed PM-Agent, which formulates data requirements to collect data similar to the failure cases. Then, we use a generative model that can simulate both data collection and annotation. However, existing generative models struggle to generate high-fidelity data conditioned on 3D layouts. To address this, we propose DriveSora, which can generate spatiotemporally consistent videos aligned with the 3D annotations requested by PM-Agent. We integrate these components into our self-correcting agentic system, CorrectAD. Importantly, our pipeline is an end-to-end model-agnostic and can be applied to improve any end-to-end planner. Evaluated on both nuScenes and a more challenging in-house dataset across multiple end-to-end planners, CorrectAD corrects 62.5% and 49.8% of failure cases, reducing collision rates by 39% and 27%, respectively. |
| 2025-11-17 | [Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation](http://arxiv.org/abs/2511.13269v1) | Lingfeng Zhang, Yuchen Zhang et al. | Vision-Language Models (VLMs), leveraging their powerful visual perception and reasoning capabilities, have been widely applied in Unmanned Aerial Vehicle (UAV) tasks. However, the spatial intelligence capabilities of existing VLMs in UAV scenarios remain largely unexplored, raising concerns about their effectiveness in navigating and interpreting dynamic environments. To bridge this gap, we introduce SpatialSky-Bench, a comprehensive benchmark specifically designed to evaluate the spatial intelligence capabilities of VLMs in UAV navigation. Our benchmark comprises two categories-Environmental Perception and Scene Understanding-divided into 13 subcategories, including bounding boxes, color, distance, height, and landing safety analysis, among others. Extensive evaluations of various mainstream open-source and closed-source VLMs reveal unsatisfactory performance in complex UAV navigation scenarios, highlighting significant gaps in their spatial capabilities. To address this challenge, we developed the SpatialSky-Dataset, a comprehensive dataset containing 1M samples with diverse annotations across various scenarios. Leveraging this dataset, we introduce Sky-VLM, a specialized VLM designed for UAV spatial reasoning across multiple granularities and contexts. Extensive experimental results demonstrate that Sky-VLM achieves state-of-the-art performance across all benchmark tasks, paving the way for the development of VLMs suitable for UAV scenarios. The source code is available at https://github.com/linglingxiansen/SpatialSKy. |
| 2025-11-17 | [TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](http://arxiv.org/abs/2511.13223v1) | Yuxiang Zhang, Zhengxu Yu et al. | Emerging reasoning LLMs such as OpenAI-o1 and DeepSeek-R1 have achieved strong performance on complex reasoning tasks by generating long chain-of-thought (CoT) traces. However, these long CoTs result in increased token usage, leading to higher inference latency and memory consumption. As a result, balancing accuracy and reasoning efficiency has become essential for deploying reasoning LLMs in practical applications. Existing long-to-short (Long2Short) methods aim to reduce inference length but often sacrifice accuracy, revealing a need for an approach that maintains performance while lowering token costs. To address this efficiency-accuracy tradeoff, we propose TokenSqueeze, a novel Long2Short method that condenses reasoning paths while preserving performance and relying exclusively on self-generated data. First, to prevent performance degradation caused by excessive compression of reasoning depth, we propose to select self-generated samples whose reasoning depth is adaptively matched to the complexity of the problem. To further optimize the linguistic expression without altering the underlying reasoning paths, we introduce a distribution-aligned linguistic refinement method that enhances the clarity and conciseness of the reasoning path while preserving its logical integrity. Comprehensive experimental results demonstrate the effectiveness of TokenSqueeze in reducing token usage while maintaining accuracy. Notably, DeepSeek-R1-Distill-Qwen-7B fine-tuned using our proposed method achieved a 50\% average token reduction while preserving accuracy on the MATH500 benchmark. TokenSqueeze exclusively utilizes the model's self-generated data, enabling efficient and high-fidelity reasoning without relying on manually curated short-answer datasets across diverse applications. Our code is available at https://github.com/zhangyx1122/TokenSqueeze. |
| 2025-11-17 | [Event-Triggered Regulation of Mixed-Autonomy Traffic Under Varying Traffic Conditions](http://arxiv.org/abs/2511.13206v1) | Yihuai Zhang, Huan Yu | Modeling and congestion mitigation of mixed-autonomy traffic systems consisting of human-driven vehicles (HVs) and autonomous vehicles (AVs) have become increasingly critical with the rapid development of autonomous driving technology. This paper develops an event-triggered control (ETC) framework for mitigating congestion in such systems, which are modeled using an extended Aw-Rascle-Zhang (ARZ) formulation consisting of coupled 4 x 4 hyperbolic partial differential equations (PDEs). Ramp metering is employed as the boundary actuation mechanism. To reduce computational and communication burdens while avoiding excessive ramp signal changes, we design the ETC strategy based on the backstepping method, together with an observer-based ETC formulation for practical implementation under limited sensing. Rigorous Lyapunov analysis ensures exponential convergence and avoidance of Zeno behavior. Extensive simulations validate the proposed approach under diverse traffic scenarios, including varying AV penetration rates, different spacing policies, multiple demand levels, and non-recurrent congestion patterns. Results show that ETC not only stabilizes mixed traffic flows but also significantly reduces control updates, improving driver comfort, and roadway safety. Higher AV penetration rates lead to longer release time and fewer triggering events, indicating the positive impact of AVs in mitigating traffic congestion while reducing computational resource usage. Compared to continuous backstepping controllers, the proposed ETC achieves near-equivalent stabilization performance with far fewer controller updates, resulting in longer signal release time that reduces driver distraction, which demonstrates great potential for ETC applications in traffic management. |
| 2025-11-17 | [VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language](http://arxiv.org/abs/2511.13127v1) | Zonghao Ying, Moyang Chen et al. | Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models. |
| 2025-11-17 | [A Comprehensive Review of Advancements in Powering and Charging Systems for Unmanned Aerial Vehicles](http://arxiv.org/abs/2511.13122v1) | Harsh Abhinandan, Aditya Dhanraj et al. | Unmanned Aerial Vehicles (UAVs) or drones have witnessed a spectacular surge in applications for military, commercial, and civilian purposes. However, their potential for flight is always limited by the finite power budget of their onboard power supplies. The limited flight time problem has led to intensive research into new sources of power and innovative charging strategies to enable protracted, autonomous flight. This paper gives a comparative summary of the current state-of-the-art in UAV power and refuelling technology. The paper begins with an analysis of the variety of energy sources, from classical batteries to fuel cells and hybrid systems, based on their relative advantages and disadvantages in energy density, weight, and safety. Subsequently, the review explores a spectrum of replenishment options, from simple manual battery swapping to sophisticated high-tech automatic docking stations and smart contact-based charging pads. Most of the review is dedicated to the newer technology of wireless power transfer, which involves near-field (inductive, capacitive) and far-field (laser, microwave) technology. The article also delves into the most important power electronic converter topologies, battery management systems, and control approaches that form the core of these charging systems. Finally, it recapitulates the most significant challenges in technical, economic, and social aspects for promising avenues of future research. The comprehensive review is a valuable guide for researchers, engineers, and policymakers striving to enhance UAV operational performance. |

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



