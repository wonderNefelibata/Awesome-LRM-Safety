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
| 2025-05-14 | [Adversarial Suffix Filtering: a Defense Pipeline for LLMs](http://arxiv.org/abs/2505.09602v1) | David Khachaturov, Robert Mullins | Large Language Models (LLMs) are increasingly embedded in autonomous systems and public-facing environments, yet they remain susceptible to jailbreak vulnerabilities that may undermine their security and trustworthiness. Adversarial suffixes are considered to be the current state-of-the-art jailbreak, consistently outperforming simpler methods and frequently succeeding even in black-box settings. Existing defenses rely on access to the internal architecture of models limiting diverse deployment, increase memory and computation footprints dramatically, or can be bypassed with simple prompt engineering methods. We introduce $\textbf{Adversarial Suffix Filtering}$ (ASF), a lightweight novel model-agnostic defensive pipeline designed to protect LLMs against adversarial suffix attacks. ASF functions as an input preprocessor and sanitizer that detects and filters adversarially crafted suffixes in prompts, effectively neutralizing malicious injections. We demonstrate that ASF provides comprehensive defense capabilities across both black-box and white-box attack settings, reducing the attack efficacy of state-of-the-art adversarial suffix generation methods to below 4%, while only minimally affecting the target model's capabilities in non-adversarial scenarios. |
| 2025-05-14 | [How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference](http://arxiv.org/abs/2505.09598v1) | Nidhal Jegham, Marwen Abdelatti et al. | As large language models (LLMs) spread across industries, understanding their environmental footprint at the inference level is no longer optional; it is essential. However, most existing studies exclude proprietary models, overlook infrastructural variability and overhead, or focus solely on training, even as inference increasingly dominates AI's environmental impact. To bridge this gap, this paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: although individual queries are efficient, their global scale drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards. |
| 2025-05-14 | [Tropical Fermat-Weber Points over Bergman Fans](http://arxiv.org/abs/2505.09584v1) | Shelby Cox, John Sabol et al. | Given a finite set $[p]:= \{1, \ldots , p\}$, it is well known that the space of all ultrametrics on $[p]$ is the Bergman fan associated to the matroid underlying the complete graph of the vertex set $[p]$. Lin et al.~showed that the set of tropical Fermat-Weber points might not be contained in the space of ultrametrics on $[p]$ even if all input data points are in the space. Here we consider a more general set up and we focus on Fermat-Weber points with respect to the tropical metric over the Bergman fan associated with a matroid with the ground set of $q$ elements. We show that there always exists a Fermat-Weber point in the Bergman fan for data in the Bergman fan and we describe explicitly the set of all Fermat-Weber points in the Bergman fan. Then we introduce the natural extension of the safety radius introduced by Atteson to the set of Fermat-Weber points in the Bergman fan of a matroid. |
| 2025-05-14 | [Conformal Bounds on Full-Reference Image Quality for Imaging Inverse Problems](http://arxiv.org/abs/2505.09528v1) | Jeffrey Wen, Rizwan Ahmad et al. | In imaging inverse problems, we would like to know how close the recovered image is to the true image in terms of full-reference image quality (FRIQ) metrics like PSNR, SSIM, LPIPS, etc. This is especially important in safety-critical applications like medical imaging, where knowing that, say, the SSIM was poor could potentially avoid a costly misdiagnosis. But since we don't know the true image, computing FRIQ is non-trivial. In this work, we combine conformal prediction with approximate posterior sampling to construct bounds on FRIQ that are guaranteed to hold up to a user-specified error probability. We demonstrate our approach on image denoising and accelerated magnetic resonance imaging (MRI) problems. Code is available at https://github.com/jwen307/quality_uq. |
| 2025-05-14 | [Evaluating GPT- and Reasoning-based Large Language Models on Physics Olympiad Problems: Surpassing Human Performance and Implications for Educational Assessment](http://arxiv.org/abs/2505.09438v1) | Paul Tschisgale, Holger Maus et al. | Large language models (LLMs) are now widely accessible, reaching learners at all educational levels. This development has raised concerns that their use may circumvent essential learning processes and compromise the integrity of established assessment formats. In physics education, where problem solving plays a central role in instruction and assessment, it is therefore essential to understand the physics-specific problem-solving capabilities of LLMs. Such understanding is key to informing responsible and pedagogically sound approaches to integrating LLMs into instruction and assessment. This study therefore compares the problem-solving performance of a general-purpose LLM (GPT-4o, using varying prompting techniques) and a reasoning-optimized model (o1-preview) with that of participants of the German Physics Olympiad, based on a set of well-defined Olympiad problems. In addition to evaluating the correctness of the generated solutions, the study analyzes characteristic strengths and limitations of LLM-generated solutions. The findings of this study indicate that both tested LLMs (GPT-4o and o1-preview) demonstrate advanced problem-solving capabilities on Olympiad-type physics problems, on average outperforming the human participants. Prompting techniques had little effect on GPT-4o's performance, while o1-preview almost consistently outperformed both GPT-4o and the human benchmark. Based on these findings, the study discusses implications for the design of summative and formative assessment in physics education, including how to uphold assessment integrity and support students in critically engaging with LLMs. |
| 2025-05-14 | [SafePath: Conformal Prediction for Safe LLM-Based Autonomous Navigation](http://arxiv.org/abs/2505.09427v1) | Achref Doula, Max Mühläuser et al. | Large Language Models (LLMs) show growing promise in autonomous driving by reasoning over complex traffic scenarios to generate path plans. However, their tendencies toward overconfidence, and hallucinations raise critical safety concerns. We introduce SafePath, a modular framework that augments LLM-based path planning with formal safety guarantees using conformal prediction. SafePath operates in three stages. In the first stage, we use an LLM that generates a set of diverse candidate paths, exploring possible trajectories based on agent behaviors and environmental cues. In the second stage, SafePath filters out high-risk trajectories while guaranteeing that at least one safe option is included with a user-defined probability, through a multiple-choice question-answering formulation that integrates conformal prediction. In the final stage, our approach selects the path with the lowest expected collision risk when uncertainty is low or delegates control to a human when uncertainty is high. We theoretically prove that SafePath guarantees a safe trajectory with a user-defined probability, and we show how its human delegation rate can be tuned to balance autonomy and safety. Extensive experiments on nuScenes and Highway-env show that SafePath reduces planning uncertainty by 77\% and collision rates by up to 70\%, demonstrating effectiveness in making LLM-driven path planning more safer. |
| 2025-05-14 | [Coordinated Multi-Valve Disturbance-Rejection Pressure Control for High-Altitude Test Stands via Exterior Penalty Functions](http://arxiv.org/abs/2505.09352v1) | Zhang Louyue, Li Xin et al. | High altitude simulation test benches for aero engines employ multi chamber, multi valve intake systems that demand effective decoupling and strong disturbance rejection during transient tests. This paper proposes a coordinated active disturbance rejection control (ADRC) scheme based on an external penalty function. The chamber pressure safety limit is reformulated as an inequality constrained optimization problem, and an exponential penalty together with a gradient based algorithm is designed for dynamic constraint relaxation, with global convergence rigorously proven. A coordination term is then integrated into a distributed ADRC framework to yield a multi valve coordinated LADRC controller, whose asymptotic stability is established via Lyapunov theory. Hardware in the loop simulations using MATLAB/Simulink and a PLC demonstrate that, under $\pm$3 kPa pressure constraints, chamber V2's maximum error is 1.782 kPa (77.1\% lower than PID control), and under a 180 kg/s^2 flow rate disturbance, valve oscillations decrease from $\pm$27\% to $\pm$5\% (an 81.5\% reduction). These results confirm the proposed method's superior disturbance rejection and decoupling performance. |
| 2025-05-14 | [Safe Primal-Dual Optimization with a Single Smooth Constraint](http://arxiv.org/abs/2505.09349v1) | Ilnura Usmanova, Kfir Yehuda Levy | This paper addresses the problem of safe optimization under a single smooth constraint, a scenario that arises in diverse real-world applications such as robotics and autonomous navigation. The objective of safe optimization is to solve a black-box minimization problem while strictly adhering to a safety constraint throughout the learning process. Existing methods often suffer from high sample complexity due to their noise sensitivity or poor scalability with number of dimensions, limiting their applicability. We propose a novel primal-dual optimization method that, by carefully adjusting dual step-sizes and constraining primal updates, ensures the safety of both primal and dual sequences throughout the optimization. Our algorithm achieves a convergence rate that significantly surpasses current state-of-the-art techniques. Furthermore, to the best of our knowledge, it is the first primal-dual approach to guarantee safe updates. Simulations corroborate our theoretical findings, demonstrating the practical benefits of our method. We also show how the method can be extended to multiple constraints. |
| 2025-05-14 | [Access Controls Will Solve the Dual-Use Dilemma](http://arxiv.org/abs/2505.09341v1) | Evžen Wybitul | AI safety systems face a dual-use dilemma. Since the same request can be either harmless or harmful depending on who made it and why, if the system makes decisions based solely on the request's content, it will refuse some legitimate queries and let pass harmful ones. To address this, we propose a conceptual access control framework, based on verified user credentials (such as institutional affiliation) and classifiers that assign model outputs to risk categories (such as advanced virology). The system permits responses only when the user's verified credentials match the category's requirements. For implementation of the model output classifiers, we introduce a theoretical approach utilizing small, gated expert modules integrated into the generator model, trained with gradient routing, that enable efficient risk detection without the capability gap problems of external monitors. While open questions remain about the verification mechanisms, risk categories, and the technical implementation, our framework makes the first step toward enabling granular governance of AI capabilities: verified users gain access to specialized knowledge without arbitrary restrictions, while adversaries are blocked from it. This contextual approach reconciles model utility with robust safety, addressing the dual-use dilemma. |
| 2025-05-14 | [Privacy-Preserving Runtime Verification](http://arxiv.org/abs/2505.09276v1) | Thomas A. Henzinger, Mahyar Karimi et al. | Runtime verification offers scalable solutions to improve the safety and reliability of systems. However, systems that require verification or monitoring by a third party to ensure compliance with a specification might contain sensitive information, causing privacy concerns when usual runtime verification approaches are used. Privacy is compromised if protected information about the system, or sensitive data that is processed by the system, is revealed. In addition, revealing the specification being monitored may undermine the essence of third-party verification.   In this work, we propose two novel protocols for the privacy-preserving runtime verification of systems against formal sequential specifications. In our first protocol, the monitor verifies whether the system satisfies the specification without learning anything else, though both parties are aware of the specification. Our second protocol ensures that the system remains oblivious to the monitored specification, while the monitor learns only whether the system satisfies the specification and nothing more. Our protocols adapt and improve existing techniques used in cryptography, and more specifically, multi-party computation.   The sequential specification defines the observation step of the monitor, whose granularity depends on the situation (e.g., banks may be monitored on a daily basis). Our protocols exchange a single message per observation step, after an initialisation phase. This design minimises communication overhead, enabling relatively lightweight privacy-preserving monitoring. We implement our approach for monitoring specifications described by register automata and evaluate it experimentally. |
| 2025-05-14 | [On verification and constraint generation for families of similar hybrid automata](http://arxiv.org/abs/2505.09244v1) | Viorica Sofronie-Stokkermans, Philipp Marohn | In this paper we give an overview of results on the analysis of parametric linear hybrid automata, and of systems of similar linear hybrid automata: We present possibilities of describing systems with a parametric (i.e. not explicitly specified) number of similar components which can be connected to other systems, such that some parts in the description might be underspecified (i.e. parametric). We consider global safety properties for such systems, expressed by universally quantified formulae, using quantification over variables ranging over the component systems. We analyze possibilities of using methods for hierarchical reasoning and symbol elimination for determining relationships on (some of) the parameters used in the description of these systems under which the global safety properties are guaranteed to be inductive invariants. We discuss an implementation and illustrate its use on several examples. |
| 2025-05-14 | [OpenLKA: An Open Dataset of Lane Keeping Assist from Recent Car Models under Real-world Driving Conditions](http://arxiv.org/abs/2505.09092v1) | Yuhang Wang, Abdulaziz Alhuraish et al. | Lane Keeping Assist (LKA) is widely adopted in modern vehicles, yet its real-world performance remains underexplored due to proprietary systems and limited data access. This paper presents OpenLKA, the first open, large-scale dataset for LKA evaluation and improvement. It includes 400 hours of driving data from 50+ production vehicle models, collected through extensive road testing in Tampa, Florida and global contributions from the Comma.ai driving community. The dataset spans a wide range of challenging scenarios, including complex road geometries, degraded lane markings, adverse weather, lighting conditions and surrounding traffic. The dataset is multimodal, comprising: i) full CAN bus streams, decoded using custom reverse-engineered DBC files to extract key LKA events (e.g., system disengagements, lane detection failures); ii) synchronized high-resolution dash-cam video; iii) real-time outputs from Openpilot, providing accurate estimates of road curvature and lane positioning; iv) enhanced scene annotations generated by Vision Language Models, describing lane visibility, pavement quality, weather, lighting, and traffic conditions. By integrating vehicle-internal signals with high-fidelity perception and rich semantic context, OpenLKA provides a comprehensive platform for benchmarking the real-world performance of production LKA systems, identifying safety-critical operational scenarios, and assessing the readiness of current road infrastructure for autonomous driving. The dataset is publicly available at: https://github.com/OpenLKA/OpenLKA. |
| 2025-05-14 | [Reach-Avoid-Stabilize Using Admissible Control Sets](http://arxiv.org/abs/2505.09058v1) | Zheng Gong, Boyang Li et al. | Hamilton-Jacobi Reachability (HJR) analysis has been successfully used in many robotics and control tasks, and is especially effective in computing reach-avoid sets and control laws that enable an agent to reach a goal while satisfying state constraints. However, the original HJR formulation provides no guarantees of safety after a) the prescribed time horizon, or b) goal satisfaction. The reach-avoid-stabilize (RAS) problem has therefore gained a lot of focus: find the set of initial states (the RAS set), such that the trajectory can reach the target, and stabilize to some point of interest (POI) while avoiding obstacles. Solving RAS problems using HJR usually requires defining a new value function, whose zero sub-level set is the RAS set. The existing methods do not consider the problem when there are a series of targets to reach and/or obstacles to avoid. We propose a method that uses the idea of admissible control sets; we guarantee that the system will reach each target while avoiding obstacles as prescribed by the given time series. Moreover, we guarantee that the trajectory ultimately stabilizes to the POI. The proposed method provides an under-approximation of the RAS set, guaranteeing safety. Numerical examples are provided to validate the theory. |
| 2025-05-14 | [FocusE: A semantic extension of FocusST](http://arxiv.org/abs/2505.09032v1) | Maria Spichkova | To analyse and verify the safety and security properties of interactive systems, a formal specification might be necessary. There are many types of formal languages and frameworks. The decision regarding what type of formal specification should be applied in each particular case depends on many factors. One of the approaches to specify interactive systems formally is to present them as a composition of components processing data and control streams. In this short paper, we present FocusE, a formal approach for modelling event-based streams. The proposed approach is based on a formal language FocusST, and can be seen as its semantic extension. |
| 2025-05-13 | [A suite of LMs comprehend puzzle statements as well as humans](http://arxiv.org/abs/2505.08996v1) | Adele E Goldberg, Supantho Rakshit et al. | Recent claims suggest that large language models (LMs) underperform humans in comprehending minimally complex English statements (Dentella et al., 2024). Here, we revisit those findings and argue that human performance was overestimated, while LLM abilities were underestimated. Using the same stimuli, we report a preregistered study comparing human responses in two conditions: one allowed rereading (replicating the original study), and one that restricted rereading (a more naturalistic comprehension test). Human accuracy dropped significantly when rereading was restricted (73%), falling below that of Falcon-180B-Chat (76%) and GPT-4 (81%). The newer GPT-o1 model achieves perfect accuracy. Results further show that both humans and models are disproportionately challenged by queries involving potentially reciprocal actions (e.g., kissing), suggesting shared pragmatic sensitivities rather than model-specific deficits. Additional analyses using Llama-2-70B log probabilities, a recoding of open-ended model responses, and grammaticality ratings of other sentences reveal systematic underestimation of model performance. We find that GPT-4o can align with either naive or expert grammaticality judgments, depending on prompt framing. These findings underscore the need for more careful experimental design and coding practices in LLM evaluation, and they challenge the assumption that current models are inherently weaker than humans at language comprehension. |
| 2025-05-13 | [Performance Gains of LLMs With Humans in a World of LLMs Versus Humans](http://arxiv.org/abs/2505.08902v1) | Lucas McCullum, Pelagie Ami Agassi et al. | Currently, a considerable research effort is devoted to comparing LLMs to a group of human experts, where the term "expert" is often ill-defined or variable, at best, in a state of constantly updating LLM releases. Without proper safeguards in place, LLMs will threaten to cause harm to the established structure of safe delivery of patient care which has been carefully developed throughout history to keep the safety of the patient at the forefront. A key driver of LLM innovation is founded on community research efforts which, if continuing to operate under "humans versus LLMs" principles, will expedite this trend. Therefore, research efforts moving forward must focus on effectively characterizing the safe use of LLMs in clinical settings that persist across the rapid development of novel LLM models. In this communication, we demonstrate that rather than comparing LLMs to humans, there is a need to develop strategies enabling efficient work of humans with LLMs in an almost symbiotic manner. |
| 2025-05-13 | [Deep reinforcement learning-based longitudinal control strategy for automated vehicles at signalised intersections](http://arxiv.org/abs/2505.08896v1) | Pankaj Kumar, Aditya Mishra et al. | Developing an autonomous vehicle control strategy for signalised intersections (SI) is one of the challenging tasks due to its inherently complex decision-making process. This study proposes a Deep Reinforcement Learning (DRL) based longitudinal vehicle control strategy at SI. A comprehensive reward function has been formulated with a particular focus on (i) distance headway-based efficiency reward, (ii) decision-making criteria during amber light, and (iii) asymmetric acceleration/ deceleration response, along with the traditional safety and comfort criteria. This reward function has been incorporated with two popular DRL algorithms, Deep Deterministic Policy Gradient (DDPG) and Soft-Actor Critic (SAC), which can handle the continuous action space of acceleration/deceleration. The proposed models have been trained on the combination of real-world leader vehicle (LV) trajectories and simulated trajectories generated using the Ornstein-Uhlenbeck (OU) process. The overall performance of the proposed models has been tested using Cumulative Distribution Function (CDF) plots and compared with the real-world trajectory data. The results show that the RL models successfully maintain lower distance headway (i.e., higher efficiency) and jerk compared to human-driven vehicles without compromising safety. Further, to assess the robustness of the proposed models, we evaluated the model performance on diverse safety-critical scenarios, in terms of car-following and traffic signal compliance. Both DDPG and SAC models successfully handled the critical scenarios, while the DDPG model showed smoother action profiles compared to the SAC model. Overall, the results confirm that DRL-based longitudinal vehicle control strategy at SI can help to improve traffic safety, efficiency, and comfort. |
| 2025-05-13 | [Intelligent Road Anomaly Detection with Real-time Notification System for Enhanced Road Safety](http://arxiv.org/abs/2505.08882v1) | Ali Almakhluk, Uthman Baroudi et al. | This study aims to improve transportation safety, especially traffic safety. Road damage anomalies such as potholes and cracks have emerged as a significant and recurring cause for accidents. To tackle this problem and improve road safety, a comprehensive system has been developed to detect potholes, cracks (e.g. alligator, transverse, longitudinal), classify their sizes, and transmit this data to the cloud for appropriate action by authorities. The system also broadcasts warning signals to nearby vehicles warning them if a severe anomaly is detected on the road. Moreover, the system can count road anomalies in real-time. It is emulated through the utilization of Raspberry Pi, a camera module, deep learning model, laptop, and cloud service. Deploying this innovative solution aims to proactively enhance road safety by notifying relevant authorities and drivers about the presence of potholes and cracks to take actions, thereby mitigating potential accidents arising from this prevalent road hazard leading to safer road conditions for the whole community. |
| 2025-05-13 | [Generative AI for Autonomous Driving: Frontiers and Opportunities](http://arxiv.org/abs/2505.08854v1) | Yuping Wang, Shuo Xing et al. | Generative Artificial Intelligence (GenAI) constitutes a transformative technological wave that reconfigures industries through its unparalleled capabilities for content creation, reasoning, planning, and multimodal understanding. This revolutionary force offers the most promising path yet toward solving one of engineering's grandest challenges: achieving reliable, fully autonomous driving, particularly the pursuit of Level 5 autonomy. This survey delivers a comprehensive and critical synthesis of the emerging role of GenAI across the autonomous driving stack. We begin by distilling the principles and trade-offs of modern generative modeling, encompassing VAEs, GANs, Diffusion Models, and Large Language Models (LLMs). We then map their frontier applications in image, LiDAR, trajectory, occupancy, video generation as well as LLM-guided reasoning and decision making. We categorize practical applications, such as synthetic data workflows, end-to-end driving strategies, high-fidelity digital twin systems, smart transportation networks, and cross-domain transfer to embodied AI. We identify key obstacles and possibilities such as comprehensive generalization across rare cases, evaluation and safety checks, budget-limited implementation, regulatory compliance, ethical concerns, and environmental effects, while proposing research plans across theoretical assurances, trust metrics, transport integration, and socio-technical influence. By unifying these threads, the survey provides a forward-looking reference for researchers, engineers, and policymakers navigating the convergence of generative AI and advanced autonomous mobility. An actively maintained repository of cited works is available at https://github.com/taco-group/GenAI4AD. |
| 2025-05-13 | [PCS-UQ: Uncertainty Quantification via the Predictability-Computability-Stability Framework](http://arxiv.org/abs/2505.08784v1) | Abhineet Agarwal, Michael Xiao et al. | As machine learning (ML) models are increasingly deployed in high-stakes domains, trustworthy uncertainty quantification (UQ) is critical for ensuring the safety and reliability of these models. Traditional UQ methods rely on specifying a true generative model and are not robust to misspecification. On the other hand, conformal inference allows for arbitrary ML models but does not consider model selection, which leads to large interval sizes. We tackle these drawbacks by proposing a UQ method based on the predictability, computability, and stability (PCS) framework for veridical data science proposed by Yu and Kumbier. Specifically, PCS-UQ addresses model selection by using a prediction check to screen out unsuitable models. PCS-UQ then fits these screened algorithms across multiple bootstraps to assess inter-sample variability and algorithmic instability, enabling more reliable uncertainty estimates. Further, we propose a novel calibration scheme that improves local adaptivity of our prediction sets. Experiments across $17$ regression and $6$ classification datasets show that PCS-UQ achieves the desired coverage and reduces width over conformal approaches by $\approx 20\%$. Further, our local analysis shows PCS-UQ often achieves target coverage across subgroups while conformal methods fail to do so. For large deep-learning models, we propose computationally efficient approximation schemes that avoid the expensive multiple bootstrap trainings of PCS-UQ. Across three computer vision benchmarks, PCS-UQ reduces prediction set size over conformal methods by $20\%$. Theoretically, we show a modified PCS-UQ algorithm is a form of split conformal inference and achieves the desired coverage with exchangeable data. |

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



