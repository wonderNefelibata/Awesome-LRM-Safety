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
| 2025-12-16 | [TACK Tunnel Data (TTD): A Benchmark Dataset for Deep Learning-Based Defect Detection in Tunnels](http://arxiv.org/abs/2512.14477v1) | Andreas Sjölander, Valeria Belloni et al. | Tunnels are essential elements of transportation infrastructure, but are increasingly affected by ageing and deterioration mechanisms such as cracking. Regular inspections are required to ensure their safety, yet traditional manual procedures are time-consuming, subjective, and costly. Recent advances in mobile mapping systems and Deep Learning (DL) enable automated visual inspections. However, their effectiveness is limited by the scarcity of tunnel datasets. This paper introduces a new publicly available dataset containing annotated images of three different tunnel linings, capturing typical defects: cracks, leaching, and water infiltration. The dataset is designed to support supervised, semi-supervised, and unsupervised DL methods for defect detection and segmentation. Its diversity in texture and construction techniques also enables investigation of model generalization and transferability across tunnel types. By addressing the critical lack of domain-specific data, this dataset contributes to advancing automated tunnel inspection and promoting safer, more efficient infrastructure maintenance strategies. |
| 2025-12-16 | [Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space](http://arxiv.org/abs/2512.14448v1) | Xingfu Zhou, Pengfei Wang | Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis. |
| 2025-12-16 | [A Comprehensive Safety Metric to Evaluate Perception in Autonomous Systems](http://arxiv.org/abs/2512.14367v1) | Georg Volk, Jörg Gamerdinger et al. | Complete perception of the environment and its correct interpretation is crucial for autonomous vehicles. Object perception is the main component of automotive surround sensing. Various metrics already exist for the evaluation of object perception. However, objects can be of different importance depending on their velocity, orientation, distance, size, or the potential damage that could be caused by a collision due to a missed detection. Thus, these additional parameters have to be considered for safety evaluation. We propose a new safety metric that incorporates all these parameters and returns a single easily interpretable safety assessment score for object perception. This new metric is evaluated with both real world and virtual data sets and compared to state of the art metrics. |
| 2025-12-16 | [Criminal Liability in AI-Enabled Autonomous Vehicles: A Comparative Study](http://arxiv.org/abs/2512.14330v1) | Sahibpreet Singh, Manjit Singh | AI revolutionizes transportation through autonomous vehicles (AVs) but introduces complex criminal liability issues regarding infractions. This study employs a comparative legal analysis of primary statutes, real-world liability claims, and academic literature across the US, Germany, UK, China, and India; jurisdictions selected for their technological advancement and contrasting regulatory approaches. The research examines the attribution of human error, AI moral agency, and the identification of primary offenders in AV incidents. Findings reveal fragmented regulatory landscapes: India and the US rely on loose networks of state laws, whereas the UK enacted the pioneering Automated and Electric Vehicles Act 2018. Germany enforces strict safety standards, distinguishing liability based on the vehicle's operating mode, while China similarly aims for a stringent liability regime. The study concludes that globally harmonized legal standards are essential to foster technological innovation while ensuring minimum risk and clear liability attribution. |
| 2025-12-16 | [Worldwide Scientific Landscape on Fires in Photovoltaic](http://arxiv.org/abs/2512.14289v1) | Esther Salmerón-Manzano, David Muñoz-Rodríguez et al. | The rapid growth of photovoltaic (PV) technology in recent years called for a comprehensive assessment of the global scientific landscape on fires associated with PV energy installations. This study examines the scientific literature indexed in Scopus from 1983 to 2023. It reveals a striking increase in output since 2011, with nearly one hundred publications in the most recent year under review. This growth of interest has occurred in parallel with the global expansion of photovoltaics. The majority of studies in this field are classified as engineering, with 34% of publications in this area. The USA leads the way with over 160 publications, followed by China with 125. Two institutions in the USA are particularly prominent in this field: Sandia National Laboratories in New Mexico with 22 publications, and the National Renewable Energy Laboratory in Colorado with 16 publications. The second institution is the University of Science and Technology of China, which has published 17 articles on the subject. A close examination of the evolution of keywords reveals a remarkable transformation in the scientific landscape over the past 10 years, from 2013 to 2023. The evolution of keywords suggests a maturation in the understanding of fire risks associated with photovoltaic energy. A total of seven scientific communities have been identified in which these works are grouped according to their keywords. These include Fire and Energy Storage, PV faults, Fire resistance, Fire hazard, Fire detectors, Deep learning, and Fire safety. It has been found that fires caused by PV installations are not listed as a cause of fire starts. This should be taken into account when conducting preventive analyses of this potential danger, particularly in light of the possible development of agrivoltaics, where facilities will be mainly located in the natural environment. |
| 2025-12-16 | [OmniDrive-R1: Reinforcement-driven Interleaved Multi-modal Chain-of-Thought for Trustworthy Vision-Language Autonomous Driving](http://arxiv.org/abs/2512.14044v1) | Zhenguo Zhang, Haohan Zhen et al. | The deployment of Vision-Language Models (VLMs) in safety-critical domains like autonomous driving (AD) is critically hindered by reliability failures, most notably object hallucination. This failure stems from their reliance on ungrounded, text-based Chain-of-Thought (CoT) reasoning.While existing multi-modal CoT approaches attempt mitigation, they suffer from two fundamental flaws: (1) decoupled perception and reasoning stages that prevent end-to-end joint optimization, and (2) reliance on expensive, dense localization labels.Thus we introduce OmniDrive-R1, an end-to-end VLM framework designed for autonomous driving, which unifies perception and reasoning through an interleaved Multi-modal Chain-of-Thought (iMCoT) mechanism. Our core innovation is an Reinforcement-driven visual grounding capability, enabling the model to autonomously direct its attention and "zoom in" on critical regions for fine-grained analysis. This capability is enabled by our pure two-stage reinforcement learning training pipeline and Clip-GRPO algorithm. Crucially, Clip-GRPO introduces an annotation-free, process-based grounding reward. This reward not only eliminates the need for dense labels but also circumvents the instability of external tool calls by enforcing real-time cross-modal consistency between the visual focus and the textual reasoning. Extensive experiments on DriveLMM-o1 demonstrate our model's significant improvements. Compared to the baseline Qwen2.5VL-7B, OmniDrive-R1 improves the overall reasoning score from 51.77% to 80.35%, and the final answer accuracy from 37.81% to 73.62%. |
| 2025-12-16 | [Liquid Handling of the JUNO Experiment](http://arxiv.org/abs/2512.14033v1) | Jiajun Li, Yuekun Heng et al. | The Filling, Overflow, and Circulation (FOC) system is a critical subsystem of the Jiangmen Underground Neutrino Observatory (JUNO), responsible for the safe handling of the Liquid Scintillator (LS) and water throughout the detector's commissioning and operational lifetime. This paper details the design and operation of the FOC system, which accomplished the filling of the world's largest LS detector--taking 45 days for water (6.4*10^4 m^3) and 200 days for LS (2.3*10^4 m^3). Throughout water filling, the liquid level difference between the Central Detector and Water Pool was rigorously maintained within safety limits. During LS filling, level control achieved +/-2 cm precision with flow regulation within +/-0.5% of setpoints. An automated control system based on Programmable Logic Controllers and the Experimental Physics and Industrial Control System framework ensured reliable operation. The system preserved LS radiopurity, maintaining 222Rn below 1 mBq/m^3 during filling and achieving 238U/232Th concentrations below 10^-16 g/g. The successful commissioning and operation of the FOC system have established it as an indispensable foundation for the stable long-term operation of the JUNO detector. |
| 2025-12-16 | [FocalComm: Hard Instance-Aware Multi-Agent Perception](http://arxiv.org/abs/2512.13982v1) | Dereje Shenkut, Vijayakumar Bhagavatula | Multi-agent collaborative perception (CP) is a promising paradigm for improving autonomous driving safety, particularly for vulnerable road users like pedestrians, via robust 3D perception. However, existing CP approaches often optimize for vehicle detection performance metrics, underperforming on smaller, safety-critical objects such as pedestrians, where detection failures can be catastrophic. Furthermore, previous CP methods rely on full feature exchange rather than communicating only salient features that help reduce false negatives. To this end, we present FocalComm, a novel collaborative perception framework that focuses on exchanging hard-instance-oriented features among connected collaborative agents. FocalComm consists of two key novel designs: (1) a learnable progressive hard instance mining (HIM) module to extract hard instance-oriented features per agent, and (2) a query-based feature-level (intermediate) fusion technique that dynamically weights these identified features during collaboration. We show that FocalComm outperforms state-of-the-art collaborative perception methods on two challenging real-world datasets (V2X-Real and DAIR-V2X) across both vehicle-centric and infrastructure-centric collaborative setups. FocalComm also shows a strong performance gain in pedestrian detection in V2X-Real. |
| 2025-12-16 | [Autonomous Construction-Site Safety Inspection Using Mobile Robots: A Multilayer VLM-LLM Pipeline](http://arxiv.org/abs/2512.13974v1) | Hossein Naderi, Alireza Shojaei et al. | Construction safety inspection remains mostly manual, and automated approaches still rely on task-specific datasets that are hard to maintain in fast-changing construction environments due to frequent retraining. Meanwhile, field inspection with robots still depends on human teleoperation and manual reporting, which are labor-intensive. This paper aims to connect what a robot sees during autonomous navigation to the safety rules that are common in construction sites, automatically generating a safety inspection report. To this end, we proposed a multi-layer framework with two main modules: robotics and AI. On the robotics side, SLAM and autonomous navigation provide repeatable coverage and targeted revisits via waypoints. On AI side, a Vision Language Model (VLM)-based layer produces scene descriptions; a retrieval component powered grounds those descriptions in OSHA and site policies; Another VLM-based layer assesses the safety situation based on rules; and finally Large Language Model (LLM) layer generates safety reports based on previous outputs. The framework is validated with a proof-of-concept implementation and evaluated in a lab environment that simulates common hazards across three scenarios. Results show high recall with competitive precision compared to state-of-the-art closed-source models. This paper contributes a transparent, generalizable pipeline that moves beyond black-box models by exposing intermediate artifacts from each layer and keeping the human in the loop. This work provides a foundation for future extensions to additional tasks and settings within and beyond construction context. |
| 2025-12-15 | [Data-Driven Control via Conditional Mean Embeddings: Formal Guarantees via Uncertain MDP Abstraction](http://arxiv.org/abs/2512.13940v1) | Ibon Gracia, Morteza Lahijanian | Controlling stochastic systems with unknown dynamics and under complex specifications is specially challenging in safety-critical settings, where performance guarantees are essential. We propose a data-driven policy synthesis framework that yields formal performance guarantees for such systems using conditional mean embeddings (CMEs) and uncertain Markov decision processes (UMDPs). From trajectory data, we learn the system's transition kernel as a CME, then construct a finite-state UMDP abstraction whose transition uncertainties capture learning and discretization errors. Next, we generate a policy with formal performance bounds through robust dynamic programming. We demonstrate and empirically validate our method through a temperature regulation benchmark. |
| 2025-12-15 | [Bond strength uncertainty quantification via confidence intervals for nondestructive evaluation of bonded composites](http://arxiv.org/abs/2512.13875v1) | Michael C. Stanley, Peter W. Spaeth et al. | As bonded composite materials are used more frequently for aerospace applications, it is necessary to certify that parts achieve desired levels of certain physical characteristics (e.g., strength) for safety and performance. Nondestructive evaluation (NDE) of adhesively bonded structures enables verification of bond physical characteristics, but uncertainty quantification (UQ) of NDE estimates is crucial for understanding risks, especially for NDE estimates like bond strength. To address the critical need for NDE UQ for adhesive bond strength estimates, we propose an optimization--based approach to computing finite--sample confidence intervals showing the range of bond strengths that could feasibly be produced by the observed data. A statistical inverse model approach is used to compute a confidence interval of specimen interfacial stiffness from swept--frequency ultrasonic phase observations and a method for propagating the interval to bond strength via a known interfacial stiffness regression is proposed. This approach requires innovating the optimization--based confidence interval to handle both a nonlinear forward model and unknown variance and developing a calibration approach to ensure that the final bond strength interval achieves at least the desired coverage level. Using model assumptions in line with current literature, we demonstrate our approach on simulated measurement data using a variety of low to high noise settings under two prototypical parameter settings. Relative to a baseline approach, we show that our method achieves better coverage and smaller intervals in high--noise settings and when a nuisance parameter is near the constraint boundary. |
| 2025-12-15 | [Safe Online Control-Informed Learning](http://arxiv.org/abs/2512.13868v1) | Tianyu Zhou, Zihao Liang et al. | This paper proposes a Safe Online Control-Informed Learning framework for safety-critical autonomous systems. The framework unifies optimal control, parameter estimation, and safety constraints into an online learning process. It employs an extended Kalman filter to incrementally update system parameters in real time, enabling robust and data-efficient adaptation under uncertainty. A softplus barrier function enforces constraint satisfaction during learning and control while eliminating the dependence on high-quality initial guesses. Theoretical analysis establishes convergence and safety guarantees, and the framework's effectiveness is demonstrated on cart-pole and robot-arm systems. |
| 2025-12-15 | [CrSe_2 and CrTe_2 Monolayers as Efficient Air Pollutants Nanosensors](http://arxiv.org/abs/2512.13843v1) | Hakkim Vovusha, Puspamitra Panigrahi et al. | Nanosensors are critical in environmental monitoring, industrial safety, and public health by detecting specific hazardous gases like CO, NO, SO_2, and CH_4 at trace levels. This study uses density functional theory (DFT) calculations to examine the gas-sensing capabilities of chromium diselenide (CrSe_2) and chromium ditelluride (CrTe_2) monolayers through their structural and electronic responses to gas adsorption. Adsorption energy analysis shows that Te vacancy-induced CrTe_2 (VTe-CrTe_2) exhibits the strongest binding with energies of -1.52, -1.79, and -1.61 eV for CO, NO, and SO_2, respectively. Similarly, CrSe_2 has its values of -1.13, -1.17, -0.90, and -1.12 eV for CO, NO, SO_2, and CH_2, respectively, indicating suitability for reversible sensing. This study also investigates how substitutional doping of Ge, Sb, and Sn influences the sensing mechanism of CrSe_2 and CrTe_2 monolayers. Density of states (DOS) analysis highlights notable electronic changes around the Fermi level, especially in VTe-CrTe_2 and Sb/Sn-doped CrTe_2, confirming their enhanced sensing abilities. Charge density difference analysis shows significant charge redistribution, with CrTe_2 experiencing stronger charge transfer effects than CrSe_2. Variations in electrostatic potential and work function further demonstrate the higher sensitivity of CrTe_2, particularly in its defective and doped forms, confirming its status as a superior material for gas sensing applications. |
| 2025-12-15 | [Constrained Policy Optimization via Sampling-Based Weight-Space Projection](http://arxiv.org/abs/2512.13788v1) | Shengfan Cao, Francesco Borrelli | Safety-critical learning requires policies that improve performance without leaving the safe operating regime. We study constrained policy learning where model parameters must satisfy unknown, rollout-based safety constraints. We propose SCPO, a sampling-based weight-space projection method that enforces safety directly in parameter space without requiring gradient access to the constraint functions. Our approach constructs a local safe region by combining trajectory rollouts with smoothness bounds that relate parameter changes to shifts in safety metrics. Each gradient update is then projected via a convex SOCP, producing a safe first-order step. We establish a safe-by-induction guarantee: starting from any safe initialization, all intermediate policies remain safe given feasible projections. In constrained control settings with a stabilizing backup policy, our approach further ensures closed-loop stability and enables safe adaptation beyond the conservative backup. On regression with harmful supervision and a constrained double-integrator task with malicious expert, our approach consistently rejects unsafe updates, maintains feasibility throughout training, and achieves meaningful primal objective improvement. |
| 2025-12-15 | [Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture Evaluation](http://arxiv.org/abs/2512.13655v1) | Richard J. Young | Safety alignment mechanisms in large language models prevent responses to harmful queries through learned refusal behavior, yet these same mechanisms impede legitimate research applications including cognitive modeling, adversarial testing, and security analysis. While abliteration techniques enable surgical removal of refusal representations through directional orthogonalization, the relative effectiveness of available implementations remains uncharacterized. This study evaluates four abliteration tools (Heretic, DECCP, ErisForge, FailSpy) across sixteen instruction-tuned models (7B-14B parameters), reporting tool compatibility on all 16 models and quantitative metrics on subsets dictated by tool support. Single-pass methods demonstrated superior capability preservation on the benchmarked subset (avg GSM8K change across three models: ErisForge -0.28 pp; DECCP -0.13 pp), while Bayesian-optimized abliteration produced variable distribution shift (KL divergence: 0.043-1.646) with model-dependent capability impact. These findings provide researchers with evidence-based selection criteria for abliteration tool deployment across diverse model architectures. The principal finding indicates that mathematical reasoning capabilities exhibit the highest sensitivity to abliteration interventions, with GSM8K change ranging from +1.51 pp to -18.81 pp (-26.5% relative) depending on tool selection and model architecture. |
| 2025-12-15 | [Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models](http://arxiv.org/abs/2512.13607v1) | Boxin Wang, Chankyu Lee et al. | Building general-purpose reasoning models with reinforcement learning (RL) entails substantial cross-domain heterogeneity, including large variation in inference-time response lengths and verification latency. Such variability complicates the RL infrastructure, slows training, and makes training curriculum (e.g., response length extension) and hyperparameter selection challenging. In this work, we propose cascaded domain-wise reinforcement learning (Cascade RL) to develop general-purpose reasoning models, Nemotron-Cascade, capable of operating in both instruct and deep thinking modes. Departing from conventional approaches that blend heterogeneous prompts from different domains, Cascade RL orchestrates sequential, domain-wise RL, reducing engineering complexity and delivering state-of-the-art performance across a wide range of benchmarks. Notably, RLHF for alignment, when used as a pre-step, boosts the model's reasoning ability far beyond mere preference optimization, and subsequent domain-wise RLVR stages rarely degrade the benchmark performance attained in earlier domains and may even improve it (see an illustration in Figure 1). Our 14B model, after RL, outperforms its SFT teacher, DeepSeek-R1-0528, on LiveCodeBench v5/v6/Pro and achieves silver-medal performance in the 2025 International Olympiad in Informatics (IOI). We transparently share our training and data recipes. |
| 2025-12-15 | [Near-Field Perception for Safety Enhancement of Autonomous Mobile Robots in Manufacturing Environments](http://arxiv.org/abs/2512.13561v1) | Li-Wei Shih, Ruo-Syuan Mei et al. | Near-field perception is essential for the safe operation of autonomous mobile robots (AMRs) in manufacturing environments. Conventional ranging sensors such as light detection and ranging (LiDAR) and ultrasonic devices provide broad situational awareness but often fail to detect small objects near the robot base. To address this limitation, this paper presents a three-tier near-field perception framework. The first approach employs light-discontinuity detection, which projects a laser stripe across the near-field zone and identifies interruptions in the stripe to perform fast, binary cutoff sensing for obstacle presence. The second approach utilizes light-displacement measurement to estimate object height by analyzing the geometric displacement of a projected stripe in the camera image, which provides quantitative obstacle height information with minimal computational overhead. The third approach employs a computer vision-based object detection model on embedded AI hardware to classify objects, enabling semantic perception and context-aware safety decisions. All methods are implemented on a Raspberry Pi 5 system, achieving real-time performance at 25 or 50 frames per second. Experimental evaluation and comparative analysis demonstrate that the proposed hierarchy balances precision, computation, and cost, thereby providing a scalable perception solution for enabling safe operations of AMRs in manufacturing environments. |
| 2025-12-15 | [neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings](http://arxiv.org/abs/2512.13481v1) | Ojas Pungalia, Rashi Upadhyay et al. | Envy is a common human behavior that shapes competitiveness and can alter outcomes in team settings. As large language models (LLMs) increasingly act on behalf of humans in collaborative and competitive workflows, there is a pressing need to evaluate whether and under what conditions they exhibit envy-like preferences. In this paper, we test whether LLMs show envy-like behavior toward each other. We considered two scenarios: (1) A point allocation game that tests whether a model tries to win over its peer. (2) A workplace setting observing behaviour when recognition is unfair. Our findings reveal consistent evidence of envy-like patterns in certain LLMs, with large variation across models and contexts. For instance, GPT-5-mini and Claude-3.7-Sonnet show a clear tendency to pull down the peer model to equalize outcomes, whereas Mistral-Small-3.2-24B instead focuses on maximizing its own individual gains. These results highlight the need to consider competitive dispositions as a safety and design factor in LLM-based multi-agent systems. |
| 2025-12-15 | [A Data Annotation Requirements Representation and Specification (DARS)](http://arxiv.org/abs/2512.13444v1) | Yi Peng, Hina Saeeda et al. | With the rise of AI-enabled cyber-physical systems, data annotation has become a critical yet often overlooked process in the development of these intelligent information systems. Existing work in requirements engineering (RE) has explored how requirements for AI systems and their data can be represented. However, related interviews with industry professionals show that data annotations and their related requirements introduce distinct challenges, indicating a need for annotation-specific requirement representations. We propose the Data Annotation Requirements Representation and Specification (DARS), including an Annotation Negotiation Card to align stakeholders on objectives and constraints, and a Scenario-Based Annotation Specification to express atomic and verifiable data annotation requirements. We evaluate DARS with an automotive perception case related to an ongoing project, and a mapping against 18 real-world data annotation error types. The results suggest that DARS mitigates root causes of completeness, accuracy, and consistency annotation errors. By integrating DARS into RE, this work improves the reliability of safety-critical systems using data annotations and demonstrates how engineering frameworks must evolve for data-dependent components of today's intelligent information systems. |
| 2025-12-15 | [Stability-Drift Early Warning for Cyber-Physical Systems Under Degradation Attacks](http://arxiv.org/abs/2512.13767v1) | Daniyal Ganiuly, Nurzhau Bolatbek et al. | Cyber-physical systems (CPS) such as unmanned aerial vehicles are vulnerable to slow degradation that develops without causing immediate or obvious failures. Small sensor biases or timing irregularities can accumulate over time, gradually reducing stability while standard monitoring mechanisms continue to report normal operation. Detecting this early phase of degradation remains a challenge, as most existing approaches focus on abrupt faults or visible trajectory deviations. This paper introduces an early warning method based on stability drift, which measures the divergence between predicted and observed state transitions over short horizons. By tracking the gradual growth of this divergence, the proposed approach identifies emerging instability before it becomes visible in the flight trajectory or estimator residuals. The method operates externally to the flight stack and relies only on standard telemetry, making it suitable for deployment without modifying autopilot firmware. The approach was evaluated on a PX4 x500 platform in a software in the loop environment under two realistic degradation scenarios, gradual IMU bias drift and timing irregularities in the control loop. In both cases, the stability drift metric provided a consistent early warning signal several seconds before visible instability appeared, while remaining stable during nominal and aggressive but non degraded flight. The results demonstrate that stability drift can serve as a practical indicator of early degradation in UAV control systems. By providing advance notice during a pre instability phase, the proposed method complements existing safety mechanisms and offers additional time for mitigation or safe mode transitions under slow and subtle attacks. |

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



