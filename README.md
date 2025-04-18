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
| 2025-04-17 | [Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](http://arxiv.org/abs/2504.13169v1) | Tsung-Han Wu, Heekyung Lee et al. | Vision-Language Models (VLMs) excel at visual understanding but often suffer from visual hallucinations, where they generate descriptions of nonexistent objects, actions, or concepts, posing significant risks in safety-critical applications. Existing hallucination mitigation methods typically follow one of two paradigms: generation adjustment, which modifies decoding behavior to align text with visual inputs, and post-hoc verification, where external models assess and correct outputs. While effective, generation adjustment methods often rely on heuristics and lack correction mechanisms, while post-hoc verification is complicated, typically requiring multiple models and tending to reject outputs rather than refine them. In this work, we introduce REVERSE, a unified framework that integrates hallucination-aware training with on-the-fly self-verification. By leveraging a new hallucination-verification dataset containing over 1.3M semi-synthetic samples, along with a novel inference-time retrospective resampling technique, our approach enables VLMs to both detect hallucinations during generation and dynamically revise those hallucinations. Our evaluations show that REVERSE achieves state-of-the-art hallucination reduction, outperforming the best existing methods by up to 12% on CHAIR-MSCOCO and 28% on HaloQuest. Our dataset, model, and code are available at: https://reverse-vlm.github.io. |
| 2025-04-17 | [Energy-Based Reward Models for Robust Language Model Alignment](http://arxiv.org/abs/2504.13134v1) | Anamika Lochab, Ruqi Zhang | Reward models (RMs) are essential for aligning Large Language Models (LLMs) with human preferences. However, they often struggle with capturing complex human preferences and generalizing to unseen data. To address these challenges, we introduce Energy-Based Reward Model (EBRM), a lightweight post-hoc refinement framework that enhances RM robustness and generalization. EBRM models the reward distribution explicitly, capturing uncertainty in human preferences and mitigating the impact of noisy or misaligned annotations. It achieves this through conflict-aware data filtering, label-noise-aware contrastive training, and hybrid initialization. Notably, EBRM enhances RMs without retraining, making it computationally efficient and adaptable across different models and tasks. Empirical evaluations on RM benchmarks demonstrate significant improvements in both robustness and generalization, achieving up to a 5.97% improvement in safety-critical alignment tasks compared to standard RMs. Furthermore, reinforcement learning experiments confirm that our refined rewards enhance alignment quality, effectively delaying reward hacking. These results demonstrate our approach as a scalable and effective enhancement for existing RMs and alignment pipelines. The code is available at EBRM. |
| 2025-04-17 | [LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard](http://arxiv.org/abs/2504.13125v1) | Varun Rao, Youran Sun et al. | This paper investigates the application of large language models (LLMs) to financial tasks. We fine-tuned foundation models using the Open FinLLM Leaderboard as a benchmark. Building on Qwen2.5 and Deepseek-R1, we employed techniques including supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) to enhance their financial capabilities. The fine-tuned models demonstrated substantial performance gains across a wide range of financial tasks. Moreover, we measured the data scaling law in the financial domain. Our work demonstrates the potential of large language models (LLMs) in financial applications. |
| 2025-04-17 | [Accuracy is Not Agreement: Expert-Aligned Evaluation of Crash Narrative Classification Models](http://arxiv.org/abs/2504.13068v1) | Sudesh Ramesh Bhagat, Ibne Farabi Shihab et al. | This study explores the relationship between deep learning (DL) model accuracy and expert agreement in the classification of crash narratives. We evaluate five DL models -- including BERT variants, the Universal Sentence Encoder (USE), and a zero-shot classifier -- against expert-labeled data and narrative text. The analysis is further extended to four large language models (LLMs): GPT-4, LLaMA 3, Qwen, and Claude. Our results reveal a counterintuitive trend: models with higher technical accuracy often exhibit lower agreement with domain experts, whereas LLMs demonstrate greater expert alignment despite relatively lower accuracy scores. To quantify and interpret model-expert agreement, we employ Cohen's Kappa, Principal Component Analysis (PCA), and SHAP-based explainability techniques. Findings indicate that expert-aligned models tend to rely more on contextual and temporal language cues, rather than location-specific keywords. These results underscore that accuracy alone is insufficient for evaluating models in safety-critical NLP applications. We advocate for incorporating expert agreement as a complementary metric in model evaluation frameworks and highlight the promise of LLMs as interpretable, scalable tools for crash analysis pipelines. |
| 2025-04-17 | [GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms](http://arxiv.org/abs/2504.13052v1) | Sinan He, An Wang | Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms. |
| 2025-04-17 | [QI-MPC: A Hybrid Quantum-Inspired Model Predictive Control for Learning Optimal Policies](http://arxiv.org/abs/2504.13041v1) | Muhammad Al-Zafar Khan, Jamal Al-Karaki | In this paper, we present Quantum-Inspired Model Predictive Control (QIMPC), an approach that uses Variational Quantum Circuits (VQCs) to learn control polices in MPC problems. The viability of the approach is tested in five experiments: A target-tracking control strategy, energy-efficient building climate control, autonomous vehicular dynamics, the simple pendulum, and the compound pendulum. Three safety guarantees were established for the approach, and the experiments gave the motivation for two important theoretical results that, in essence, identify systems for which the approach works best. |
| 2025-04-17 | [Safe Physics-Informed Machine Learning for Dynamics and Control](http://arxiv.org/abs/2504.12952v1) | Jan Drgona, Truong X. Nghiem et al. | This tutorial paper focuses on safe physics-informed machine learning in the context of dynamics and control, providing a comprehensive overview of how to integrate physical models and safety guarantees. As machine learning techniques enhance the modeling and control of complex dynamical systems, ensuring safety and stability remains a critical challenge, especially in safety-critical applications like autonomous vehicles, robotics, medical decision-making, and energy systems. We explore various approaches for embedding and ensuring safety constraints, such as structural priors, Lyapunov functions, Control Barrier Functions, predictive control, projections, and robust optimization techniques, ensuring that the learned models respect stability and safety criteria. Additionally, we delve into methods for uncertainty quantification and safety verification, including reachability analysis and neural network verification tools, which help validate that control policies remain within safe operating bounds even in uncertain environments. The paper includes illustrative examples demonstrating the implementation aspects of safe learning frameworks that combine the strengths of data-driven approaches with the rigor of physical principles, offering a path toward the safe control of complex dynamical systems. |
| 2025-04-17 | [In Which Areas of Technical AI Safety Could Geopolitical Rivals Cooperate?](http://arxiv.org/abs/2504.12914v1) | Ben Bucknall, Saad Siddiqui et al. | International cooperation is common in AI research, including between geopolitical rivals. While many experts advocate for greater international cooperation on AI safety to address shared global risks, some view cooperation on AI with suspicion, arguing that it can pose unacceptable risks to national security. However, the extent to which cooperation on AI safety poses such risks, as well as provides benefits, depends on the specific area of cooperation. In this paper, we consider technical factors that impact the risks of international cooperation on AI safety research, focusing on the degree to which such cooperation can advance dangerous capabilities, result in the sharing of sensitive information, or provide opportunities for harm. We begin by why nations historically cooperate on strategic technologies and analyse current US-China cooperation in AI as a case study. We further argue that existing frameworks for managing associated risks can be supplemented with consideration of key risks specific to cooperation on technical AI safety research. Through our analysis, we find that research into AI verification mechanisms and shared protocols may be suitable areas for such cooperation. Through this analysis we aim to help researchers and governments identify and mitigate the risks of international cooperation on AI safety research, so that the benefits of cooperation can be fully realised. |
| 2025-04-17 | [UncAD: Towards Safe End-to-end Autonomous Driving via Online Map Uncertainty](http://arxiv.org/abs/2504.12826v1) | Pengxuan Yang, Yupeng Zheng et al. | End-to-end autonomous driving aims to produce planning trajectories from raw sensors directly. Currently, most approaches integrate perception, prediction, and planning modules into a fully differentiable network, promising great scalability. However, these methods typically rely on deterministic modeling of online maps in the perception module for guiding or constraining vehicle planning, which may incorporate erroneous perception information and further compromise planning safety. To address this issue, we delve into the importance of online map uncertainty for enhancing autonomous driving safety and propose a novel paradigm named UncAD. Specifically, UncAD first estimates the uncertainty of the online map in the perception module. It then leverages the uncertainty to guide motion prediction and planning modules to produce multi-modal trajectories. Finally, to achieve safer autonomous driving, UncAD proposes an uncertainty-collision-aware planning selection strategy according to the online map uncertainty to evaluate and select the best trajectory. In this study, we incorporate UncAD into various state-of-the-art (SOTA) end-to-end methods. Experiments on the nuScenes dataset show that integrating UncAD, with only a 1.9% increase in parameters, can reduce collision rates by up to 26% and drivable area conflict rate by up to 42%. Codes, pre-trained models, and demo videos can be accessed at https://github.com/pengxuanyang/UncAD. |
| 2025-04-17 | [Supporting Urban Low-Altitude Economy: Channel Gain Map Inference Based on 3D Conditional GAN](http://arxiv.org/abs/2504.12794v1) | Yonghao Wang, Ruoguang Li et al. | The advancement of advanced air mobility (AAM) in recent years has given rise to the concept of low-altitude economy (LAE). However, the diverse flight activities associated with the emerging LAE applications in urban scenarios confront complex physical environments, which urgently necessitates ubiquitous and reliable communication to guarantee the operation safety of the low-altitude aircraft. As one of promising technologies for the sixth generation (6G) mobile networks, channel knowledge map (CKM) enables the environment-aware communication by constructing a site-specific dataset, thereby providing a priori on-site information for the aircraft to obtain the channel state information (CSI) at arbitrary locations with much reduced online overhead. Diverse base station (BS) deployments in the three-dimensional (3D) urban low-altitude environment require efficient 3D CKM construction to capture spatial channel characteristics with less overhead. Towards this end, this paper proposes a 3D channel gain map (CGM) inference method based on a 3D conditional generative adversarial network (3D-CGAN). Specifically, we first analyze the potential deployment types of BSs in urban low-altitude scenario, and investigate the CGM representation with the corresponding 3D channel gain model. The framework of the proposed 3D-CGAN is then discussed, which is trained by a dataset consisting of existing CGMs. Consequently, the trained 3D-CGAN is capable of inferring the corresponding CGM only based on the BS coordinate without additional measurement. The simulation results demonstrate that the CGMs inferred by the proposed 3D-CGAN outperform those of the benchmark schemes, which can accurately reflect the radio propagation condition in 3D environment. |
| 2025-04-17 | [On Error Classification from Physiological Signals within Airborne Environment](http://arxiv.org/abs/2504.12769v1) | Niall McGuire, Yashar Moshfeghi | Human error remains a critical concern in aviation safety, contributing to 70-80% of accidents despite technological advancements. While physiological measures show promise for error detection in laboratory settings, their effectiveness in dynamic flight environments remains underexplored. Through live flight trials with nine commercial pilots, we investigated whether established error-detection approaches maintain accuracy during actual flight operations. Participants completed standardized multi-tasking scenarios across conditions ranging from laboratory settings to straight-and-level flight and 2G manoeuvres while we collected synchronized physiological data. Our findings demonstrate that EEG-based classification maintains high accuracy (87.83%) during complex flight manoeuvres, comparable to laboratory performance (89.23%). Eye-tracking showed moderate performance (82.50\%), while ECG performed near chance level (51.50%). Classification accuracy remained stable across flight conditions, with minimal degradation during 2G manoeuvres. These results provide the first evidence that physiological error detection can translate effectively to operational aviation environments. |
| 2025-04-17 | [Falcon: Advancing Asynchronous BFT Consensus for Lower Latency and Enhanced Throughput](http://arxiv.org/abs/2504.12766v1) | Xiaohai Dai, Chaozheng Ding et al. | Asynchronous Byzantine Fault Tolerant (BFT) consensus protocols have garnered significant attention with the rise of blockchain technology. A typical asynchronous protocol is designed by executing sequential instances of the Asynchronous Common Sub-seQuence (ACSQ). The ACSQ protocol consists of two primary components: the Asynchronous Common Subset (ACS) protocol and a block sorting mechanism, with the ACS protocol comprising two stages: broadcast and agreement. However, current protocols encounter three critical issues: high latency arising from the execution of the agreement stage, latency instability due to the integral-sorting mechanism, and reduced throughput caused by block discarding. To address these issues,we propose Falcon, an asynchronous BFT protocol that achieves low latency and enhanced throughput. Falcon introduces a novel broadcast protocol, Graded Broadcast (GBC), which enables a block to be included in the ACS set directly, bypassing the agreement stage and thereby reducing latency. To ensure safety, Falcon incorporates a new binary agreement protocol called Asymmetrical Asynchronous Binary Agreement (AABA), designed to complement GBC. Additionally, Falcon employs a partial-sorting mechanism, allowing continuous rather than simultaneous block committing, enhancing latency stability. Finally, we incorporate an agreement trigger that, before its activation, enables nodes to wait for more blocks to be delivered and committed, thereby boosting throughput. We conduct a series of experiments to evaluate Falcon, demonstrating its superior performance. |
| 2025-04-17 | [Distributed Intelligent Sensing and Communications for 6G: Architecture and Use Cases](http://arxiv.org/abs/2504.12765v1) | Kyriakos Stylianopoulos, Giyyarpuram Madhusudan et al. | The Distributed Intelligent Sensing and Communication (DISAC) framework redefines Integrated Sensing and Communication (ISAC) for 6G by leveraging distributed architectures to enhance scalability, adaptability, and resource efficiency. This paper presents key architectural enablers, including advanced data representation, seamless target handover, support for heterogeneous devices, and semantic integration. Two use cases illustrate the transformative potential of DISAC: smart factory shop floors and Vulnerable Road User (VRU) protection at smart intersections. These scenarios demonstrate significant improvements in precision, safety, and operational efficiency compared to traditional ISAC systems. The preliminary DISAC architecture incorporates intelligent data processing, distributed coordination, and emerging technologies such as Reconfigurable Intelligent Surfaces (RIS) to meet 6G's stringent requirements. By addressing critical challenges in sensing accuracy, latency, and real-time decision-making, DISAC positions itself as a cornerstone for next-generation wireless networks, advancing innovation in dynamic and complex environments. |
| 2025-04-17 | [Incorporating a Deep Neural Network into Moving Horizon Estimation for Embedded Thermal Torque Derating of an Electric Machine](http://arxiv.org/abs/2504.12736v1) | Alexander Winkler, Pranav Shah et al. | This study introduces a novel state estimation framework that incorporates Deep Neural Networks (DNNs) into Moving Horizon Estimation (MHE), shifting from traditional physics-based models to rapidly developed data-driven techniques. A DNN model with Long Short-Term Memory (LSTM) nodes is trained on synthetic data generated by a high-fidelity thermal model of a Permanent Magnet Synchronous Machine (PMSM), which undergoes thermal derating as part of the torque control strategy in a battery electric vehicle. The MHE is constructed by integrating the trained DNN with a simplified driving dynamics model in a discrete-time formulation, incorporating the LSTM hidden and cell states in the state vector to retain system dynamics. The resulting optimal control problem (OCP) is formulated as a nonlinear program (NLP) and implemented using the acados framework. Model-in-the-loop (MiL) simulations demonstrate accurate temperature estimation, even under noisy sensor conditions or failures. Achieving threefold real-time capability on embedded hardware confirms the feasibility of the approach for practical deployment. The primary focus of this study is to assess the feasibility of the MHE framework using a DNN-based plant model instead of focusing on quantitative comparisons of vehicle performance. Overall, this research highlights the potential of DNN-based MHE for real-time, safety-critical applications by combining the strengths of model-based and data-driven methods. |
| 2025-04-17 | [Collaborative Perception Datasets for Autonomous Driving: A Review](http://arxiv.org/abs/2504.12696v1) | Naibang Wang, Deyong Shang et al. | Collaborative perception has attracted growing interest from academia and industry due to its potential to enhance perception accuracy, safety, and robustness in autonomous driving through multi-agent information fusion. With the advancement of Vehicle-to-Everything (V2X) communication, numerous collaborative perception datasets have emerged, varying in cooperation paradigms, sensor configurations, data sources, and application scenarios. However, the absence of systematic summarization and comparative analysis hinders effective resource utilization and standardization of model evaluation. As the first comprehensive review focused on collaborative perception datasets, this work reviews and compares existing resources from a multi-dimensional perspective. We categorize datasets based on cooperation paradigms, examine their data sources and scenarios, and analyze sensor modalities and supported tasks. A detailed comparative analysis is conducted across multiple dimensions. We also outline key challenges and future directions, including dataset scalability, diversity, domain adaptation, standardization, privacy, and the integration of large language models. To support ongoing research, we provide a continuously updated online repository of collaborative perception datasets and related literature: https://github.com/frankwnb/Collaborative-Perception-Datasets-for-Autonomous-Driving. |
| 2025-04-17 | [Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning](http://arxiv.org/abs/2504.12680v1) | Baining Zhao, Ziyou Wang et al. | Humans can perceive and reason about spatial relationships from sequential visual observations, such as egocentric video streams. However, how pretrained models acquire such abilities, especially high-level reasoning, remains unclear. This paper introduces Embodied-R, a collaborative framework combining large-scale Vision-Language Models (VLMs) for perception and small-scale Language Models (LMs) for reasoning. Using Reinforcement Learning (RL) with a novel reward system considering think-answer logical consistency, the model achieves slow-thinking capabilities with limited computational resources. After training on only 5k embodied video samples, Embodied-R with a 3B LM matches state-of-the-art multimodal reasoning models (OpenAI-o1, Gemini-2.5-pro) on both in-distribution and out-of-distribution embodied spatial reasoning tasks. Embodied-R also exhibits emergent thinking patterns such as systematic analysis and contextual integration. We further explore research questions including response length, training on VLM, strategies for reward design, and differences in model generalization after SFT (Supervised Fine-Tuning) and RL training. |
| 2025-04-17 | [Predicting Driver's Perceived Risk: a Model Based on Semi-Supervised Learning Strategy](http://arxiv.org/abs/2504.12665v1) | Siwei Huang, Chenhao Yang et al. | Drivers' perception of risk determines their acceptance, trust, and use of the Automated Driving Systems (ADSs). However, perceived risk is subjective and difficult to evaluate using existing methods. To address this issue, a driver's subjective perceived risk (DSPR) model is proposed, regarding perceived risk as a dynamically triggered mechanism with anisotropy and attenuation. 20 participants are recruited for a driver-in-the-loop experiment to report their real-time subjective risk ratings (SRRs) when experiencing various automatic driving scenarios. A convolutional neural network and bidirectional long short-term memory network with temporal pattern attention (CNN-Bi-LSTM-TPA) is embedded into a semi-supervised learning strategy to predict SRRs, aiming to reduce data noise caused by subjective randomness of participants. The results illustrate that DSPR achieves the highest prediction accuracy of 87.91% in predicting SRRs, compared to three state-of-the-art risk models. The semi-supervised strategy improves accuracy by 20.12%. Besides, CNN-Bi-LSTM-TPA network presents the highest accuracy among four different LSTM structures. This study offers an effective method for assessing driver's perceived risk, providing support for the safety enhancement of ADS and driver's trust improvement. |
| 2025-04-17 | [VLMGuard-R1: Proactive Safety Alignment for VLMs via Reasoning-Driven Prompt Optimization](http://arxiv.org/abs/2504.12661v1) | Menglan Chen, Xianghe Pang et al. | Aligning Vision-Language Models (VLMs) with safety standards is essential to mitigate risks arising from their multimodal complexity, where integrating vision and language unveils subtle threats beyond the reach of conventional safeguards. Inspired by the insight that reasoning across modalities is key to preempting intricate vulnerabilities, we propose a novel direction for VLM safety: multimodal reasoning-driven prompt rewriting. To this end, we introduce VLMGuard-R1, a proactive framework that refines user inputs through a reasoning-guided rewriter, dynamically interpreting text-image interactions to deliver refined prompts that bolster safety across diverse VLM architectures without altering their core parameters. To achieve this, we devise a three-stage reasoning pipeline to synthesize a dataset that trains the rewriter to infer subtle threats, enabling tailored, actionable responses over generic refusals. Extensive experiments across three benchmarks with five VLMs reveal that VLMGuard-R1 outperforms four baselines. In particular, VLMGuard-R1 achieves a remarkable 43.59\% increase in average safety across five models on the SIUO benchmark. |
| 2025-04-17 | [Photon Calibration Performance of KAGRA during the 4th Joint Observing Run (O4)](http://arxiv.org/abs/2504.12657v1) | Dan Chen, Shingo Hido et al. | KAGRA is a kilometer-scale cryogenic gravitational-wave (GW) detector in Japan. It joined the 4th joint observing run (O4) in May 2023 in collaboration with the Laser Interferometer GW Observatory (LIGO) in the USA, and Virgo in Italy. After one month of observations, KAGRA entered a break period to enhance its sensitivity to GWs, and it is planned to rejoin O4 before its scheduled end in October 2025. To accurately recover the information encoded in the GW signals, it is essential to properly calibrate the observed signals. We employ a photon calibration (Pcal) system as a reference signal injector to calibrate the output signals obtained from the telescope. In ideal future conditions, the uncertainty in Pcal could dominate the uncertainty in the observed data. In this paper, we present the methods used to estimate the uncertainty in the Pcal systems employed during KAGRA O4 and report an estimated system uncertainty of 0.79%, which is three times lower than the uncertainty achieved in the previous 3rd joint observing run (O3) in 2020. Additionally, we investigate the uncertainty in the Pcal laser power sensors, which had the highest impact on the Pcal uncertainty, and estimate the beam positions on the KAGRA main mirror, which had the second highest impact. The Pcal systems in KAGRA are the first fully functional calibration systems for a cryogenic GW telescope. To avoid interference with the KAGRA cryogenic systems, the Pcal systems incorporate unique features regarding their placement and the use of telephoto cameras, which can capture images of the mirror surface at almost normal incidence. As future GW telescopes, such as the Einstein Telescope, are expected to adopt cryogenic techniques, the performance of the KAGRA Pcal systems can serve as a valuable reference. |
| 2025-04-17 | [Graph-based Path Planning with Dynamic Obstacle Avoidance for Autonomous Parking](http://arxiv.org/abs/2504.12616v1) | Farhad Nawaz, Minjun Sung et al. | Safe and efficient path planning in parking scenarios presents a significant challenge due to the presence of cluttered environments filled with static and dynamic obstacles. To address this, we propose a novel and computationally efficient planning strategy that seamlessly integrates the predictions of dynamic obstacles into the planning process, ensuring the generation of collision-free paths. Our approach builds upon the conventional Hybrid A star algorithm by introducing a time-indexed variant that explicitly accounts for the predictions of dynamic obstacles during node exploration in the graph, thus enabling dynamic obstacle avoidance. We integrate the time-indexed Hybrid A star algorithm within an online planning framework to compute local paths at each planning step, guided by an adaptively chosen intermediate goal. The proposed method is validated in diverse parking scenarios, including perpendicular, angled, and parallel parking. Through simulations, we showcase our approach's potential in greatly improving the efficiency and safety when compared to the state of the art spline-based planning method for parking situations. |

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



