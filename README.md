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
| 2025-07-24 | [Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment](http://arxiv.org/abs/2507.18631v1) | Hao Li, Lijun Li et al. | With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a \textbf{L}ayer-\textbf{A}ware \textbf{R}epresentation \textbf{F}iltering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at \href{https://github.com/LLLeoLi/LARF}{https://github.com/LLLeoLi/LARF}. |
| 2025-07-24 | [SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law](http://arxiv.org/abs/2507.18576v1) | Shanghai AI Lab, : et al. | We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI. |
| 2025-07-24 | [Synthetic Data Augmentation for Enhanced Chicken Carcass Instance Segmentation](http://arxiv.org/abs/2507.18558v1) | Yihong Feng, Chaitanya Pallerla et al. | The poultry industry has been driven by broiler chicken production and has grown into the world's largest animal protein sector. Automated detection of chicken carcasses on processing lines is vital for quality control, food safety, and operational efficiency in slaughterhouses and poultry processing plants. However, developing robust deep learning models for tasks like instance segmentation in these fast-paced industrial environments is often hampered by the need for laborious acquisition and annotation of large-scale real-world image datasets. We present the first pipeline generating photo-realistic, automatically labeled synthetic images of chicken carcasses. We also introduce a new benchmark dataset containing 300 annotated real-world images, curated specifically for poultry segmentation research. Using these datasets, this study investigates the efficacy of synthetic data and automatic data annotation to enhance the instance segmentation of chicken carcasses, particularly when real annotated data from the processing line is scarce. A small real dataset with varying proportions of synthetic images was evaluated in prominent instance segmentation models. Results show that synthetic data significantly boosts segmentation performance for chicken carcasses across all models. This research underscores the value of synthetic data augmentation as a viable and effective strategy to mitigate data scarcity, reduce manual annotation efforts, and advance the development of robust AI-driven automated detection systems for chicken carcasses in the poultry processing industry. |
| 2025-07-24 | [Reinforced Embodied Active Defense: Exploiting Adaptive Interaction for Robust Visual Perception in Adversarial 3D Environments](http://arxiv.org/abs/2507.18484v1) | Xiao Yang, Lingxuan Wu et al. | Adversarial attacks in 3D environments have emerged as a critical threat to the reliability of visual perception systems, particularly in safety-sensitive applications such as identity verification and autonomous driving. These attacks employ adversarial patches and 3D objects to manipulate deep neural network (DNN) predictions by exploiting vulnerabilities within complex scenes. Existing defense mechanisms, such as adversarial training and purification, primarily employ passive strategies to enhance robustness. However, these approaches often rely on pre-defined assumptions about adversarial tactics, limiting their adaptability in dynamic 3D settings. To address these challenges, we introduce Reinforced Embodied Active Defense (Rein-EAD), a proactive defense framework that leverages adaptive exploration and interaction with the environment to improve perception robustness in 3D adversarial contexts. By implementing a multi-step objective that balances immediate prediction accuracy with predictive entropy minimization, Rein-EAD optimizes defense strategies over a multi-step horizon. Additionally, Rein-EAD involves an uncertainty-oriented reward-shaping mechanism that facilitates efficient policy updates, thereby reducing computational overhead and supporting real-world applicability without the need for differentiable environments. Comprehensive experiments validate the effectiveness of Rein-EAD, demonstrating a substantial reduction in attack success rates while preserving standard accuracy across diverse tasks. Notably, Rein-EAD exhibits robust generalization to unseen and adaptive attacks, making it suitable for real-world complex tasks, including 3D object classification, face recognition and autonomous driving. |
| 2025-07-24 | [PDB-Eval: An Evaluation of Large Multimodal Models for Description and Explanation of Personalized Driving Behavior](http://arxiv.org/abs/2507.18447v1) | Junda Wu, Jessica Echterhoff et al. | Understanding a driver's behavior and intentions is important for potential risk assessment and early accident prevention. Safety and driver assistance systems can be tailored to individual drivers' behavior, significantly enhancing their effectiveness. However, existing datasets are limited in describing and explaining general vehicle movements based on external visual evidence. This paper introduces a benchmark, PDB-Eval, for a detailed understanding of Personalized Driver Behavior, and aligning Large Multimodal Models (MLLMs) with driving comprehension and reasoning. Our benchmark consists of two main components, PDB-X and PDB-QA. PDB-X can evaluate MLLMs' understanding of temporal driving scenes. Our dataset is designed to find valid visual evidence from the external view to explain the driver's behavior from the internal view. To align MLLMs' reasoning abilities with driving tasks, we propose PDB-QA as a visual explanation question-answering task for MLLM instruction fine-tuning. As a generic learning task for generative models like MLLMs, PDB-QA can bridge the domain gap without harming MLLMs' generalizability. Our evaluation indicates that fine-tuning MLLMs on fine-grained descriptions and explanations can effectively bridge the gap between MLLMs and the driving domain, which improves zero-shot performance on question-answering tasks by up to 73.2%. We further evaluate the MLLMs fine-tuned on PDB-X in Brain4Cars' intention prediction and AIDE's recognition tasks. We observe up to 12.5% performance improvements on the turn intention prediction task in Brain4Cars, and consistent performance improvements up to 11.0% on all tasks in AIDE. |
| 2025-07-24 | [Residual Koopman Model Predictive Control for Enhanced Vehicle Dynamics with Small On-Track Data Input](http://arxiv.org/abs/2507.18396v1) | Yonghao Fu, Cheng Hu et al. | In vehicle trajectory tracking tasks, the simplest approach is the Pure Pursuit (PP) Control. However, this single-point preview tracking strategy fails to consider vehicle model constraints, compromising driving safety. Model Predictive Control (MPC) as a widely adopted control method, optimizes control actions by incorporating mechanistic models and physical constraints. While its control performance critically depends on the accuracy of vehicle modeling. Traditional vehicle modeling approaches face inherent trade-offs between capturing nonlinear dynamics and maintaining computational efficiency, often resulting in reduced control performance. To address these challenges, this paper proposes Residual Koopman Model Predictive Control (RKMPC) framework. This method uses two linear MPC architecture to calculate control inputs: a Linear Model Predictive Control (LMPC) computes the baseline control input based on the vehicle kinematic model, and a neural network-based RKMPC calculates the compensation input. The final control command is obtained by adding these two components. This design preserves the reliability and interpretability of traditional mechanistic model while achieving performance optimization through residual modeling. This method has been validated on the Carsim-Matlab joint simulation platform and a physical 1:10 scale F1TENTH racing car. Experimental results show that RKMPC requires only 20% of the training data needed by traditional Koopman Model Predictive Control (KMPC) while delivering superior tracking performance. Compared to traditional LMPC, RKMPC reduces lateral error by 11.7%-22.1%, decreases heading error by 8.9%-15.8%, and improves front-wheel steering stability by up to 27.6%. The implementation code is available at: https://github.com/ZJU-DDRX/Residual Koopman. |
| 2025-07-24 | [Reasoning Beyond the Obvious: Evaluating Divergent and Convergent Thinking in LLMs for Financial Scenarios](http://arxiv.org/abs/2507.18368v1) | Zhuang Qiang Bok, Watson Wei Khong Chua | Most reasoning benchmarks for LLMs emphasize factual accuracy or step-by-step logic. In finance, however, professionals must not only converge on optimal decisions but also generate creative, plausible futures under uncertainty. We introduce ConDiFi, a benchmark that jointly evaluates divergent and convergent thinking in LLMs for financial tasks.   ConDiFi features 607 macro-financial prompts for divergent reasoning and 990 multi-hop adversarial MCQs for convergent reasoning. Using this benchmark, we evaluated 14 leading models and uncovered striking differences. Despite high fluency, GPT-4o underperforms on Novelty and Actionability. In contrast, models like DeepSeek-R1 and Cohere Command R+ rank among the top for generating actionable, insights suitable for investment decisions. ConDiFi provides a new perspective to assess reasoning capabilities essential to safe and strategic deployment of LLMs in finance. |
| 2025-07-24 | [Enhanced Velocity-Adaptive Scheme: Joint Fair Access and Age of Information Optimization in Vehicular Networks](http://arxiv.org/abs/2507.18328v1) | Xiao Xu, Qiong Wu et al. | In this paper, we consider the fair access problem and the Age of Information (AoI) under 5G New Radio (NR) Vehicle-to-Infrastructure (V2I) Mode 2 in vehicular networks. Specifically, vehicles follow Mode 2 to communicate with Roadside Units (RSUs) to obtain accurate data for driving assistance.Nevertheless, vehicles often have different velocity when they are moving in adjacent lanes, leading to difference in RSU dwelltime and communication duration. This results in unfair access to network resources, potentially influencing driving safety. To ensure the freshness of received data, the AoI should be analyzed. Mode 2 introduces a novel preemption mechanism, necessitating simultaneous optimization of fair access and AoI to guarantee timely and relevant data delivery. We propose a joint optimization framework for vehicular network, defining a fairness index and employing Stochastic Hybrid Systems (SHS) to model AoI under preemption mechanism. By adaptively adjusting the selection window of Semi-Persistent Scheduling (SPS) in Mode 2, we address the optimization of fairness and AoI. We apply a large language model (LLM)-Based Multi-objective Evolutionary Algorithm Based on Decomposition (MOEA/D) to solve this problem. Simulation results demonstrate the effectiveness of our scheme in balancing fair access and minimizing AoI. |
| 2025-07-24 | [A Concept for Efficient Scalability of Automated Driving Allowing for Technical, Legal, Cultural, and Ethical Differences](http://arxiv.org/abs/2507.18326v1) | Lars Ullrich, Michael Buchholz et al. | Efficient scalability of automated driving (AD) is key to reducing costs, enhancing safety, conserving resources, and maximizing impact. However, research focuses on specific vehicles and context, while broad deployment requires scalability across various configurations and environments. Differences in vehicle types, sensors, actuators, but also traffic regulations, legal requirements, cultural dynamics, or even ethical paradigms demand high flexibility of data-driven developed capabilities. In this paper, we address the challenge of scalable adaptation of generic capabilities to desired systems and environments. Our concept follows a two-stage fine-tuning process. In the first stage, fine-tuning to the specific environment takes place through a country-specific reward model that serves as an interface between technological adaptations and socio-political requirements. In the second stage, vehicle-specific transfer learning facilitates system adaptation and governs the validation of design decisions. In sum, our concept offers a data-driven process that integrates both technological and socio-political aspects, enabling effective scalability across technical, legal, cultural, and ethical differences. |
| 2025-07-24 | [State of Health Estimation of Batteries Using a Time-Informed Dynamic Sequence-Inverted Transformer](http://arxiv.org/abs/2507.18320v1) | Janak M. Patel, Milad Ramezankhani et al. | The rapid adoption of battery-powered vehicles and energy storage systems over the past decade has made battery health monitoring increasingly critical. Batteries play a central role in the efficiency and safety of these systems, yet they inevitably degrade over time due to repeated charge-discharge cycles. This degradation leads to reduced energy efficiency and potential overheating, posing significant safety concerns. Accurate estimation of a State of Health (SoH) of battery is therefore essential for ensuring operational reliability and safety. Several machine learning architectures, such as LSTMs, transformers, and encoder-based models, have been proposed to estimate SoH from discharge cycle data. However, these models struggle with the irregularities inherent in real-world measurements: discharge readings are often recorded at non-uniform intervals, and the lengths of discharge cycles vary significantly. To address this, most existing approaches extract features from the sequences rather than processing them in full, which introduces information loss and compromises accuracy. To overcome these challenges, we propose a novel architecture: Time-Informed Dynamic Sequence Inverted Transformer (TIDSIT). TIDSIT incorporates continuous time embeddings to effectively represent irregularly sampled data and utilizes padded sequences with temporal attention mechanisms to manage variable-length inputs without discarding sequence information. Experimental results on the NASA battery degradation dataset show that TIDSIT significantly outperforms existing models, achieving over 50% reduction in prediction error and maintaining an SoH prediction error below 0.58%. Furthermore, the architecture is generalizable and holds promise for broader applications in health monitoring tasks involving irregular time-series data. |
| 2025-07-24 | [AF-RLIO: Adaptive Fusion of Radar-LiDAR-Inertial Information for Robust Odometry in Challenging Environments](http://arxiv.org/abs/2507.18317v1) | Chenglong Qian, Yang Xu et al. | In robotic navigation, maintaining precise pose estimation and navigation in complex and dynamic environments is crucial. However, environmental challenges such as smoke, tunnels, and adverse weather can significantly degrade the performance of single-sensor systems like LiDAR or GPS, compromising the overall stability and safety of autonomous robots. To address these challenges, we propose AF-RLIO: an adaptive fusion approach that integrates 4D millimeter-wave radar, LiDAR, inertial measurement unit (IMU), and GPS to leverage the complementary strengths of these sensors for robust odometry estimation in complex environments. Our method consists of three key modules. Firstly, the pre-processing module utilizes radar data to assist LiDAR in removing dynamic points and determining when environmental conditions are degraded for LiDAR. Secondly, the dynamic-aware multimodal odometry selects appropriate point cloud data for scan-to-map matching and tightly couples it with the IMU using the Iterative Error State Kalman Filter. Lastly, the factor graph optimization module balances weights between odometry and GPS data, constructing a pose graph for optimization. The proposed approach has been evaluated on datasets and tested in real-world robotic environments, demonstrating its effectiveness and advantages over existing methods in challenging conditions such as smoke and tunnels. |
| 2025-07-24 | [Surface Material Dependence in Powder Triboelectric Charging](http://arxiv.org/abs/2507.18193v1) | Tom F. O'Hara, Ellen Player et al. | Triboelectric charging of granular materials against container walls is a critical yet poorly understood phenomenon affecting many industrial powder handling processes. Charge accumulation can cause material flow disruptions, adhesion issues, and pose serious safety risks, such as providing ignition sources for dust cloud explosions. This study investigates particle-wall charging behaviour for materials with varying size, shape, and electrical resistivity. Aluminium surfaces are used as a reference case, followed by analysis of labradorite, an analogue for volcanic ash, to examine charging interactions with various wall materials. Finally, the triboelectric response of industrially relevant materials used in flexible intermediate bulk containers, including both lined and unlined variants, is assessed. Results show that stainless steel surfaces generate the least charge, while the presence of insulating polyethene liners increases charging significantly. These findings contribute to a deeper understanding of triboelectric charging mechanisms in powder transfer operations and inform safer industrial practice. |
| 2025-07-24 | [Actively evaluating and learning the distinctions that matter: Vaccine safety signal detection from emergency triage notes](http://arxiv.org/abs/2507.18123v1) | Sedigh Khademi, Christopher Palmer et al. | The rapid development of COVID-19 vaccines has showcased the global communitys ability to combat infectious diseases. However, the need for post-licensure surveillance systems has grown due to the limited window for safety data collection in clinical trials and early widespread implementation. This study aims to employ Natural Language Processing techniques and Active Learning to rapidly develop a classifier that detects potential vaccine safety issues from emergency department notes. ED triage notes, containing expert, succinct vital patient information at the point of entry to health systems, can significantly contribute to timely vaccine safety signal surveillance. While keyword-based classification can be effective, it may yield false positives and demand extensive keyword modifications. This is exacerbated by the infrequency of vaccination-related ED presentations and their similarity to other reasons for ED visits. NLP offers a more accurate and efficient alternative, albeit requiring annotated data, which is often scarce in the medical field. Active learning optimizes the annotation process and the quality of annotated data, which can result in faster model implementation and improved model performance. This work combines active learning, data augmentation, and active learning and evaluation techniques to create a classifier that is used to enhance vaccine safety surveillance from ED triage notes. |
| 2025-07-24 | [Multiscale Neural PDE Surrogates for Prediction and Downscaling: Application to Ocean Currents](http://arxiv.org/abs/2507.18067v1) | Abdessamad El-Kabid, Loubna Benabbou et al. | Accurate modeling of physical systems governed by partial differential equations is a central challenge in scientific computing. In oceanography, high-resolution current data are critical for coastal management, environmental monitoring, and maritime safety. However, available satellite products, such as Copernicus data for sea water velocity at ~0.08 degrees spatial resolution and global ocean models, often lack the spatial granularity required for detailed local analyses. In this work, we (a) introduce a supervised deep learning framework based on neural operators for solving PDEs and providing arbitrary resolution solutions, and (b) propose downscaling models with an application to Copernicus ocean current data. Additionally, our method can model surrogate PDEs and predict solutions at arbitrary resolution, regardless of the input resolution. We evaluated our model on real-world Copernicus ocean current data and synthetic Navier-Stokes simulation datasets. |
| 2025-07-24 | [Technical Report of TeleChat2, TeleChat2.5 and T1](http://arxiv.org/abs/2507.18013v1) | Zihan Wang, Xinzhang Liu et al. | We introduce the latest series of TeleChat models: \textbf{TeleChat2}, \textbf{TeleChat2.5}, and \textbf{T1}, offering a significant upgrade over their predecessor, TeleChat. Despite minimal changes to the model architecture, the new series achieves substantial performance gains through enhanced training strategies in both pre-training and post-training stages. The series begins with \textbf{TeleChat2}, which undergoes pretraining on 10 trillion high-quality and diverse tokens. This is followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) to further enhance its capabilities. \textbf{TeleChat2.5} and \textbf{T1} expand the pipeline by incorporating a continual pretraining phase with domain-specific datasets, combined with reinforcement learning (RL) to improve performance in code generation and mathematical reasoning tasks. The \textbf{T1} variant is designed for complex reasoning, supporting long Chain-of-Thought (CoT) reasoning and demonstrating substantial improvements in mathematics and coding. In contrast, \textbf{TeleChat2.5} prioritizes speed, delivering rapid inference. Both flagship models of \textbf{T1} and \textbf{TeleChat2.5} are dense Transformer-based architectures with 115B parameters, showcasing significant advancements in reasoning and general task performance compared to the original TeleChat. Notably, \textbf{T1-115B} outperform proprietary models such as OpenAI's o1-mini and GPT-4o. We publicly release \textbf{TeleChat2}, \textbf{TeleChat2.5} and \textbf{T1}, including post-trained versions with 35B and 115B parameters, to empower developers and researchers with state-of-the-art language models tailored for diverse applications. |
| 2025-07-23 | [Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation](http://arxiv.org/abs/2507.17937v1) | Jaechul Roh, Zachary Novack et al. | Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (jrohsc.github.io/music_attack/). |
| 2025-07-23 | [From Seed to Harvest: Augmenting Human Creativity with AI for Red-teaming Text-to-Image Models](http://arxiv.org/abs/2507.17922v1) | Jessica Quaye, Charvi Rastogi et al. | Text-to-image (T2I) models have become prevalent across numerous applications, making their robust evaluation against adversarial attacks a critical priority. Continuous access to new and challenging adversarial prompts across diverse domains is essential for stress-testing these models for resilience against novel attacks from multiple vectors. Current techniques for generating such prompts are either entirely authored by humans or synthetically generated. On the one hand, datasets of human-crafted adversarial prompts are often too small in size and imbalanced in their cultural and contextual representation. On the other hand, datasets of synthetically-generated prompts achieve scale, but typically lack the realistic nuances and creative adversarial strategies found in human-crafted prompts. To combine the strengths of both human and machine approaches, we propose Seed2Harvest, a hybrid red-teaming method for guided expansion of culturally diverse, human-crafted adversarial prompt seeds. The resulting prompts preserve the characteristics and attack patterns of human prompts while maintaining comparable average attack success rates (0.31 NudeNet, 0.36 SD NSFW, 0.12 Q16). Our expanded dataset achieves substantially higher diversity with 535 unique geographic locations and a Shannon entropy of 7.48, compared to 58 locations and 5.28 entropy in the original dataset. Our work demonstrates the importance of human-machine collaboration in leveraging human creativity and machine computational capacity to achieve comprehensive, scalable red-teaming for continuous T2I model safety evaluation. |
| 2025-07-23 | [Safe Reinforcement Learning-based Automatic Generation Control](http://arxiv.org/abs/2507.17868v1) | Amr S. Mohamed, Emily Nguyen et al. | Amidst the growing demand for implementing advanced control and decision-making algorithms|to enhance the reliability, resilience, and stability of power systems|arises a crucial concern regarding the safety of employing machine learning techniques. While these methods can be applied to derive more optimal control decisions, they often lack safety assurances. This paper proposes a framework based on control barrier functions to facilitate safe learning and deployment of reinforcement learning agents for power system control applications, specifically in the context of automatic generation control. We develop the safety barriers and reinforcement learning framework necessary to establish trust in reinforcement learning as a safe option for automatic generation control - as foundation for future detailed verification and application studies. |
| 2025-07-23 | [A Step-by-step Guide on Nonlinear Model Predictive Control for Safe Mobile Robot Navigation](http://arxiv.org/abs/2507.17856v1) | Dennis Benders, Laura Ferranti et al. | Designing a Model Predictive Control (MPC) scheme that enables a mobile robot to safely navigate through an obstacle-filled environment is a complicated yet essential task in robotics. In this technical report, safety refers to ensuring that the robot respects state and input constraints while avoiding collisions with obstacles despite the presence of disturbances and measurement noise. This report offers a step-by-step approach to implementing Nonlinear Model Predictive Control (NMPC) schemes addressing these safety requirements. Numerous books and survey papers provide comprehensive overviews of linear MPC (LMPC) \cite{bemporad2007robust,kouvaritakis2016model}, NMPC \cite{rawlings2017model,allgower2004nonlinear,mayne2014model,grune2017nonlinear,saltik2018outlook}, and their applications in various domains, including robotics \cite{nascimento2018nonholonomic,nguyen2021model,shi2021advanced,wei2022mpc}. This report does not aim to replicate those exhaustive reviews. Instead, it focuses specifically on NMPC as a foundation for safe mobile robot navigation. The goal is to provide a practical and accessible path from theoretical concepts to mathematical proofs and implementation, emphasizing safety and performance guarantees. It is intended for researchers, robotics engineers, and practitioners seeking to bridge the gap between theoretical NMPC formulations and real-world robotic applications.   This report is not necessarily meant to remain fixed over time. If someone finds an error in the presented theory, please reach out via the given email addresses. We are happy to update the document if necessary. |
| 2025-07-23 | [BetterCheck: Towards Safeguarding VLMs for Automotive Perception Systems](http://arxiv.org/abs/2507.17722v1) | Malsha Ashani Mahawatta Dona, Beatriz Cabrero-Daniel et al. | Large language models (LLMs) are growingly extended to process multimodal data such as text and video simultaneously. Their remarkable performance in understanding what is shown in images is surpassing specialized neural networks (NNs) such as Yolo that is supporting only a well-formed but very limited vocabulary, ie., objects that they are able to detect. When being non-restricted, LLMs and in particular state-of-the-art vision language models (VLMs) show impressive performance to describe even complex traffic situations. This is making them potentially suitable components for automotive perception systems to support the understanding of complex traffic situations or edge case situation. However, LLMs and VLMs are prone to hallucination, which mean to either potentially not seeing traffic agents such as vulnerable road users who are present in a situation, or to seeing traffic agents who are not there in reality. While the latter is unwanted making an ADAS or autonomous driving systems (ADS) to unnecessarily slow down, the former could lead to disastrous decisions from an ADS. In our work, we are systematically assessing the performance of 3 state-of-the-art VLMs on a diverse subset of traffic situations sampled from the Waymo Open Dataset to support safety guardrails for capturing such hallucinations in VLM-supported perception systems. We observe that both, proprietary and open VLMs exhibit remarkable image understanding capabilities even paying thorough attention to fine details sometimes difficult to spot for us humans. However, they are also still prone to making up elements in their descriptions to date requiring hallucination detection strategies such as BetterCheck that we propose in our work. |

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



