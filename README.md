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
| 2025-10-28 | [Learning to Drive Safely with Hybrid Options](http://arxiv.org/abs/2510.24674v1) | Bram De Cooman, Johan Suykens | Out of the many deep reinforcement learning approaches for autonomous driving, only few make use of the options (or skills) framework. That is surprising, as this framework is naturally suited for hierarchical control applications in general, and autonomous driving tasks in specific. Therefore, in this work the options framework is applied and tailored to autonomous driving tasks on highways. More specifically, we define dedicated options for longitudinal and lateral manoeuvres with embedded safety and comfort constraints. This way, prior domain knowledge can be incorporated into the learning process and the learned driving behaviour can be constrained more easily. We propose several setups for hierarchical control with options and derive practical algorithms following state-of-the-art reinforcement learning techniques. By separately selecting actions for longitudinal and lateral control, the introduced policies over combined and hybrid options obtain the same expressiveness and flexibility that human drivers have, while being easier to interpret than classical policies over continuous actions. Of all the investigated approaches, these flexible policies over hybrid options perform the best under varying traffic conditions, outperforming the baseline policies over actions. |
| 2025-10-28 | [Semi-supervised and unsupervised learning for health indicator extraction from guided waves in aerospace composite structures](http://arxiv.org/abs/2510.24614v1) | James Josep Perry, Pablo Garcia-Conde Ortiz et al. | Health indicators (HIs) are central to diagnosing and prognosing the condition of aerospace composite structures, enabling efficient maintenance and operational safety. However, extracting reliable HIs remains challenging due to variability in material properties, stochastic damage evolution, and diverse damage modes. Manufacturing defects (e.g., disbonds) and in-service incidents (e.g., bird strikes) further complicate this process. This study presents a comprehensive data-driven framework that learns HIs via two learning approaches integrated with multi-domain signal processing. Because ground-truth HIs are unavailable, a semi-supervised and an unsupervised approach are proposed: (i) a diversity deep semi-supervised anomaly detection (Diversity-DeepSAD) approach augmented with continuous auxiliary labels used as hypothetical damage proxies, which overcomes the limitation of prior binary labels that only distinguish healthy and failed states while neglecting intermediate degradation, and (ii) a degradation-trend-constrained variational autoencoder (DTC-VAE), in which the monotonicity criterion is embedded via an explicit trend constraint. Guided waves with multiple excitation frequencies are used to monitor single-stiffener composite structures under fatigue loading. Time, frequency, and time-frequency representations are explored, and per-frequency HIs are fused via unsupervised ensemble learning to mitigate frequency dependence and reduce variance. Using fast Fourier transform features, the augmented Diversity-DeepSAD model achieved 81.6% performance, while DTC-VAE delivered the most consistent HIs with 92.3% performance, outperforming existing baselines. |
| 2025-10-28 | [OSWorld-MCP: Benchmarking MCP Tool Invocation In Computer-Use Agents](http://arxiv.org/abs/2510.24563v1) | Hongrui Jia, Jitong Liao et al. | With advances in decision-making and reasoning capabilities, multimodal agents show strong potential in computer application scenarios. Past evaluations have mainly assessed GUI interaction skills, while tool invocation abilities, such as those enabled by the Model Context Protocol (MCP), have been largely overlooked. Comparing agents with integrated tool invocation to those evaluated only on GUI interaction is inherently unfair. We present OSWorld-MCP, the first comprehensive and fair benchmark for assessing computer-use agents' tool invocation, GUI operation, and decision-making abilities in a real-world environment. We design a novel automated code-generation pipeline to create tools and combine them with a curated selection from existing tools. Rigorous manual validation yields 158 high-quality tools (covering 7 common applications), each verified for correct functionality, practical applicability, and versatility. Extensive evaluations of state-of-the-art multimodal agents on OSWorld-MCP show that MCP tools generally improve task success rates (e.g., from 8.3% to 20.4% for OpenAI o3 at 15 steps, from 40.1% to 43.3% for Claude 4 Sonnet at 50 steps), underscoring the importance of assessing tool invocation capabilities. However, even the strongest models have relatively low tool invocation rates, Only 36.3%, indicating room for improvement and highlighting the benchmark's challenge. By explicitly measuring MCP tool usage skills, OSWorld-MCP deepens understanding of multimodal agents and sets a new standard for evaluating performance in complex, tool-assisted environments. Our code, environment, and data are publicly available at https://osworld-mcp.github.io. |
| 2025-10-28 | [Quantum Combinatorial Reasoning for Large Language Models](http://arxiv.org/abs/2510.24509v1) | Carlos Flores-Garrigos, Gaurav Dev et al. | We design and implement a quantum combinatorial reasoning framework for large language models (QCR-LLM), integrating a real quantum computer in the hybrid workflow. QCR-LLM reformulates reasoning aggregation as a higher-order unconstrained binary optimization (HUBO) problem. In this sense, reasoning fragments are represented as binary variables and their interactions encode statistical relevance, logical coherence, and semantic redundancy. We tackle the resulting high-order optimization problem both classically, via simulated annealing, and quantumly through the bias-field digitized counterdiabatic quantum optimizer (BF-DCQO) executed on IBM's superconducting digital quantum processors. Experiments on BIG-Bench Extra Hard (BBEH) benchmarks demonstrate that our QCR-LLM consistently improves reasoning accuracy across multiple LLM backbones, surpassing reasoning-native systems such as o3-high and DeepSeek R1 by up to $+9\,$pp. Despite requiring multiple reasoning samples per query, our QCR-LLM remains approximately five times more energy-efficient than o3-high, owing to the low per-token energy footprint of its GPT-4o backbone. These results constitute the first experimental evidence of quantum-assisted reasoning, showing that hybrid quantum-classical optimization can efficiently enhance reasoning coherence, interpretability, and sustainability in large-scale language models. We have opened the doors to the emergence of quantum intelligence, where harder prompts require quantum optimizers at quantum-advantage level. |
| 2025-10-28 | [LuxIT: A Luxembourgish Instruction Tuning Dataset from Monolingual Seed Data](http://arxiv.org/abs/2510.24434v1) | Julian Valline, Cedric Lothritz et al. | The effectiveness of instruction-tuned Large Language Models (LLMs) is often limited in low-resource linguistic settings due to a lack of high-quality training data. We introduce LuxIT, a novel, monolingual instruction tuning dataset for Luxembourgish developed to mitigate this challenge. We synthesize the dataset from a corpus of native Luxembourgish texts, utilizing DeepSeek-R1-0528, chosen for its shown proficiency in Luxembourgish. Following generation, we apply a quality assurance process, employing an LLM-as-a-judge approach. To investigate the practical utility of the dataset, we fine-tune several smaller-scale LLMs on LuxIT. Subsequent benchmarking against their base models on Luxembourgish language proficiency examinations, however, yields mixed results, with performance varying significantly across different models. LuxIT represents a critical contribution to Luxembourgish natural language processing and offers a replicable monolingual methodology, though our findings highlight the need for further research to optimize its application. |
| 2025-10-28 | [XAI Evaluation Framework for Semantic Segmentation](http://arxiv.org/abs/2510.24414v1) | Reem Hammoud, Abdul karim Gizzini et al. | Ensuring transparency and trust in artificial intelligence (AI) models is essential, particularly as they are increasingly applied in safety-critical and high-stakes domains. Explainable AI (XAI) has emerged as a promising approach to address this challenge, yet the rigorous evaluation of XAI methods remains crucial for optimizing the trade-offs between model complexity, predictive performance, and interpretability. While extensive progress has been achieved in evaluating XAI techniques for classification tasks, evaluation strategies tailored to semantic segmentation remain relatively underexplored. This work introduces a comprehensive and systematic evaluation framework specifically designed for assessing XAI in semantic segmentation, explicitly accounting for both spatial and contextual task complexities. The framework employs pixel-level evaluation strategies and carefully designed metrics to provide fine-grained interpretability insights. Simulation results using recently adapted class activation mapping (CAM)-based XAI schemes demonstrate the efficiency, robustness, and reliability of the proposed methodology. These findings contribute to advancing transparent, trustworthy, and accountable semantic segmentation models. |
| 2025-10-28 | [OS-Sentinel: Towards Safety-Enhanced Mobile GUI Agents via Hybrid Validation in Realistic Workflows](http://arxiv.org/abs/2510.24411v1) | Qiushi Sun, Mukai Li et al. | Computer-using agents powered by Vision-Language Models (VLMs) have demonstrated human-like capabilities in operating digital environments like mobile platforms. While these agents hold great promise for advancing digital automation, their potential for unsafe operations, such as system compromise and privacy leakage, is raising significant concerns. Detecting these safety concerns across the vast and complex operational space of mobile environments presents a formidable challenge that remains critically underexplored. To establish a foundation for mobile agent safety research, we introduce MobileRisk-Live, a dynamic sandbox environment accompanied by a safety detection benchmark comprising realistic trajectories with fine-grained annotations. Built upon this, we propose OS-Sentinel, a novel hybrid safety detection framework that synergistically combines a Formal Verifier for detecting explicit system-level violations with a VLM-based Contextual Judge for assessing contextual risks and agent actions. Experiments show that OS-Sentinel achieves 10%-30% improvements over existing approaches across multiple metrics. Further analysis provides critical insights that foster the development of safer and more reliable autonomous mobile agents. |
| 2025-10-28 | [Filtering instances and rejecting predictions to obtain reliable models in healthcare](http://arxiv.org/abs/2510.24368v1) | Maria Gabriela Valeriano, David Kohan Marzagão et al. | Machine Learning (ML) models are widely used in high-stakes domains such as healthcare, where the reliability of predictions is critical. However, these models often fail to account for uncertainty, providing predictions even with low confidence. This work proposes a novel two-step data-centric approach to enhance the performance of ML models by improving data quality and filtering low-confidence predictions. The first step involves leveraging Instance Hardness (IH) to filter problematic instances during training, thereby refining the dataset. The second step introduces a confidence-based rejection mechanism during inference, ensuring that only reliable predictions are retained. We evaluate our approach using three real-world healthcare datasets, demonstrating its effectiveness at improving model reliability while balancing predictive performance and rejection rate. Additionally, we use alternative criteria - influence values for filtering and uncertainty for rejection - as baselines to evaluate the efficiency of the proposed method. The results demonstrate that integrating IH filtering with confidence-based rejection effectively enhances model performance while preserving a large proportion of instances. This approach provides a practical method for deploying ML systems in safety-critical applications. |
| 2025-10-28 | [Global-State-Free Obstacle Avoidance for Quadrotor Control in Air-Ground Cooperation](http://arxiv.org/abs/2510.24315v1) | Baozhe Zhang, Xinwei Chen et al. | CoNi-MPC provides an efficient framework for UAV control in air-ground cooperative tasks by relying exclusively on relative states, eliminating the need for global state estimation. However, its lack of environmental information poses significant challenges for obstacle avoidance. To address this issue, we propose a novel obstacle avoidance algorithm, Cooperative Non-inertial frame-based Obstacle Avoidance (CoNi-OA), designed explicitly for UAV-UGV cooperative scenarios without reliance on global state estimation or obstacle prediction. CoNi-OA uniquely utilizes a single frame of raw LiDAR data from the UAV to generate a modulation matrix, which directly adjusts the quadrotor's velocity to achieve obstacle avoidance. This modulation-based method enables real-time generation of collision-free trajectories within the UGV's non-inertial frame, significantly reducing computational demands (less than 5 ms per iteration) while maintaining safety in dynamic and unpredictable environments. The key contributions of this work include: (1) a modulation-based obstacle avoidance algorithm specifically tailored for UAV-UGV cooperation in non-inertial frames without global states; (2) rapid, real-time trajectory generation based solely on single-frame LiDAR data, removing the need for obstacle modeling or prediction; and (3) adaptability to both static and dynamic environments, thus extending applicability to featureless or unknown scenarios. |
| 2025-10-28 | [Trajectory Design for UAV-Based Low-Altitude Wireless Networks in Unknown Environments: A Digital Twin-Assisted TD3 Approach](http://arxiv.org/abs/2510.24255v1) | Jihao Luo, Zesong Fei et al. | Unmanned aerial vehicles (UAVs) are emerging as key enablers for low-altitude wireless network (LAWN), particularly when terrestrial networks are unavailable. In such scenarios, the environmental topology is typically unknown; hence, designing efficient and safe UAV trajectories is essential yet challenging. To address this, we propose a digital twin (DT)-assisted training and deployment framework. In this framework, the UAV transmits integrated sensing and communication signals to provide communication services to ground users, while simultaneously collecting echoes that are uploaded to the DT server to progressively construct virtual environments (VEs). These VEs accelerate model training and are continuously updated with real-time UAV sensing data during deployment, supporting decision-making and enhancing flight safety. Based on this framework, we further develop a trajectory design scheme that integrates simulated annealing for efficient user scheduling with the twin-delayed deep deterministic policy gradient algorithm for continuous trajectory design, aiming to minimize mission completion time while ensuring obstacle avoidance. Simulation results demonstrate that the proposed approach achieves faster convergence, higher flight safety, and shorter mission completion time compared with baseline methods, providing a robust and efficient solution for LAWN deployment in unknown environments. |
| 2025-10-28 | [Advancing Interdisciplinary Approaches to Online Safety Research](http://arxiv.org/abs/2510.24227v1) | Senuri Wijenayake, Joanne Gray et al. | The growing prevalence of negative experiences in online spaces demands urgent attention from the human-computer interaction (HCI) community. However, research on online safety remains fragmented across different HCI subfields, with limited communication and collaboration between disciplines. This siloed approach risks creating ineffective responses, including design solutions that fail to meet the diverse needs of users, and policy efforts that overlook critical usability concerns. This workshop aims to foster interdisciplinary dialogue on online safety by bringing together researchers from within and beyond HCI - including but not limited to Social Computing, Digital Design, Internet Policy, Cybersecurity, Ethics, and Social Sciences. By uniting researchers, policymakers, industry practitioners, and community advocates we aim to identify shared challenges in online safety research, highlight gaps in current knowledge, and establish common research priorities. The workshop will support the development of interdisciplinary research plans and establish collaborative environments - both within and beyond Australia - to action them. |
| 2025-10-28 | [UniPlanner: A Unified Motion Planning Framework for Autonomous Vehicle Decision-Making Systems via Multi-Dataset Integration](http://arxiv.org/abs/2510.24166v1) | Xin Yang, Yuhang Zhang et al. | Motion planning is a critical component of autonomous vehicle decision-making systems, directly determining trajectory safety and driving efficiency. While deep learning approaches have advanced planning capabilities, existing methods remain confined to single-dataset training, limiting their robustness in planning.   Through systematic analysis, we discover that vehicular trajectory distributions and history-future correlations demonstrate remarkable consistency across different datasets. Based on these findings, we propose UniPlanner, the first planning framework designed for multi-dataset integration in autonomous vehicle decision-making. UniPlanner achieves unified cross-dataset learning through three synergistic innovations.   First, the History-Future Trajectory Dictionary Network (HFTDN) aggregates history-future trajectory pairs from multiple datasets, using historical trajectory similarity to retrieve relevant futures and generate cross-dataset planning guidance.   Second, the Gradient-Free Trajectory Mapper (GFTM) learns robust history-future correlations from multiple datasets, transforming historical trajectories into universal planning priors. Its gradient-free design ensures the introduction of valuable priors while preventing shortcut learning, making the planning knowledge safely transferable. Third, the Sparse-to-Dense (S2D) paradigm implements adaptive dropout to selectively suppress planning priors during training for robust learning, while enabling full prior utilization during inference to maximize planning performance. |
| 2025-10-28 | [Enhancing Vision-Language Models for Autonomous Driving through Task-Specific Prompting and Spatial Reasoning](http://arxiv.org/abs/2510.24152v1) | Aodi Wu, Xubo Luo | This technical report presents our solution for the RoboSense Challenge at IROS 2025, which evaluates Vision-Language Models (VLMs) on autonomous driving scene understanding across perception, prediction, planning, and corruption detection tasks. We propose a systematic framework built on four core components. First, a Mixture-of-Prompts router classifies questions and dispatches them to task-specific expert prompts, eliminating interference across diverse question types. Second, task-specific prompts embed explicit coordinate systems, spatial reasoning rules, role-playing, Chain-of-Thought/Tree-of-Thought reasoning, and few-shot examples tailored to each task. Third, a visual assembly module composes multi-view images with object crops, magenta markers, and adaptive historical frames based on question requirements. Fourth, we configure model inference parameters (temperature, top-p, message roles) per task to optimize output quality. Implemented on Qwen2.5-VL-72B, our approach achieves 70.87% average accuracy on Phase-1 (clean data) and 72.85% on Phase-2 (corrupted data), demonstrating that structured prompting and spatial grounding substantially enhance VLM performance on safety-critical autonomous driving tasks. Code and prompt are available at https://github.com/wuaodi/UCAS-CSU-phase2. |
| 2025-10-28 | [Modeling Electric Vehicle Car-Following Behavior: Classical vs Machine Learning Approach](http://arxiv.org/abs/2510.24085v1) | Md. Shihab Uddin, Md Nazmus Shakib et al. | The increasing adoption of electric vehicles (EVs) necessitates an understanding of their driving behavior to enhance traffic safety and develop smart driving systems. This study compares classical and machine learning models for EV car following behavior. Classical models include the Intelligent Driver Model (IDM), Optimum Velocity Model (OVM), Optimal Velocity Relative Velocity (OVRV), and a simplified CACC model, while the machine learning approach employs a Random Forest Regressor. Using a real world dataset of an EV following an internal combustion engine (ICE) vehicle under varied driving conditions, we calibrated classical model parameters by minimizing the RMSE between predictions and real data. The Random Forest model predicts acceleration using spacing, speed, and gap type as inputs. Results demonstrate the Random Forest's superior accuracy, achieving RMSEs of 0.0046 (medium gap), 0.0016 (long gap), and 0.0025 (extra long gap). Among physics based models, CACC performed best, with an RMSE of 2.67 for long gaps. These findings highlight the machine learning model's performance across all scenarios. Such models are valuable for simulating EV behavior and analyzing mixed autonomy traffic dynamics in EV integrated environments. |
| 2025-10-28 | [SynAD: Enhancing Real-World End-to-End Autonomous Driving Models through Synthetic Data Integration](http://arxiv.org/abs/2510.24052v1) | Jongsuk Kim, Jaeyoung Lee et al. | Recent advancements in deep learning and the availability of high-quality real-world driving datasets have propelled end-to-end autonomous driving. Despite this progress, relying solely on real-world data limits the variety of driving scenarios for training. Synthetic scenario generation has emerged as a promising solution to enrich the diversity of training data; however, its application within E2E AD models remains largely unexplored. This is primarily due to the absence of a designated ego vehicle and the associated sensor inputs, such as camera or LiDAR, typically provided in real-world scenarios. To address this gap, we introduce SynAD, the first framework designed to enhance real-world E2E AD models using synthetic data. Our method designates the agent with the most comprehensive driving information as the ego vehicle in a multi-agent synthetic scenario. We further project path-level scenarios onto maps and employ a newly developed Map-to-BEV Network to derive bird's-eye-view features without relying on sensor inputs. Finally, we devise a training strategy that effectively integrates these map-based synthetic data with real driving data. Experimental results demonstrate that SynAD effectively integrates all components and notably enhances safety performance. By bridging synthetic scenario generation and E2E AD, SynAD paves the way for more comprehensive and robust autonomous driving models. |
| 2025-10-28 | [AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](http://arxiv.org/abs/2510.24034v1) | Yufan Liu, Wanqian Zhang et al. | Despite rapid advancements in text-to-image (T2I) models, their safety mechanisms are vulnerable to adversarial prompts, which maliciously generate unsafe images. Current red-teaming methods for proactively assessing such vulnerabilities usually require white-box access to T2I models, and rely on inefficient per-prompt optimization, as well as inevitably generate semantically meaningless prompts easily blocked by filters. In this paper, we propose APT (AutoPrompT), a black-box framework that leverages large language models (LLMs) to automatically generate human-readable adversarial suffixes for benign prompts. We first introduce an alternating optimization-finetuning pipeline between adversarial suffix optimization and fine-tuning the LLM utilizing the optimized suffix. Furthermore, we integrates a dual-evasion strategy in optimization phase, enabling the bypass of both perplexity-based filter and blacklist word filter: (1) we constrain the LLM generating human-readable prompts through an auxiliary LLM perplexity scoring, which starkly contrasts with prior token-level gibberish, and (2) we also introduce banned-token penalties to suppress the explicit generation of banned-tokens in blacklist. Extensive experiments demonstrate the excellent red-teaming performance of our human-readable, filter-resistant adversarial prompts, as well as superior zero-shot transferability which enables instant adaptation to unseen prompts and exposes critical vulnerabilities even in commercial APIs (e.g., Leonardo.Ai.). |
| 2025-10-28 | [Lifecycle-Aware code generation: Leveraging Software Engineering Phases in LLMs](http://arxiv.org/abs/2510.24019v1) | Xing Xing, Wei Wang et al. | Recent progress in large language models (LLMs) has advanced automatic code generation, yet most approaches rely on direct, single-step translation from problem descriptions to code, disregarding structured software engineering practices. We introduce a lifecycle-aware framework that systematically incorporates intermediate artifacts such as requirements analysis, state machine modeling, and pseudocode into both the training and inference stages. This design aligns code generation with standard software development phases and enables more structured reasoning. Experiments show that lifecycle-level fine-tuning improves code correctness by up to 75% over the same model before fine-tuning, with performance gains compounding across intermediate stages. Multi-step inference consistently surpasses single-step generation, demonstrating the effectiveness of intermediate scaffolding. Notably, open-source LLMs, once fine-tuned under our framework, match or slightly outperform models pretrained on code. When applied to DeepSeek-Coder-1.3B, our framework yields relative CodeBLEU improvements of 34.3%, 20.0%, 11.2%, and 22.3% over ChatGPT-3.5, ChatGPT-4o-mini, DeepSeek-R1, and LLaMA-8B, respectively. Our pipeline also proves robust with up to 80\% less training data, confirming its resilience. Ablation studies further reveal that each intermediate artifact contributes distinctly to final code quality, with state machine modeling yielding the most substantial impact. Our source code and detailed experimental data are available at https://anonymous.4open.science/r/Lifecycle-Aware-3CCB. |
| 2025-10-28 | [Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](http://arxiv.org/abs/2510.24012v1) | Byeonghu Na, Mina Kang et al. | Text-to-image models have recently made significant advances in generating realistic and semantically coherent images, driven by advanced diffusion models and large-scale web-crawled datasets. However, these datasets often contain inappropriate or biased content, raising concerns about the generation of harmful outputs when provided with malicious text prompts. We propose Safe Text embedding Guidance (STG), a training-free approach to improve the safety of diffusion models by guiding the text embeddings during sampling. STG adjusts the text embeddings based on a safety function evaluated on the expected final denoised image, allowing the model to generate safer outputs without additional training. Theoretically, we show that STG aligns the underlying model distribution with safety constraints, thereby achieving safer outputs while minimally affecting generation quality. Experiments on various safety scenarios, including nudity, violence, and artist-style removal, show that STG consistently outperforms both training-based and training-free baselines in removing unsafe content while preserving the core semantic intent of input prompts. Our code is available at https://github.com/aailab-kaist/STG. |
| 2025-10-28 | [How Does Environmental Information Disclosure Affect Corporate Environmental Performance? Evidence from Chinese A-Share Listed Companies](http://arxiv.org/abs/2510.24002v1) | Zehao Lin | Global climate warming and air pollution pose severe threats to economic development and public safety, presenting significant challenges to sustainable development worldwide. Corporations, as key players in resource utilization and emissions, have drawn increasing attention from policymakers, researchers, and the public regarding their environmental strategies and practices. This study employs a two-way fixed effects panel model to examine the impact of environmental information disclosure on corporate environmental performance, its regional heterogeneity, and the underlying mechanisms. The results demonstrate that environmental information disclosure significantly improves corporate environmental performance, with the effect being more pronounced in areas of high population density and limited green space. These findings provide empirical evidence supporting the role of environmental information disclosure as a critical tool for improving corporate environmental practices. The study highlights the importance of targeted, region-specific policies to maximize the effectiveness of disclosure, offering valuable insights for promoting sustainable development through enhanced corporate transparency. |
| 2025-10-28 | [VOCALoco: Viability-Optimized Cost-aware Adaptive Locomotion](http://arxiv.org/abs/2510.23997v1) | Stanley Wu, Mohamad H. Danesh et al. | Recent advancements in legged robot locomotion have facilitated traversal over increasingly complex terrains. Despite this progress, many existing approaches rely on end-to-end deep reinforcement learning (DRL), which poses limitations in terms of safety and interpretability, especially when generalizing to novel terrains. To overcome these challenges, we introduce VOCALoco, a modular skill-selection framework that dynamically adapts locomotion strategies based on perceptual input. Given a set of pre-trained locomotion policies, VOCALoco evaluates their viability and energy-consumption by predicting both the safety of execution and the anticipated cost of transport over a fixed planning horizon. This joint assessment enables the selection of policies that are both safe and energy-efficient, given the observed local terrain. We evaluate our approach on staircase locomotion tasks, demonstrating its performance in both simulated and real-world scenarios using a quadrupedal robot. Empirical results show that VOCALoco achieves improved robustness and safety during stair ascent and descent compared to a conventional end-to-end DRL policy |

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



