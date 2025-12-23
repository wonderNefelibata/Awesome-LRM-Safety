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
| 2025-12-22 | [Scalably Enhancing the Clinical Validity of a Task Benchmark with Physician Oversight](http://arxiv.org/abs/2512.19691v1) | Junze Ye, Daniel Tawfik et al. | Automating the calculation of clinical risk scores offers a significant opportunity to reduce physician administrative burden and enhance patient care. The current standard for evaluating this capability is MedCalc-Bench, a large-scale dataset constructed using LLM-based feature extraction and rule-based aggregation. However, treating such model-generated benchmarks as static oracles risks enshrining historical model errors as evaluation gold standards, a problem dangerously amplified when these datasets serve as reward signals for Reinforcement Learning (RL). In this work, we propose viewing benchmarks for complex tasks such as clinical score computation as ''in-progress living documents'' that should be periodically re-evaluated as the processes for creating them improve. We introduce a systematic, physician-in-the-loop pipeline that leverages advanced agentic verifiers to audit and relabel MedCalc-Bench, utilizing automated triage to reserve scarce clinician attention for the most contentious instances. Our audit reveals that a notable fraction of original labels diverge from medical ground truth due to extraction errors, calculator logic mismatches, and clinical ambiguity. To study whether this label noise meaningfully impacts downstream RL training, we fine-tune a Qwen3-8B model via Group Relative Policy Optimization (GRPO) and demonstrate that training on corrected labels yields an 8.7% absolute improvement in accuracy over the original baseline -- validating that label noise materially affects model evaluation. These findings underscore that in safety-critical domains, rigorous benchmark maintenance is a prerequisite for genuine model alignment. |
| 2025-12-22 | [Optimal-coupling-observer AV motion control securing comfort in the presence of cyber attacks](http://arxiv.org/abs/2512.19679v1) | Farzam Tajdari, Georgios Papaioannou et al. | The security of Automated Vehicles (AVs) is an important emerging area of research in traffic safety. Methods have been published and evaluated in experimental vehicles to secure safe AV control in the presence of attacks, but human motion comfort is rarely investigated in such studies.   In this paper, we present an innovative optimal-coupling-observer-based framework that rejects the impact of bounded sensor attacks in a network of connected and automated vehicles from safety and comfort point of view.   We demonstrate its performance in car following with cooperative adaptive cruise control for platoons with redundant distance and velocity sensors.   The error dynamics are formulated as a Linear Time Variant (LTV) system, resulting in complex stability conditions that are investigated using a Linear Matrix Inequality (LMI) approach guaranteeing global asymptotic stability.   We prove the capability of the framework to secure occupants' safety and comfort in the presence of bounded attacks. In the onset of attack, the framework rapidly detects attacked sensors and switches to the most reliable observer eliminating attacked sensors, even with modest attack magnitudes. Without our proposed method, severe (but bounded) attacks result in collisions and major discomfort. With our method, attacks had negligible effects on motion comfort evaluated using ISO-2631 Ride Comfort and Motion Sickness indexes. The results pave the path to bring comfort to the forefront of AVs security. |
| 2025-12-22 | [Optimal Uncertainty Quantification under General Moment Constraints on Input Subdomains](http://arxiv.org/abs/2512.19572v1) | Rong Jin, Xingsheng Sun | We present an optimal uncertainty quantification (OUQ) framework for systems whose uncertain inputs are characterized by truncated moment constraints defined over subdomains. Based on this partial information, rigorous optimal upper and lower bounds on the probability of failure (PoF) are derived over the admissible set of probability measures, providing a principled basis for system safety certification. We formulate the OUQ problem under general subdomain moment constraints and develop a high-performance computational framework to compute the optimal bounds. This approach transforms the original infinite-dimensional optimization problems into finite-dimensional unconstrained ones parameterized solely by free canonical moments. To address the prohibitive cost of PoF evaluation in high-dimensional settings, we incorporate inverse transform sampling (ITS), enabling efficient and accurate PoF estimation within the OUQ optimization. We also demonstrate that constraining inputs only by zeroth-order moments over subdomains yields a formulation equivalent to evidence theory. Three groups of numerical examples demonstrate the framework's effectiveness and scalability. Results show that increasing the number of subdomains or the moment order systematically tightens the bound interval. For high-dimensional problems, the ITS strategy reduces computational costs by up to two orders of magnitude while maintaining relative error below 1%. Furthermore, we identify regimes where optimal bounds are sensitive to subdomain partitioning or higher-order moments, guiding uncertainty reduction efforts for safety certification. |
| 2025-12-22 | [Results of the 2024 CommonRoad Motion Planning Competition for Autonomous Vehicles](http://arxiv.org/abs/2512.19564v1) | Yanliang Huang, Xia Yan et al. | Over the past decade, a wide range of motion planning approaches for autonomous vehicles has been developed to handle increasingly complex traffic scenarios. However, these approaches are rarely compared on standardized benchmarks, limiting the assessment of relative strengths and weaknesses. To address this gap, we present the setup and results of the 4th CommonRoad Motion Planning Competition held in 2024, conducted using the CommonRoad benchmark suite. This annual competition provides an open-source and reproducible framework for benchmarking motion planning algorithms. The benchmark scenarios span highway and urban environments with diverse traffic participants, including passenger cars, buses, and bicycles. Planner performance is evaluated along four dimensions: efficiency, safety, comfort, and compliance with selected traffic rules. This report introduces the competition format and provides a comparison of representative high-performing planners from the 2023 and 2024 editions. |
| 2025-12-22 | [Augmenting Intelligence: A Hybrid Framework for Scalable and Stable Explanations](http://arxiv.org/abs/2512.19557v1) | Lawrence Krukrubo, Julius Odede et al. | Current approaches to Explainable AI (XAI) face a "Scalability-Stability Dilemma." Post-hoc methods (e.g., LIME, SHAP) may scale easily but suffer from instability, while supervised explanation frameworks (e.g., TED) offer stability but require prohibitive human effort to label every training instance. This paper proposes a Hybrid LRR-TED framework that addresses this dilemma through a novel "Asymmetry of Discovery." When applied to customer churn prediction, we demonstrate that automated rule learners (GLRM) excel at identifying broad "Safety Nets" (retention patterns) but struggle to capture specific "Risk Traps" (churn triggers)-a phenomenon we term the Anna Karenina Principle of Churn. By initialising the explanation matrix with automated safety rules and augmenting it with a Pareto-optimal set of just four human-defined risk rules, our approach achieves 94.00% predictive accuracy. This configuration outperforms the full 8-rule manual expert baseline while reducing human annotation effort by 50%, proposing a shift in the paradigm for Human-in-the-Loop AI: moving experts from the role of "Rule Writers" to "Exception Handlers." |
| 2025-12-22 | [Ferro-hydrodynamics of droplet necking filaments](http://arxiv.org/abs/2512.19459v1) | Neeladri Sekhar Bera, Apurba Roy et al. | We explore the necking, filament thinning, and pinchoff dynamics of ferrofluid droplets within a magnetic field, via a simple and low-cost experimental method. In our studies, both the Ohnesorge number Oh and the Deborah number De are O1, a typically inaccessible regime with conventional extensional rheometers. Under magnetic forcing, the nanoparticles assemble into field aligned, chainlike structures, that generate a tunable magnetoelastic response, and markedly alter the extensional flow. Although behaving as Newtonian liquids in the absence of a magnetic field, the field induces extensional thickening, and the emergence of beads on a string BOAS structures in the ferrofluid filaments, a non-Newtonian signature. By combining controlled elongation with high speed imaging, we directly quantify the magnetic field-dependent extensional viscosity and relaxation time. Our findings underscore how magnetically induced microstructures govern filament stability and extensional dynamics in ferrofluids. |
| 2025-12-22 | [Real-Time Machine Learning for Embedded Anomaly Detection](http://arxiv.org/abs/2512.19383v1) | Abdelmadjid Benmachiche, Khadija Rais et al. | The spread of a resource-constrained Internet of Things (IoT) environment and embedded devices has put pressure on the real-time detection of anomalies occurring at the edge. This survey presents an overview of machine-learning methods aimed specifically at on-device anomaly detection with extremely strict constraints for latency, memory, and power consumption. Lightweight algorithms such as Isolation Forest, One-Class SVM, recurrent architectures, and statistical techniques are compared here according to the realities of embedded implementation. Our survey brings out significant trade-offs of accuracy and computational efficiency of detection, as well as how hardware constraints end up fundamentally redefining algorithm choice. The survey is completed with a set of practical recommendations on the choice of the algorithm depending on the equipment profiles and new trends in TinyML, which can help close the gap between detection capabilities and embedded reality. The paper serves as a strategic roadmap for engineers deploying anomaly detection in edge environments that are constrained by bandwidth and may be safety-critical. |
| 2025-12-22 | [Time-Vertex Machine Learning for Optimal Sensor Placement in Temporal Graph Signals: Applications in Structural Health Monitoring](http://arxiv.org/abs/2512.19309v1) | Keivan Faghih Niresi, Jun Qing et al. | Structural Health Monitoring (SHM) plays a crucial role in maintaining the safety and resilience of infrastructure. As sensor networks grow in scale and complexity, identifying the most informative sensors becomes essential to reduce deployment costs without compromising monitoring quality. While Graph Signal Processing (GSP) has shown promise by leveraging spatial correlations among sensor nodes, conventional approaches often overlook the temporal dynamics of structural behavior. To overcome this limitation, we propose Time-Vertex Machine Learning (TVML), a novel framework that integrates GSP, time-domain analysis, and machine learning to enable interpretable and efficient sensor placement by identifying representative nodes that minimize redundancy while preserving critical information. We evaluate the proposed approach on two bridge datasets for damage detection and time-varying graph signal reconstruction tasks. The results demonstrate the effectiveness of our approach in enhancing SHM systems by providing a robust, adaptive, and efficient solution for sensor placement. |
| 2025-12-22 | [Stability Analysis of a B-Spline Deep Neural Operator for Nonlinear Systems](http://arxiv.org/abs/2512.19291v1) | Raffaele Romagnoli, Soummya Kar | This paper investigates the stability properties of neural operators through the structured representation offered by the Hybrid B-spline Deep Neural Operator (HBDNO). While existing stability-aware architectures typically enforce restrictive constraints that limit universality, HBDNO preserves full expressive power by representing outputs via B-spline control points. We show that these control points form a natural observable for post-training stability analysis. By applying Dynamic Mode Decomposition and connecting the resulting discrete dynamics to the Koopman operator framework, we provide a principled approach to spectral characterization of learned operators. Numerical results demonstrate the ability to assess stability and reveal future directions for safety-critical applications. |
| 2025-12-22 | [Digital Twin-Driven Zero-Shot Fault Diagnosis of Axial Piston Pumps Using Fluid-Borne Noise Signals](http://arxiv.org/abs/2512.19280v1) | Chang Dong, Jianfeng Tao et al. | Axial piston pumps are crucial components in fluid power systems, where reliable fault diagnosis is essential for ensuring operational safety and efficiency. Traditional data-driven methods require extensive labeled fault data, which is often impractical to obtain, while model-based approaches suffer from parameter uncertainties. This paper proposes a digital twin (DT)-driven zero-shot fault diagnosis framework utilizing fluid-borne noise (FBN) signals. The framework calibrates a high-fidelity DT model using only healthy-state data, generates synthetic fault signals for training deep learning classifiers, and employs a physics-informed neural network (PINN) as a virtual sensor for flow ripple estimation. Gradient-weighted class activation mapping (Grad-CAM) is integrated to visualize the decision-making process of neural networks, revealing that large kernels matching the subsequence length in time-domain inputs and small kernels in time-frequency domain inputs enable higher diagnostic accuracy by focusing on physically meaningful features. Experimental validations demonstrate that training on signals from the calibrated DT model yields diagnostic accuracies exceeding 95\% on real-world benchmarks, while uncalibrated models result in significantly lower performance, highlighting the framework's effectiveness in data-scarce scenarios. |
| 2025-12-22 | [Identifying Features Associated with Bias Against 93 Stigmatized Groups in Language Models and Guardrail Model Safety Mitigation](http://arxiv.org/abs/2512.19238v1) | Anna-Maria Gueorguieva, Aylin Caliskan | Large language models (LLMs) have been shown to exhibit social bias, however, bias towards non-protected stigmatized identities remain understudied. Furthermore, what social features of stigmas are associated with bias in LLM outputs is unknown. From psychology literature, it has been shown that stigmas contain six shared social features: aesthetics, concealability, course, disruptiveness, origin, and peril. In this study, we investigate if human and LLM ratings of the features of stigmas, along with prompt style and type of stigma, have effect on bias towards stigmatized groups in LLM outputs. We measure bias against 93 stigmatized groups across three widely used LLMs (Granite 3.0-8B, Llama-3.1-8B, Mistral-7B) using SocialStigmaQA, a benchmark that includes 37 social scenarios about stigmatized identities; for example deciding wether to recommend them for an internship. We find that stigmas rated by humans to be highly perilous (e.g., being a gang member or having HIV) have the most biased outputs from SocialStigmaQA prompts (60% of outputs from all models) while sociodemographic stigmas (e.g. Asian-American or old age) have the least amount of biased outputs (11%). We test if the amount of biased outputs could be decreased by using guardrail models, models meant to identify harmful input, using each LLM's respective guardrail model (Granite Guardian 3.0, Llama Guard 3.0, Mistral Moderation API). We find that bias decreases significantly by 10.4%, 1.4%, and 7.8%, respectively. However, we show that features with significant effect on bias remain unchanged post-mitigation and that guardrail models often fail to recognize the intent of bias in prompts. This work has implications for using LLMs in scenarios involving stigmatized groups and we suggest future work towards improving guardrail models for bias mitigation. |
| 2025-12-22 | [PEDESTRIAN: An Egocentric Vision Dataset for Obstacle Detection on Pavements](http://arxiv.org/abs/2512.19190v1) | Marios Thoma, Zenonas Theodosiou et al. | Walking has always been a primary mode of transportation and is recognized as an essential activity for maintaining good health. Despite the need for safe walking conditions in urban environments, sidewalks are frequently obstructed by various obstacles that hinder free pedestrian movement. Any object obstructing a pedestrian's path can pose a safety hazard. The advancement of pervasive computing and egocentric vision techniques offers the potential to design systems that can automatically detect such obstacles in real time, thereby enhancing pedestrian safety. The development of effective and efficient identification algorithms relies on the availability of comprehensive and well-balanced datasets of egocentric data. In this work, we introduce the PEDESTRIAN dataset, comprising egocentric data for 29 different obstacles commonly found on urban sidewalks. A total of 340 videos were collected using mobile phone cameras, capturing a pedestrian's point of view. Additionally, we present the results of a series of experiments that involved training several state-of-the-art deep learning algorithms using the proposed dataset, which can be used as a benchmark for obstacle detection and recognition tasks. The dataset can be used for training pavement obstacle detectors to enhance the safety of pedestrians in urban areas. |
| 2025-12-22 | [AMap: Distilling Future Priors for Ahead-Aware Online HD Map Construction](http://arxiv.org/abs/2512.19150v1) | Ruikai Li, Xinrun Li et al. | Online High-Definition (HD) map construction is pivotal for autonomous driving. While recent approaches leverage historical temporal fusion to improve performance, we identify a critical safety flaw in this paradigm: it is inherently ``spatially backward-looking." These methods predominantly enhance map reconstruction in traversed areas, offering minimal improvement for the unseen road ahead. Crucially, our analysis of downstream planning tasks reveals a severe asymmetry: while rearward perception errors are often tolerable, inaccuracies in the forward region directly precipitate hazardous driving maneuvers. To bridge this safety gap, we propose AMap, a novel framework for Ahead-aware online HD Mapping. We pioneer a ``distill-from-future" paradigm, where a teacher model with privileged access to future temporal contexts guides a lightweight student model restricted to the current frame. This process implicitly compresses prospective knowledge into the student model, endowing it with ``look-ahead" capabilities at zero inference-time cost. Technically, we introduce a Multi-Level BEV Distillation strategy with spatial masking and an Asymmetric Query Adaptation module to effectively transfer future-aware representations to the student's static queries. Extensive experiments on the nuScenes and Argoverse 2 benchmark demonstrate that AMap significantly enhances current-frame perception. Most notably, it outperforms state-of-the-art temporal models in critical forward regions while maintaining the efficiency of single current frame inference. |
| 2025-12-22 | [WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](http://arxiv.org/abs/2512.19133v1) | Pengxuan Yang, Ben Lu et al. | Latent World Models enhance scene representation through temporal self-supervised learning, presenting a perception annotation-free paradigm for end-to-end autonomous driving. However, the reconstruction-oriented representation learning tangles perception with planning tasks, leading to suboptimal optimization for planning. To address this challenge, we propose WorldRFT, a planning-oriented latent world model framework that aligns scene representation learning with planning via a hierarchical planning decomposition and local-aware interactive refinement mechanism, augmented by reinforcement learning fine-tuning (RFT) to enhance safety-critical policy performance. Specifically, WorldRFT integrates a vision-geometry foundation model to improve 3D spatial awareness, employs hierarchical planning task decomposition to guide representation optimization, and utilizes local-aware iterative refinement to derive a planning-oriented driving policy. Furthermore, we introduce Group Relative Policy Optimization (GRPO), which applies trajectory Gaussianization and collision-aware rewards to fine-tune the driving policy, yielding systematic improvements in safety. WorldRFT achieves state-of-the-art (SOTA) performance on both open-loop nuScenes and closed-loop NavSim benchmarks. On nuScenes, it reduces collision rates by 83% (0.30% -> 0.05%). On NavSim, using camera-only sensors input, it attains competitive performance with the LiDAR-based SOTA method DiffusionDrive (87.8 vs. 88.1 PDMS). |
| 2025-12-22 | [GW231123: A Case for Binary Microlensing in a Strong Lensing Field](http://arxiv.org/abs/2512.19118v1) | Xikai Shan, Huan Yang et al. | The unusual properties of GW231123, including component masses within the pair-instability mass gap ($137^{+22}_{-17}\mathrm{M}_\odot$ and $103^{+20}_{-52}\mathrm{M}_\odot$ at 90\% credible intervals) and extremely large spins near the Kerr limit, have challenged standard formation scenarios. While gravitational lensing has been proposed as an explanation, current millilensing studies suggest the signal consists of three overlapping images, a configuration that exceeds the predictions of the isolated point-mass lens model. In this work, we investigate a binary lens model embedded within a strong lensing galaxy. This is the simplest model that not only naturally produces the observed number of images but also aligns with the fact that microlensing objects usually reside in galaxies. To overcome the high computational cost of the diffraction integral required for wave optics, we constructed a Transformer-based neural network that accurately generates lensing waveforms within milliseconds per waveform. Using the NRSur7dq4 waveform model, we find primary and secondary lens masses of $714^{+239}_{-309} \mathrm{M}_\odot$ and $87^{+139}_{-73} \mathrm{M}_\odot$, respectively. We also find a strong lensing magnification of $5.56^{+2.78}_{-1.98}$ (at 90\% credible intervals) and a Bayes factor of $\log_{10}B^\mathrm{Binary}_\mathrm{Single}\simeq1.34$. This result underscores the necessity of considering multi-body and environmental effects in microlensing studies. More crucially, under this embedded binary lens interpretation, the inferred source-frame binary black hole masses ($80.0^{+21.3}_{-14.4} \mathrm{M}_\odot$ and $62.0^{+19.8}_{-29.4} \mathrm{M}_\odot$) and spins ($0.37^{+0.51}_{-0.33}$ and $0.40^{+0.52}_{-0.35}$) shift to values consistent with the current population constrained from O1--O3. |
| 2025-12-22 | [The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation](http://arxiv.org/abs/2512.19025v1) | Hengrui Jia, Taoran Li et al. | Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have ``forgotten'' the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose \name, an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$β$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets. |
| 2025-12-22 | [DREAM: Dynamic Red-teaming across Environments for AI Models](http://arxiv.org/abs/2512.19016v1) | Liming Lu, Xiang Gu et al. | Large Language Models (LLMs) are increasingly used in agentic systems, where their interactions with diverse tools and environments create complex, multi-stage safety challenges. However, existing benchmarks mostly rely on static, single-turn assessments that miss vulnerabilities from adaptive, long-chain attacks. To fill this gap, we introduce DREAM, a framework for systematic evaluation of LLM agents against dynamic, multi-stage attacks. At its core, DREAM uses a Cross-Environment Adversarial Knowledge Graph (CE-AKG) to maintain stateful, cross-domain understanding of vulnerabilities. This graph guides a Contextualized Guided Policy Search (C-GPS) algorithm that dynamically constructs attack chains from a knowledge base of 1,986 atomic actions across 349 distinct digital environments. Our evaluation of 12 leading LLM agents reveals a critical vulnerability: these attack chains succeed in over 70% of cases for most models, showing the power of stateful, cross-environment exploits. Through analysis of these failures, we identify two key weaknesses in current agents: contextual fragility, where safety behaviors fail to transfer across environments, and an inability to track long-term malicious intent. Our findings also show that traditional safety measures, such as initial defense prompts, are largely ineffective against attacks that build context over multiple interactions. To advance agent safety research, we release DREAM as a tool for evaluating vulnerabilities and developing more robust defenses. |
| 2025-12-22 | [Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline](http://arxiv.org/abs/2512.19011v1) | Akshaj Prashanth Rao, Advait Singh et al. | Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.   Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time to completion from approximately 450s to 47s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.   Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications. |
| 2025-12-21 | [CrashChat: A Multimodal Large Language Model for Multitask Traffic Crash Video Analysis](http://arxiv.org/abs/2512.18878v1) | Kaidi Liang, Ke Li et al. | Automating crash video analysis is essential to leverage the growing availability of driving video data for traffic safety research and accountability attribution in autonomous driving. Crash video analysis is a challenging multitask problem due to the complex spatiotemporal dynamics of crash events in video data and the diverse analytical requirements involved. It requires capabilities spanning crash recognition, temporal grounding, and high-level video understanding. Existing models, however, cannot perform all these tasks within a unified framework, and effective training strategies for such models remain underexplored. To fill these gaps, this paper proposes CrashChat, a multimodal large language model (MLLM) for multitask traffic crash analysis, built upon VideoLLaMA3. CrashChat acquires domain-specific knowledge through instruction fine-tuning and employs a novel multitask learning strategy based on task decoupling and grouping, which maximizes the benefit of joint learning within and across task groups while mitigating negative transfer. Numerical experiments on consolidated public datasets demonstrate that CrashChat consistently outperforms existing MLLMs across model scales and traditional vision-based methods, achieving state-of-the-art performance. It reaches near-perfect accuracy in crash recognition, a 176\% improvement in crash localization, and a 40\% improvement in the more challenging pre-crash localization. Compared to general MLLMs, it substantially enhances textual accuracy and content coverage in crash description and reasoning tasks, with 0.18-0.41 increases in BLEU scores and 0.18-0.42 increases in ROUGE scores. Beyond its strong performance, CrashChat is a convenient, end-to-end analytical tool ready for practical implementation. The dataset and implementation code for CrashChat are available at https://github.com/Liangkd/CrashChat. |
| 2025-12-21 | [Multimodal Classification Network Guided Trajectory Planning for Four-Wheel Independent Steering Autonomous Parking Considering Obstacle Attributes](http://arxiv.org/abs/2512.18836v1) | Jingjia Teng, Yang Li et al. | Four-wheel Independent Steering (4WIS) vehicles have attracted increasing attention for their superior maneuverability. Human drivers typically choose to cross or drive over the low-profile obstacles (e.g., plastic bags) to efficiently navigate through narrow spaces, while existing planners neglect obstacle attributes, causing inefficiency or path-finding failures. To address this, we propose a trajectory planning framework integrating the 4WIS hybrid A* and Optimal Control Problem (OCP), in which the hybrid A* provides an initial path to enhance the OCP solution. Specifically, a multimodal classification network is introduced to assess scene complexity (hard/easy task) by fusing image and vehicle state data. For hard tasks, guided points are set to decompose complex tasks into local subtasks, improving the search efficiency of 4WIS hybrid A*. The multiple steering modes of 4WIS vehicles (Ackermann, diagonal, and zero-turn) are also incorporated into node expansion and heuristic designs. Moreover, a hierarchical obstacle handling strategy is designed to guide the node expansion considering obstacle attributes, i.e., 'non-traversable', 'crossable', and 'drive-over' obstacles. It allows crossing or driving over obstacles instead of the 'avoid-only' strategy, greatly enhancing success rates of pathfinding. We also design a logical constraint for the 'drive-over' obstacle by limiting its velocity to ensure safety. Furthermore, to address dynamic obstacles with motion uncertainty, we introduce a probabilistic risk field model, constructing risk-aware driving corridors that serve as linear collision constraints in OCP. Experimental results demonstrate the proposed framework's effectiveness in generating safe, efficient, and smooth trajectories for 4WIS vehicles, especially in constrained environments. |

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



